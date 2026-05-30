/**
 * Copyright (c) 2014-2019 Timothy Elliott
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "oil_jxl_rowbuf.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------- lock-free per-row tile buffer ----------
 *
 * A decoder's image-out callback delivers partial scanline segments (x,y,n)
 * from worker threads out of order. Each is routed into fixed-width tile slots
 * (malloc'd on first touch via CAS) in a lazily-allocated per-row block; the
 * thread that completes a row's last tile coalesces the row and signals the
 * consumer, which walks rows top-to-bottom and releases each. Bookkeeping
 * scales with the in-flight row set, not image height.
 *
 * Scoped to the cropped fed rect [x0,x0+w) x [y0,y0+h): segments are clipped
 * and shifted to crop-local coords, so pixels outside it are never buffered.
 *
 * Reassembly is lock-free (atomics). The two blocking points -- consumer
 * waiting for a row, producer waiting on back-pressure -- go through the
 * caller-supplied oil_jxl_waiter: its lock guards the coordination state below
 * (consumer_waiting/consumer_blocked/window), its ROW channel parks the
 * consumer, its WINDOW channel parks back-pressured producers. */

/* Back-pressure cap: at most `window` finalized-but-unconsumed rows may sit
 * ahead of the consumer, bounding heap use to ~window*row_bytes regardless of
 * height. The window self-tunes (see struct comment), starting at
 * OIL_JXL_WINDOW_START and growing no further than OIL_JXL_WINDOW_MAX_BYTES. */
#define OIL_JXL_WINDOW_START      256
#define OIL_JXL_WINDOW_MAX_BYTES  (64u * 1024 * 1024)

/* OIL_JXL_WINDOW=N pins the window to N rows (N=0 = unbounded), disabling
 * self-tuning. Unset or malformed = adaptive; malformed must not read as 0,
 * which would silently disable back-pressure. */
#define OIL_JXL_WINDOW_ENV "OIL_JXL_WINDOW"

typedef _Atomic(uint8_t *) atomic_ptr;

struct oil_jxl_rowbuf {
	size_t x0, y0;       /* crop origin in full-image coords */
	size_t w, h, bpp;    /* crop dimensions */
	size_t row_bytes;
	size_t tile_w;
	size_t tile_bytes;
	size_t tiles_per_row;
	size_t track_bytes;

	_Atomic(void *)    *track;       /* [h] rowtrack block, lazy */
	_Atomic uint32_t   *tiles_done;  /* [h]; tiles_per_row can exceed uint16_t */
	_Atomic(uint8_t *) *row_buf;     /* [h] finalized scanlines */

	_Atomic int aborted;             /* producer hit a decode error */

	/* Blocking primitive (borrowed). Its lock guards consumer_waiting,
	 * consumer_blocked, and window; its ROW/WINDOW channels park the consumer
	 * and back-pressured producers respectively. */
	const struct oil_jxl_waiter *waiter;

	/* Instrumentation. live_rows: finalized-but-unconsumed rows on the heap;
	 * peak_rows: its high-water mark (the footprint the window caps). */
	_Atomic size_t live_rows;
	_Atomic size_t peak_rows;

	/* Starvation instrumentation. consumer_waits: times the consumer blocked
	 * for an unready row. induced_starvations: the subset where a producer was
	 * paused (parked>0) -- a pause that cost work. window_grows: enlargements.
	 * Healthy steady state: zero induced starvations, stable window. */
	_Atomic size_t consumer_waits;
	_Atomic size_t induced_starvations;
	_Atomic size_t window_grows;

	/* Back-pressure. A worker finalizing row y parks while
	 * y >= consume_pos + window, so finalized rows never run more than `window`
	 * ahead of the consumer; the needed row (y == consume_pos) is always within
	 * the window, so the cap never blocks it.
	 *
	 * Self-tuning: if the consumer blocks while a producer is paused (parked>0)
	 * the window was too small -- grow x2 (monotonic, capped at window_max, so
	 * it converges). Blocking with nothing paused is raw decode lag, not us, so
	 * leave it. consumer_blocked lifts the cap entirely while the consumer is
	 * stalled (safe: a stalled consumer means the buffer drained) -- the
	 * liveness backstop against adversarial delivery order.
	 *
	 * adaptive == 0 pins the window (env override). */
	size_t window;
	size_t window_max;
	int    adaptive;
	_Atomic int parked;         /* producers currently paused on WINDOW */
	_Atomic size_t consume_pos; /* crop-local row the consumer needs next */
	int consumer_waiting;       /* consumer is parked in wait_row */
	int consumer_blocked;       /* cap lifted: consumer starved by a pause */
};

static void rb_lock(struct oil_jxl_rowbuf *s)
{
	s->waiter->lock(s->waiter->opaque);
}
static void rb_unlock(struct oil_jxl_rowbuf *s)
{
	s->waiter->unlock(s->waiter->opaque);
}
static void rb_wait(struct oil_jxl_rowbuf *s, int channel)
{
	s->waiter->wait(s->waiter->opaque, channel);
}
static void rb_wake(struct oil_jxl_rowbuf *s, int channel, int all)
{
	s->waiter->wake(s->waiter->opaque, channel, all);
}

/* One calloc per row, packed: [ atomic_ptr buf[tpr] ][ _Atomic uint16_t
 * fill[tpr] ] -- tile pointers then per-tile fill counts. */
static atomic_ptr *rt_buf(const struct oil_jxl_rowbuf *s, void *rt)
{
	(void)s;
	return (atomic_ptr *)rt;
}
static _Atomic uint16_t *rt_fill(const struct oil_jxl_rowbuf *s, void *rt)
{
	return (_Atomic uint16_t *)((char *)rt
		+ s->tiles_per_row * sizeof(atomic_ptr));
}

static size_t tile_w_of(const struct oil_jxl_rowbuf *s, size_t k)
{
	size_t end = (k + 1) * s->tile_w;
	if (end > s->w) end = s->w;
	return end - k * s->tile_w;
}

static void *get_rowtrack(struct oil_jxl_rowbuf *s, size_t y)
{
	void *rt = atomic_load_explicit(&s->track[y], memory_order_acquire);
	void *expected, *blk;
	if (rt) return rt;
	blk = calloc(1, s->track_bytes);
	if (!blk) abort();
	expected = NULL;
	if (atomic_compare_exchange_strong_explicit(
	        &s->track[y], &expected, blk,
	        memory_order_release, memory_order_acquire))
		return blk;
	free(blk);
	return expected;
}

void oil_jxl_rowbuf_write_segment(struct oil_jxl_rowbuf *s,
                                  size_t x, size_t y, size_t n,
                                  const void *pixels)
{
	const uint8_t *src = pixels;
	size_t seg_lo, seg_hi, crop_hi, tile_lo, tile_hi, k;
	int row_just_completed = 0;
	void *rt;

	if (y < s->y0 || y >= s->y0 + s->h)
		return;

	/* Clip columns to the crop, then shift to crop-local coords. */
	crop_hi = s->x0 + s->w;
	seg_lo  = x > s->x0 ? x : s->x0;
	seg_hi  = (x + n) < crop_hi ? (x + n) : crop_hi;
	if (seg_hi <= seg_lo)
		return;
	src += (seg_lo - x) * s->bpp;
	x = seg_lo - s->x0;
	n = seg_hi - seg_lo;
	y -= s->y0;

	tile_lo = x / s->tile_w;
	tile_hi = (x + n - 1) / s->tile_w;
	rt = get_rowtrack(s, y);
	atomic_ptr       *buf  = rt_buf(s, rt);
	_Atomic uint16_t *fill = rt_fill(s, rt);

	for (k = tile_lo; k <= tile_hi; k++) {
		size_t tx0 = k * s->tile_w;
		size_t tx1 = tx0 + s->tile_w;
		size_t cp0 = x       > tx0 ? x       : tx0;
		size_t cp1 = (x + n) < tx1 ? (x + n) : tx1;
		size_t copy_pixels = cp1 - cp0;
		size_t off_in_tile    = cp0 - tx0;
		size_t off_in_partial = cp0 - x;
		uint8_t *dst;
		uint16_t prev_fill;

		dst = atomic_load_explicit(&buf[k], memory_order_acquire);
		if (!dst) {
			uint8_t *new_tile = malloc(s->tile_bytes);
			uint8_t *expected = NULL;
			if (!new_tile) abort();
			if (atomic_compare_exchange_strong_explicit(
			        &buf[k], &expected, new_tile,
			        memory_order_release, memory_order_acquire)) {
				dst = new_tile;
			} else {
				free(new_tile);
				dst = expected;
			}
		}

		memcpy(dst + off_in_tile * s->bpp,
		       src + off_in_partial * s->bpp,
		       copy_pixels * s->bpp);

		prev_fill = atomic_fetch_add_explicit(
			&fill[k], (uint16_t)copy_pixels,
			memory_order_acq_rel);
		if (prev_fill + copy_pixels == tile_w_of(s, k)) {
			uint32_t prev_done = atomic_fetch_add_explicit(
				&s->tiles_done[y], 1, memory_order_acq_rel);
			if ((size_t)(prev_done + 1) == s->tiles_per_row)
				row_just_completed = 1;
		}
	}

	if (row_just_completed) {
		uint8_t *row;
		size_t kk;

		/* Hold this row while it sits >window rows ahead of the consumer
		 * (unless consumer_blocked). Tiles stay allocated; coalescing and
		 * publication wait until the window admits it. */
		rb_lock(s);
		while (!atomic_load_explicit(&s->aborted, memory_order_acquire)
		    && !s->consumer_blocked
		    && y >= atomic_load_explicit(&s->consume_pos,
		                memory_order_acquire) + s->window) {
			atomic_fetch_add_explicit(&s->parked, 1,
			                           memory_order_relaxed);
			/* Nudge a waiting consumer to re-evaluate and lift the cap --
			 * closes the race where it blocked with nothing yet parked. */
			if (s->consumer_waiting)
				rb_wake(s, OIL_JXL_WAIT_ROW, 0);
			rb_wait(s, OIL_JXL_WAIT_WINDOW);
			atomic_fetch_sub_explicit(&s->parked, 1,
			                           memory_order_relaxed);
		}
		rb_unlock(s);

		/* Aborted: consumer is gone. Skip publishing (rowbuf_destroy frees
		 * the tiles) so a caller that frees without draining unwinds promptly
		 * instead of stranding parked workers in JxlDecoderProcessInput. */
		if (atomic_load_explicit(&s->aborted, memory_order_acquire))
			return;

		row = malloc(s->row_bytes);
		if (!row) abort();
		for (kk = 0; kk < s->tiles_per_row; kk++) {
			uint8_t *tile = atomic_load_explicit(
				&buf[kk], memory_order_acquire);
			memcpy(row + kk * s->tile_w * s->bpp,
			       tile,
			       tile_w_of(s, kk) * s->bpp);
			free(tile);
		}
		atomic_store_explicit(&s->track[y], NULL,
		                       memory_order_relaxed);
		free(rt);

		rb_lock(s);
		/* Account before publishing: the consumer sees the row via the row_buf
		 * release store below, so incrementing first stops it from decrementing
		 * live_rows before we increment (which would underflow and corrupt
		 * peak_rows). row_buf's release/acquire supplies the ordering. */
		{
			size_t cur = atomic_fetch_add_explicit(
				&s->live_rows, 1, memory_order_relaxed) + 1;
			size_t pk = atomic_load_explicit(
				&s->peak_rows, memory_order_relaxed);
			while (cur > pk && !atomic_compare_exchange_weak_explicit(
			        &s->peak_rows, &pk, cur,
			        memory_order_relaxed, memory_order_relaxed))
				;
		}
		atomic_store_explicit(&s->row_buf[y], row,
		                       memory_order_release);
		rb_wake(s, OIL_JXL_WAIT_ROW, 1);
		rb_unlock(s);
	}
}

unsigned char *oil_jxl_rowbuf_try_row(struct oil_jxl_rowbuf *s, size_t y)
{
	return atomic_load_explicit(&s->row_buf[y], memory_order_acquire);
}

/* Returns the finalized scanline for row y, or NULL if the producer aborted
 * before that row completed. */
unsigned char *oil_jxl_rowbuf_wait_row(struct oil_jxl_rowbuf *s, size_t y)
{
	uint8_t *buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_acquire);
	int blocked = 0;
	if (buf) return buf;
	rb_lock(s);
	while (!(buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_acquire))) {
		if (atomic_load_explicit(&s->aborted, memory_order_acquire))
			break;
		blocked = 1;
		s->consumer_waiting = 1;
		/* Lift the cap only if a producer is actually paused -- then our own
		 * back-pressure is what withholds the row ("induced"): release it and
		 * grow the window. Nothing paused = the decoder is just behind, and
		 * lifting would only waste memory. Re-checked each wakeup. */
		if (!s->consumer_blocked
		    && atomic_load_explicit(&s->parked, memory_order_relaxed) > 0) {
			atomic_fetch_add_explicit(&s->induced_starvations, 1,
			        memory_order_relaxed);
			if (s->adaptive && s->window < s->window_max) {
				s->window *= 2;
				if (s->window > s->window_max)
					s->window = s->window_max;
				atomic_fetch_add_explicit(&s->window_grows, 1,
				        memory_order_relaxed);
			}
			s->consumer_blocked = 1;
			rb_wake(s, OIL_JXL_WAIT_WINDOW, 1);
		}
		rb_wait(s, OIL_JXL_WAIT_ROW);
	}
	s->consumer_waiting = 0;
	s->consumer_blocked = 0;
	rb_unlock(s);
	if (blocked)
		atomic_fetch_add_explicit(&s->consumer_waits, 1,
		                           memory_order_relaxed);
	return buf;
}

void oil_jxl_rowbuf_release_row(struct oil_jxl_rowbuf *s, size_t y)
{
	uint8_t *buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_relaxed);
	atomic_store_explicit(&s->row_buf[y], NULL, memory_order_relaxed);
	if (buf)
		atomic_fetch_sub_explicit(&s->live_rows, 1,
		                           memory_order_relaxed);
	free(buf);

	/* Window slid up one slot; wake one parked worker (not all -- that
	 * thundering herd dominates narrow-image cost). A wrong-worker wakeup just
	 * re-parks; consumer_blocked is the progress backstop. Lock only if parked. */
	atomic_store_explicit(&s->consume_pos, y + 1, memory_order_release);
	if (atomic_load_explicit(&s->parked, memory_order_relaxed) > 0) {
		rb_lock(s);
		rb_wake(s, OIL_JXL_WAIT_WINDOW, 0);
		rb_unlock(s);
	}
}

/* Wake all waiters so they observe the abort instead of deadlocking. */
void oil_jxl_rowbuf_abort(struct oil_jxl_rowbuf *s)
{
	rb_lock(s);
	atomic_store_explicit(&s->aborted, 1, memory_order_release);
	rb_wake(s, OIL_JXL_WAIT_ROW, 1);
	rb_wake(s, OIL_JXL_WAIT_WINDOW, 1);
	rb_unlock(s);
}

struct oil_jxl_rowbuf *oil_jxl_rowbuf_create(size_t x0, size_t y0,
                                             size_t w, size_t h,
                                             size_t bpp, size_t tile_w,
                                             const struct oil_jxl_waiter *waiter)
{
	struct oil_jxl_rowbuf *s;
	const char *env;
	size_t row_bytes, max_rows;
	if (tile_w == 0 || tile_w > 65535) return NULL;
	if (!waiter) return NULL;
	s = calloc(1, sizeof(*s));
	if (!s) return NULL;

	s->x0 = x0; s->y0 = y0;
	s->w = w; s->h = h; s->bpp = bpp;
	s->row_bytes     = w * bpp;
	s->tile_w        = tile_w;
	s->tile_bytes    = tile_w * bpp;
	s->tiles_per_row = (w + tile_w - 1) / tile_w;
	s->track_bytes   = s->tiles_per_row
	                 * (sizeof(atomic_ptr) + sizeof(_Atomic uint16_t));
	s->waiter        = waiter;

	s->track      = calloc(h, sizeof(*s->track));
	s->tiles_done = calloc(h, sizeof(*s->tiles_done));
	s->row_buf    = calloc(h, sizeof(*s->row_buf));
	if (!s->track || !s->tiles_done || !s->row_buf) {
		free((void *)s->track);
		free((void *)s->tiles_done);
		free((void *)s->row_buf);
		free(s);
		return NULL;
	}

	/* Adaptive by default (grows on induced starvation up to a byte budget);
	 * env override pins it. */
	row_bytes = s->row_bytes ? s->row_bytes : 1;
	max_rows = OIL_JXL_WINDOW_MAX_BYTES / row_bytes;
	if (max_rows < 1) max_rows = 1;
	if (max_rows > h) max_rows = h;

	env = getenv(OIL_JXL_WINDOW_ENV);
	if (env && *env) {
		char *end;
		long v = strtol(env, &end, 10);
		if (*end != '\0') {
			env = NULL;   /* trailing junk: ignore, fall back to adaptive */
		} else {
			s->adaptive = 0;
			s->window = (v <= 0) ? h : (size_t)v;
			if (s->window > h) s->window = h;
			s->window_max = s->window;
		}
	} else {
		env = NULL;       /* unset or empty */
	}
	if (!env) {
		s->adaptive = 1;
		s->window = OIL_JXL_WINDOW_START;
		if (s->window > h) s->window = h;
		s->window_max = max_rows < s->window ? s->window : max_rows;
	}

	s->parked = 0;
	atomic_store_explicit(&s->consume_pos, 0, memory_order_relaxed);
	s->consumer_waiting = 0;
	s->consumer_blocked = 0;

	return s;
}

void oil_jxl_rowbuf_destroy(struct oil_jxl_rowbuf *s)
{
	size_t i, kk;
	for (i = 0; i < s->h; i++) {
		void *rt = atomic_load_explicit(&s->track[i],
		                                 memory_order_relaxed);
		if (rt) {
			atomic_ptr *buf = rt_buf(s, rt);
			for (kk = 0; kk < s->tiles_per_row; kk++)
				free(atomic_load_explicit(&buf[kk],
				                           memory_order_relaxed));
			free(rt);
		}
		free(atomic_load_explicit(&s->row_buf[i],
		                           memory_order_relaxed));
	}
	free((void *)s->track);
	free((void *)s->tiles_done);
	free((void *)s->row_buf);
	free(s);
}

size_t oil_jxl_rowbuf_peak_rows(const struct oil_jxl_rowbuf *s)
{
	return atomic_load_explicit(&s->peak_rows, memory_order_relaxed);
}

size_t oil_jxl_rowbuf_consumer_waits(const struct oil_jxl_rowbuf *s)
{
	return atomic_load_explicit(&s->consumer_waits, memory_order_relaxed);
}

size_t oil_jxl_rowbuf_induced_starvations(const struct oil_jxl_rowbuf *s)
{
	return atomic_load_explicit(&s->induced_starvations,
	                             memory_order_relaxed);
}

size_t oil_jxl_rowbuf_window_grows(const struct oil_jxl_rowbuf *s)
{
	return atomic_load_explicit(&s->window_grows, memory_order_relaxed);
}

size_t oil_jxl_rowbuf_window(const struct oil_jxl_rowbuf *s)
{
	return s->window;
}
