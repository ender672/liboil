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

#include "oil_libjxl.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <jxl/thread_parallel_runner.h>   /* default worker-count helper */

/* ---------- lock-free per-row tile buffer ----------
 *
 * libjxl's image-out callback delivers partial scanline segments (x,y,n) from
 * worker threads out of order. Each is routed into fixed-width tile slots
 * (malloc'd on first touch via CAS) in a lazily-allocated per-row block; the
 * thread that completes a row's last tile coalesces the row and signals the
 * consumer, which walks rows top-to-bottom and releases each. Bookkeeping
 * scales with the in-flight row set, not image height.
 *
 * Scoped to the cropped fed rect [x0,x0+w) x [y0,y0+h): segments are clipped
 * and shifted to crop-local coords, so pixels outside it are never buffered. */

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

struct oil_jxl_tile_buf {
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

	pthread_mutex_t wait_lock;
	pthread_cond_t  cv_row_complete;

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
	_Atomic int parked;         /* producers currently paused on cv_window */
	_Atomic size_t consume_pos; /* crop-local row the consumer needs next */
	int consumer_waiting;       /* consumer is parked in wait_for_row */
	int consumer_blocked;       /* cap lifted: consumer starved by a pause */
	pthread_cond_t cv_window;
};

/* One calloc per row, packed: [ atomic_ptr buf[tpr] ][ _Atomic uint16_t
 * fill[tpr] ] -- tile pointers then per-tile fill counts. */
static atomic_ptr *rt_buf(const struct oil_jxl_tile_buf *s, void *rt)
{
	(void)s;
	return (atomic_ptr *)rt;
}
static _Atomic uint16_t *rt_fill(const struct oil_jxl_tile_buf *s, void *rt)
{
	return (_Atomic uint16_t *)((char *)rt
		+ s->tiles_per_row * sizeof(atomic_ptr));
}

static size_t tile_w_of(const struct oil_jxl_tile_buf *s, size_t k)
{
	size_t end = (k + 1) * s->tile_w;
	if (end > s->w) end = s->w;
	return end - k * s->tile_w;
}

static void *get_rowtrack(struct oil_jxl_tile_buf *s, size_t y)
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

static void tile_buf_partial(struct oil_jxl_tile_buf *s,
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
		pthread_mutex_lock(&s->wait_lock);
		while (!atomic_load_explicit(&s->aborted, memory_order_acquire)
		    && !s->consumer_blocked
		    && y >= atomic_load_explicit(&s->consume_pos,
		                memory_order_acquire) + s->window) {
			atomic_fetch_add_explicit(&s->parked, 1,
			                           memory_order_relaxed);
			/* Nudge a waiting consumer to re-evaluate and lift the cap --
			 * closes the race where it blocked with nothing yet parked. */
			if (s->consumer_waiting)
				pthread_cond_signal(&s->cv_row_complete);
			pthread_cond_wait(&s->cv_window, &s->wait_lock);
			atomic_fetch_sub_explicit(&s->parked, 1,
			                           memory_order_relaxed);
		}
		pthread_mutex_unlock(&s->wait_lock);

		/* Aborted: consumer is gone. Skip publishing (tile_buf_destroy frees
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

		pthread_mutex_lock(&s->wait_lock);
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
		pthread_cond_broadcast(&s->cv_row_complete);
		pthread_mutex_unlock(&s->wait_lock);
	}
}

/* Returns the finalized scanline for row y, or NULL if the producer aborted
 * before that row completed. */
static uint8_t *tile_buf_wait_for_row(struct oil_jxl_tile_buf *s, size_t y)
{
	uint8_t *buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_acquire);
	int blocked = 0;
	if (buf) return buf;
	pthread_mutex_lock(&s->wait_lock);
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
			pthread_cond_broadcast(&s->cv_window);
		}
		pthread_cond_wait(&s->cv_row_complete, &s->wait_lock);
	}
	s->consumer_waiting = 0;
	s->consumer_blocked = 0;
	pthread_mutex_unlock(&s->wait_lock);
	if (blocked)
		atomic_fetch_add_explicit(&s->consumer_waits, 1,
		                           memory_order_relaxed);
	return buf;
}

static void tile_buf_release_row(struct oil_jxl_tile_buf *s, size_t y)
{
	uint8_t *buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_relaxed);
	atomic_store_explicit(&s->row_buf[y], NULL, memory_order_relaxed);
	if (buf)
		atomic_fetch_sub_explicit(&s->live_rows, 1,
		                           memory_order_relaxed);
	free(buf);

	/* Window slid up one slot; wake one parked worker (not a broadcast -- that
	 * thundering herd dominates narrow-image cost). A wrong-worker wakeup just
	 * re-parks; consumer_blocked is the progress backstop. Lock only if parked. */
	atomic_store_explicit(&s->consume_pos, y + 1, memory_order_release);
	if (atomic_load_explicit(&s->parked, memory_order_relaxed) > 0) {
		pthread_mutex_lock(&s->wait_lock);
		pthread_cond_signal(&s->cv_window);
		pthread_mutex_unlock(&s->wait_lock);
	}
}

/* Wake all waiters so they observe the abort instead of deadlocking. */
static void tile_buf_abort(struct oil_jxl_tile_buf *s)
{
	pthread_mutex_lock(&s->wait_lock);
	atomic_store_explicit(&s->aborted, 1, memory_order_release);
	pthread_cond_broadcast(&s->cv_row_complete);
	pthread_cond_broadcast(&s->cv_window);
	pthread_mutex_unlock(&s->wait_lock);
}

static struct oil_jxl_tile_buf *tile_buf_create(size_t x0, size_t y0,
                                                size_t w, size_t h,
                                                size_t bpp, size_t tile_w)
{
	struct oil_jxl_tile_buf *s;
	const char *env;
	size_t row_bytes, max_rows;
	if (tile_w == 0 || tile_w > 65535) return NULL;
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

	pthread_mutex_init(&s->wait_lock, NULL);
	pthread_cond_init(&s->cv_row_complete, NULL);
	pthread_cond_init(&s->cv_window, NULL);
	return s;
}

static void tile_buf_destroy(struct oil_jxl_tile_buf *s)
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
	pthread_cond_destroy(&s->cv_row_complete);
	pthread_cond_destroy(&s->cv_window);
	pthread_mutex_destroy(&s->wait_lock);
	free(s);
}

/* ---------- cancellable parallel runner ----------
 *
 * A fixed-size thread pool implementing JxlParallelRunner that, unlike
 * JxlThreadParallelRunner, checks a cancel flag before each work item so a
 * decode can be abandoned mid-frame: the in-flight runner call returns an error
 * and JxlDecoderProcessInput unwinds instead of finishing the frame. This lets
 * oil_libjxl_cancel/_free drop libjxl's large frame state promptly.
 *
 * Cancel reaches workers spinning in run_job via the flag, and workers parked
 * on back-pressure via tile_buf_abort (oil_libjxl_cancel does both). libjxl
 * calls the runner once per parallel section, so one job runs at a time. */

struct oil_jxl_worker_arg { struct oil_jxl_runner *r; size_t id; };

struct oil_jxl_runner {
	size_t n;                  /* worker count == concurrency told to libjxl */
	pthread_t *workers;
	struct oil_jxl_worker_arg *args;
	size_t n_started;

	pthread_mutex_t lock;
	pthread_cond_t  cv_start;  /* workers wait here for a job */
	pthread_cond_t  cv_done;   /* dispatcher waits here for completion */

	/* Current job; stable while running > 0. */
	JxlParallelRunFunction func;
	void *jxl;
	uint32_t end;
	_Atomic uint32_t cursor;   /* next index to claim */
	size_t running;            /* workers still in the job */
	uint64_t generation;       /* bumped per job so workers detect new work */

	_Atomic int cancel;
	int shutdown;
};

static void run_job(struct oil_jxl_runner *r, size_t tid)
{
	for (;;) {
		uint32_t i;
		if (atomic_load_explicit(&r->cancel, memory_order_acquire))
			return;
		i = atomic_fetch_add_explicit(&r->cursor, 1,
		                               memory_order_relaxed);
		if (i >= r->end)
			return;
		r->func(r->jxl, i, tid);
	}
}

static void *oil_jxl_worker(void *arg)
{
	struct oil_jxl_worker_arg *wa = arg;
	struct oil_jxl_runner *r = wa->r;
	size_t tid = wa->id;
	uint64_t seen;

	pthread_mutex_lock(&r->lock);
	/* Baseline at generation 0, not the current value: a worker that hasn't run
	 * yet when the first job is dispatched would otherwise capture the
	 * already-bumped generation, park, and never run it -- yet the dispatcher
	 * counted it in `running`, so it would wait forever. Startup deadlock. */
	seen = 0;
	for (;;) {
		while (!r->shutdown && r->generation == seen)
			pthread_cond_wait(&r->cv_start, &r->lock);
		if (r->shutdown)
			break;
		seen = r->generation;
		pthread_mutex_unlock(&r->lock);

		run_job(r, tid);

		pthread_mutex_lock(&r->lock);
		if (--r->running == 0)
			pthread_cond_broadcast(&r->cv_done);
	}
	pthread_mutex_unlock(&r->lock);
	return NULL;
}

JxlParallelRetCode oil_libjxl_parallel_runner(void *opaque, void *jxl,
	JxlParallelRunInit init, JxlParallelRunFunction func,
	uint32_t start_range, uint32_t end_range)
{
	struct oil_jxl_runner *r = opaque;
	int rc;

	if (atomic_load_explicit(&r->cancel, memory_order_acquire))
		return -1;
	rc = init(jxl, r->n);
	if (rc != 0)
		return rc;
	if (end_range <= start_range)
		return 0;

	pthread_mutex_lock(&r->lock);
	r->func = func;
	r->jxl = jxl;
	r->end = end_range;
	atomic_store_explicit(&r->cursor, start_range, memory_order_relaxed);
	r->running = r->n;
	r->generation++;
	pthread_cond_broadcast(&r->cv_start);
	while (r->running != 0)
		pthread_cond_wait(&r->cv_done, &r->lock);
	pthread_mutex_unlock(&r->lock);

	return atomic_load_explicit(&r->cancel, memory_order_acquire) ? -1 : 0;
}

void *oil_libjxl_runner_create(size_t num_threads)
{
	struct oil_jxl_runner *r;
	size_t i;

	if (num_threads == 0)
		num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
	if (num_threads < 1)
		num_threads = 1;

	r = calloc(1, sizeof(*r));
	if (!r)
		return NULL;
	r->n = num_threads;
	r->workers = calloc(num_threads, sizeof(*r->workers));
	r->args = calloc(num_threads, sizeof(*r->args));
	if (!r->workers || !r->args) {
		free(r->workers);
		free(r->args);
		free(r);
		return NULL;
	}
	pthread_mutex_init(&r->lock, NULL);
	pthread_cond_init(&r->cv_start, NULL);
	pthread_cond_init(&r->cv_done, NULL);

	for (i = 0; i < num_threads; i++) {
		r->args[i].r = r;
		r->args[i].id = i;
		if (pthread_create(&r->workers[i], NULL, oil_jxl_worker,
		                   &r->args[i]) != 0)
			break;
		r->n_started++;
	}
	if (r->n_started != num_threads) {
		oil_libjxl_runner_destroy(r);
		return NULL;
	}
	return r;
}

void oil_libjxl_runner_destroy(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	size_t i;
	if (!r)
		return;
	pthread_mutex_lock(&r->lock);
	r->shutdown = 1;
	pthread_cond_broadcast(&r->cv_start);
	pthread_mutex_unlock(&r->lock);
	for (i = 0; i < r->n_started; i++)
		pthread_join(r->workers[i], NULL);
	pthread_cond_destroy(&r->cv_start);
	pthread_cond_destroy(&r->cv_done);
	pthread_mutex_destroy(&r->lock);
	free(r->args);
	free(r->workers);
	free(r);
}

void oil_libjxl_runner_reset(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 0, memory_order_release);
}

static void runner_request_cancel(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 1, memory_order_release);
}

/* ---------- libjxl image-out callbacks ---------- */

static void *jxl_init_cb(void *init_opaque, size_t nt, size_t npp)
{
	(void)nt; (void)npp;
	return init_opaque;  /* forward tile_buf* to run_cb */
}
static void jxl_destroy_cb(void *o) { (void)o; }
static void jxl_run_cb(void *run_opaque, size_t tid,
                       size_t x, size_t y, size_t n, const void *pixels)
{
	(void)tid;
	tile_buf_partial(run_opaque, x, y, n, pixels);
}

/* ---------- producer thread ---------- */

static void *jxl_producer(void *arg)
{
	struct oil_libjxl *ol = arg;

	for (;;) {
		JxlDecoderStatus s = JxlDecoderProcessInput(ol->dec);
		if (s == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
			if (JxlDecoderSetMultithreadedImageOutCallback(ol->dec,
			        &ol->fmt, jxl_init_cb, jxl_run_cb,
			        jxl_destroy_cb, ol->tb) != JXL_DEC_SUCCESS)
				break;
			continue;
		}
		if (s == JXL_DEC_FULL_IMAGE) continue;
		if (s == JXL_DEC_SUCCESS) return NULL;
		/* JXL_DEC_ERROR, JXL_DEC_NEED_MORE_INPUT (truncated), or any
		 * unexpected status: decode cannot complete. */
		break;
	}

	tile_buf_abort(ol->tb);
	return NULL;
}

/* ---------- oil wrapper ---------- */

enum oil_colorspace jxl_cs_to_oil(const JxlBasicInfo *info)
{
	int alpha = info->alpha_bits > 0;
	switch (info->num_color_channels) {
	case 1:
		return alpha ? OIL_CS_GA : OIL_CS_G;
	case 3:
		return alpha ? OIL_CS_RGBA : OIL_CS_RGB;
	default:
		return OIL_CS_UNKNOWN;
	}
}

int oil_libjxl_init(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height)
{
	return oil_libjxl_init_ex(ol, dec, info, out_width, out_height,
		0.0, 0.0, (double)info->xsize, (double)info->ysize,
		OIL_CS_UNKNOWN);
}

int oil_libjxl_init_ex(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override)
{
	int ret, cmp;
	int fed_x, fed_y, fed_w, fed_h;
	enum oil_colorspace cs;

	ol->dec = dec;
	ol->tb = NULL;
	ol->runner = NULL;
	ol->producer_started = 0;
	ol->inbuf = NULL;
	ol->in_vpos = 0;
	ol->have_row = 0;
	ol->fed_x = ol->fed_y = ol->fed_width = ol->fed_height = 0;
	ol->img_width = info->xsize;
	ol->img_height = info->ysize;
	ol->components = 0;
	ol->error = 0;

	cs = jxl_cs_to_oil(info);
	if (cs == OIL_CS_UNKNOWN) {
		return -1;
	}
	cmp = OIL_CMP(cs);
	if (cs_override != OIL_CS_UNKNOWN) {
		if (OIL_CMP(cs_override) != cmp) {
			return -1;
		}
		cs = cs_override;
	}

	if (oil_required_input_rect(info->ysize, info->xsize,
		src_y, src_height, src_x, src_width,
		out_height, out_width,
		&fed_y, &fed_h, &fed_x, &fed_w) < 0) {
		return -1;
	}
	ol->fed_x = fed_x;
	ol->fed_y = fed_y;
	ol->fed_width = fed_w;
	ol->fed_height = fed_h;
	ol->components = cmp;

	ret = oil_scale_init_ex(&ol->os, fed_h, out_height, fed_w, out_width,
		src_y - fed_y, src_height,
		src_x - fed_x, src_width,
		cs);
	if (ret != 0) {
		return ret;
	}

	/* Fallback row returned to the scaler if the decode fails partway. */
	ol->inbuf = calloc((size_t)fed_w * cmp, 1);
	if (!ol->inbuf) {
		oil_scale_free(&ol->os);
		return -2;
	}

	ol->fmt.num_channels = cmp;
	ol->fmt.data_type    = JXL_TYPE_UINT8;
	ol->fmt.endianness   = JXL_NATIVE_ENDIAN;
	ol->fmt.align        = 0;

	ol->tb = tile_buf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256);
	if (!ol->tb) {
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -2;
	}

	if (pthread_create(&ol->producer, NULL, jxl_producer, ol) != 0) {
		tile_buf_destroy(ol->tb);
		ol->tb = NULL;
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -3;
	}
	ol->producer_started = 1;

	return 0;
}

size_t oil_libjxl_peak_buffered_rows(const struct oil_libjxl *ol)
{
	if (!ol->tb)
		return 0;
	return atomic_load_explicit(&ol->tb->peak_rows, memory_order_relaxed);
}

size_t oil_libjxl_consumer_waits(const struct oil_libjxl *ol)
{
	if (!ol->tb)
		return 0;
	return atomic_load_explicit(&ol->tb->consumer_waits,
	                             memory_order_relaxed);
}

size_t oil_libjxl_induced_starvations(const struct oil_libjxl *ol)
{
	if (!ol->tb)
		return 0;
	return atomic_load_explicit(&ol->tb->induced_starvations,
	                             memory_order_relaxed);
}

size_t oil_libjxl_window_grows(const struct oil_libjxl *ol)
{
	if (!ol->tb)
		return 0;
	return atomic_load_explicit(&ol->tb->window_grows,
	                             memory_order_relaxed);
}

size_t oil_libjxl_window(const struct oil_libjxl *ol)
{
	return ol->tb ? ol->tb->window : 0;
}

void oil_libjxl_cancel(struct oil_libjxl *ol)
{
	if (ol->tb)
		tile_buf_abort(ol->tb);             /* release back-pressure-parked workers */
	if (ol->runner)
		runner_request_cancel(ol->runner);  /* stop issuing work -> ProcessInput unwinds */
}

void oil_libjxl_free(struct oil_libjxl *ol)
{
	if (ol->producer_started) {
		/* Cancel so the producer can exit and be joined: without it a free
		 * before draining every row would deadlock the join (parked workers
		 * keep ProcessInput from returning). Idempotent on a drained decode. */
		oil_libjxl_cancel(ol);
		pthread_join(ol->producer, NULL);
	}
	if (ol->tb) {
		tile_buf_destroy(ol->tb);
	}
	if (ol->inbuf) {
		free(ol->inbuf);
	}
	oil_scale_free(&ol->os);
}

/* Release the previously checked-out row, then return the next fed scanline
 * (in_vpos counts fed rows from 0). On producer abort, returns a zeroed
 * fallback row and sets ol->error. */
static unsigned char *jxl_next_row(struct oil_libjxl *ol)
{
	uint8_t *row;

	if (ol->have_row) {
		tile_buf_release_row(ol->tb, ol->in_vpos);
		ol->in_vpos++;
		ol->have_row = 0;
	}

	row = tile_buf_wait_for_row(ol->tb, ol->in_vpos);
	if (!row) {
		ol->error = 1;
		return ol->inbuf;
	}
	ol->have_row = 1;
	return row;
}

void oil_libjxl_decode_row(struct oil_libjxl *ol, unsigned char *dst)
{
	unsigned char *row = jxl_next_row(ol);
	memcpy(dst, row, (size_t)ol->fed_width * ol->components);
}

void oil_libjxl_read_scanline(struct oil_libjxl *ol, unsigned char *outbuf)
{
	while (oil_scale_slots(&ol->os)) {
		oil_scale_in(&ol->os, jxl_next_row(ol));
	}
	oil_scale_out(&ol->os, outbuf);
}
