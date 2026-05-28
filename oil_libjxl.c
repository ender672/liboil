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

/* ---------- lock-free per-row tile buffer ----------
 *
 * libjxl's image-out callback delivers partial scanline segments (x, y, n)
 * from arbitrary worker threads in no particular order. Each segment is routed
 * into fixed-width tile slots (malloc'd on first touch via CAS) inside a
 * per-row tracking block that is itself lazily allocated. The thread whose
 * write completes a row's last tile coalesces that row's tiles into one
 * contiguous scanline and signals waiters; the consumer walks rows
 * top-to-bottom, consumes each scanline, and releases it. Bookkeeping memory
 * scales with the in-flight row set, not image height. */

typedef _Atomic(uint8_t *) atomic_ptr;

struct oil_jxl_tile_buf {
	size_t w, h, bpp;
	size_t row_bytes;
	size_t tile_w;
	size_t tile_bytes;
	size_t tiles_per_row;
	size_t track_bytes;

	_Atomic(void *)    *track;       /* [h] rowtrack block, lazy */
	_Atomic uint16_t   *tiles_done;  /* [h] */
	_Atomic(uint8_t *) *row_buf;     /* [h] finalized scanlines */

	_Atomic int aborted;             /* producer hit a decode error */

	pthread_mutex_t wait_lock;
	pthread_cond_t  cv_row_complete;
};

/* A rowtrack block packs tiles_per_row tile pointers followed by
 * tiles_per_row fill counters in one calloc:
 *   [ atomic_ptr buf[tpr] ][ _Atomic uint16_t fill[tpr] ] */
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
	size_t tile_lo = x / s->tile_w;
	size_t tile_hi = (x + n - 1) / s->tile_w;
	size_t k;
	int row_just_completed = 0;
	void *rt = get_rowtrack(s, y);
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
			uint16_t prev_done = atomic_fetch_add_explicit(
				&s->tiles_done[y], 1, memory_order_acq_rel);
			if ((size_t)(prev_done + 1) == s->tiles_per_row)
				row_just_completed = 1;
		}
	}

	if (row_just_completed) {
		uint8_t *row = malloc(s->row_bytes);
		size_t kk;
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
	if (buf) return buf;
	pthread_mutex_lock(&s->wait_lock);
	while (!(buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_acquire))) {
		if (atomic_load_explicit(&s->aborted, memory_order_acquire))
			break;
		pthread_cond_wait(&s->cv_row_complete, &s->wait_lock);
	}
	pthread_mutex_unlock(&s->wait_lock);
	return buf;
}

static void tile_buf_release_row(struct oil_jxl_tile_buf *s, size_t y)
{
	uint8_t *buf = atomic_load_explicit(&s->row_buf[y],
	                                     memory_order_relaxed);
	atomic_store_explicit(&s->row_buf[y], NULL, memory_order_relaxed);
	free(buf);
}

/* Wake every waiter so consumers blocked on a row that will never complete
 * can observe the failure instead of deadlocking. */
static void tile_buf_abort(struct oil_jxl_tile_buf *s)
{
	pthread_mutex_lock(&s->wait_lock);
	atomic_store_explicit(&s->aborted, 1, memory_order_release);
	pthread_cond_broadcast(&s->cv_row_complete);
	pthread_mutex_unlock(&s->wait_lock);
}

static struct oil_jxl_tile_buf *tile_buf_create(size_t w, size_t h,
                                                size_t bpp, size_t tile_w)
{
	struct oil_jxl_tile_buf *s;
	if (tile_w == 0 || tile_w > 65535) return NULL;
	s = calloc(1, sizeof(*s));
	if (!s) return NULL;

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

	pthread_mutex_init(&s->wait_lock, NULL);
	pthread_cond_init(&s->cv_row_complete, NULL);
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
	pthread_mutex_destroy(&s->wait_lock);
	free(s);
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
	ol->producer_started = 0;
	ol->inbuf = NULL;
	ol->in_vpos = 0;
	ol->have_row = 0;
	ol->fed_x = ol->fed_y = ol->fed_width = ol->fed_height = 0;
	ol->inbuf_offset = 0;
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
	ol->inbuf_offset = fed_x * cmp;

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

	/* The image-out callback delivers full-width rows; the consumer slices
	 * the fed_x..fed_x+fed_w columns out of each one. */
	ol->fmt.num_channels = cmp;
	ol->fmt.data_type    = JXL_TYPE_UINT8;
	ol->fmt.endianness   = JXL_NATIVE_ENDIAN;
	ol->fmt.align        = 0;

	ol->tb = tile_buf_create(info->xsize, info->ysize, cmp, 256);
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

void oil_libjxl_free(struct oil_libjxl *ol)
{
	if (ol->producer_started) {
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

/* Release the row last checked out (if any), skip any rows above the fed rect
 * on first use, then return the next fed scanline at its fed_x column offset.
 * Returns a stable, zeroed fallback row and sets ol->error if the producer
 * aborted before the requested row completed. */
static unsigned char *jxl_next_row(struct oil_libjxl *ol)
{
	uint8_t *row;

	if (ol->have_row) {
		tile_buf_release_row(ol->tb, ol->in_vpos);
		ol->in_vpos++;
		ol->have_row = 0;
	}
	while (ol->in_vpos < ol->fed_y) {
		if (!tile_buf_wait_for_row(ol->tb, ol->in_vpos)) {
			ol->error = 1;
			return ol->inbuf;
		}
		tile_buf_release_row(ol->tb, ol->in_vpos);
		ol->in_vpos++;
	}

	row = tile_buf_wait_for_row(ol->tb, ol->in_vpos);
	if (!row) {
		ol->error = 1;
		return ol->inbuf;
	}
	ol->have_row = 1;
	return row + ol->inbuf_offset;
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
