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
#include "oil_jxl_rowbuf.h"
#include "oil_jxl_threads.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------- libjxl image-out callbacks ---------- */

static void *jxl_init_cb(void *init_opaque, size_t nt, size_t npp)
{
	(void)nt; (void)npp;
	return init_opaque;  /* forward rowbuf* to run_cb */
}
static void jxl_destroy_cb(void *o) { (void)o; }
static void jxl_run_cb(void *run_opaque, size_t tid,
                       size_t x, size_t y, size_t n, const void *pixels)
{
	(void)tid;
	oil_jxl_rowbuf_write_segment(run_opaque, x, y, n, pixels);
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

	oil_jxl_rowbuf_abort(ol->tb);
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
	ol->waiter = NULL;
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

	/* Efficient blocking primitive for the rowbuf; the rowbuf borrows it, so
	 * it must outlive the rowbuf (freed in oil_libjxl_free, after the rowbuf). */
	ol->waiter = oil_jxl_condvar_waiter_create();
	if (!ol->waiter) {
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -2;
	}

	ol->tb = oil_jxl_rowbuf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256,
		ol->waiter);
	if (!ol->tb) {
		oil_jxl_condvar_waiter_destroy(ol->waiter);
		ol->waiter = NULL;
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -2;
	}

	if (pthread_create(&ol->producer, NULL, jxl_producer, ol) != 0) {
		oil_jxl_rowbuf_destroy(ol->tb);
		ol->tb = NULL;
		oil_jxl_condvar_waiter_destroy(ol->waiter);
		ol->waiter = NULL;
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -3;
	}
	ol->producer_started = 1;

	return 0;
}

size_t oil_libjxl_peak_buffered_rows(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_peak_rows(ol->tb) : 0;
}

size_t oil_libjxl_consumer_waits(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_consumer_waits(ol->tb) : 0;
}

size_t oil_libjxl_induced_starvations(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_induced_starvations(ol->tb) : 0;
}

size_t oil_libjxl_window_grows(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_window_grows(ol->tb) : 0;
}

size_t oil_libjxl_window(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_window(ol->tb) : 0;
}

void oil_libjxl_cancel(struct oil_libjxl *ol)
{
	if (ol->tb)
		oil_jxl_rowbuf_abort(ol->tb);       /* release back-pressure-parked workers */
	if (ol->runner)
		oil_libjxl_runner_cancel(ol->runner);  /* stop issuing work -> ProcessInput unwinds */
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
		oil_jxl_rowbuf_destroy(ol->tb);
	}
	if (ol->waiter) {
		/* After the rowbuf, which borrowed it. */
		oil_jxl_condvar_waiter_destroy(ol->waiter);
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
		oil_jxl_rowbuf_release_row(ol->tb, ol->in_vpos);
		ol->in_vpos++;
		ol->have_row = 0;
	}

	row = oil_jxl_rowbuf_wait_row(ol->tb, ol->in_vpos);
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
