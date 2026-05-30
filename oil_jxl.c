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

#include "oil_jxl.h"
#include "oil_jxl_rowbuf.h"
#include "oil_jxl_threads.h"
#include <stdlib.h>
#include <pthread.h>

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

/* ---------- decode driver ---------- */

int oil_jxl_run_decode(JxlDecoder *dec, const JxlPixelFormat *fmt,
	struct oil_jxl_rowbuf *rb)
{
	for (;;) {
		JxlDecoderStatus s = JxlDecoderProcessInput(dec);
		if (s == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
			if (JxlDecoderSetMultithreadedImageOutCallback(dec,
			        fmt, jxl_init_cb, jxl_run_cb,
			        jxl_destroy_cb, rb) != JXL_DEC_SUCCESS)
				break;
			continue;
		}
		if (s == JXL_DEC_FULL_IMAGE) continue;
		if (s == JXL_DEC_SUCCESS) return 0;
		/* JXL_DEC_ERROR, JXL_DEC_NEED_MORE_INPUT (truncated), or any
		 * unexpected status: decode cannot complete. */
		break;
	}

	oil_jxl_rowbuf_abort(rb);
	return -1;
}

/* ---------- colorspace ---------- */

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

/* ---------- one-call convenience ----------
 *
 * oil_jxl_resample composes the public helpers -- oil_jxl_rowbuf,
 * oil_jxl_condvar_waiter, oil_jxl_run_decode and oil_scale -- the same way a
 * caller would, spawning one thread to drive the decode while it pulls and
 * scales rows. It is reproducible by hand from those helpers; a caller wanting
 * to own the threading, the runner, or the allocator composes them directly. */

struct oil_jxl_drive_arg {
	JxlDecoder *dec;
	const JxlPixelFormat *fmt;
	struct oil_jxl_rowbuf *rb;
};

static void *oil_jxl_drive_thread(void *arg)
{
	struct oil_jxl_drive_arg *d = arg;
	oil_jxl_run_decode(d->dec, d->fmt, d->rb);
	return NULL;
}

int oil_jxl_resample(JxlDecoder *dec, const JxlBasicInfo *info,
	int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override,
	unsigned char *out, size_t out_stride)
{
	struct oil_scale os;
	struct oil_jxl_rowbuf *rb;
	struct oil_jxl_waiter *waiter;
	struct oil_jxl_drive_arg drv;
	JxlPixelFormat fmt;
	pthread_t driver;
	unsigned char *zero;
	enum oil_colorspace cs;
	int cmp, fed_x, fed_y, fed_w, fed_h, y, ret = 0;
	size_t vpos = 0;

	cs = jxl_cs_to_oil(info);
	if (cs == OIL_CS_UNKNOWN)
		return -1;
	cmp = OIL_CMP(cs);
	if (cs_override != OIL_CS_UNKNOWN) {
		if (OIL_CMP(cs_override) != cmp)
			return -1;
		cs = cs_override;
	}

	if (oil_required_input_rect(info->ysize, info->xsize,
		src_y, src_height, src_x, src_width,
		out_height, out_width,
		&fed_y, &fed_h, &fed_x, &fed_w) < 0)
		return -1;

	if (oil_scale_init_ex(&os, fed_h, out_height, fed_w, out_width,
		src_y - fed_y, src_height, src_x - fed_x, src_width, cs) != 0)
		return -2;

	/* Zero fallback fed to the scaler for any row the decode failed to
	 * produce (so a partial decode still fills the output, with ret < 0). */
	zero = calloc((size_t)fed_w * cmp, 1);
	waiter = oil_jxl_condvar_waiter_create();
	rb = waiter ? oil_jxl_rowbuf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256,
		waiter) : NULL;
	if (!zero || !waiter || !rb) {
		if (rb) oil_jxl_rowbuf_destroy(rb);
		if (waiter) oil_jxl_condvar_waiter_destroy(waiter);
		free(zero);
		oil_scale_free(&os);
		return -2;
	}

	fmt.num_channels = cmp;
	fmt.data_type    = JXL_TYPE_UINT8;
	fmt.endianness   = JXL_NATIVE_ENDIAN;
	fmt.align        = 0;

	drv.dec = dec;
	drv.fmt = &fmt;
	drv.rb = rb;
	if (pthread_create(&driver, NULL, oil_jxl_drive_thread, &drv) != 0) {
		oil_jxl_rowbuf_destroy(rb);
		oil_jxl_condvar_waiter_destroy(waiter);
		free(zero);
		oil_scale_free(&os);
		return -3;
	}

	for (y = 0; y < out_height; y++) {
		while (oil_scale_slots(&os)) {
			unsigned char *row = oil_jxl_rowbuf_wait_row(rb, vpos);
			if (row) {
				oil_scale_in(&os, row);
				oil_jxl_rowbuf_release_row(rb, vpos);
			} else {
				oil_scale_in(&os, zero);   /* decode aborted */
				ret = -1;
			}
			vpos++;
		}
		oil_scale_out(&os, out + (size_t)y * out_stride);
	}

	pthread_join(driver, NULL);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	free(zero);
	oil_scale_free(&os);
	return ret;
}
