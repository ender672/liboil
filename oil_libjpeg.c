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

#include "oil_libjpeg.h"
#include <stdlib.h>
#include <string.h>

int oil_libjpeg_init(struct oil_libjpeg *ol,
	struct jpeg_decompress_struct *dinfo, int out_width, int out_height)
{
	return oil_libjpeg_init_ex(ol, dinfo, out_width, out_height,
		0.0, 0.0,
		(double)dinfo->output_width, (double)dinfo->output_height);
}

int oil_libjpeg_init_ex(struct oil_libjpeg *ol,
	struct jpeg_decompress_struct *dinfo, int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height)
{
	int ret, fed_x, fed_y, fed_w, fed_h, buf_w;
	enum oil_colorspace cs;
#ifdef LIBJPEG_TURBO_VERSION
	JDIMENSION crop_x, crop_w;
#else
	int i;
#endif

	ol->dinfo = dinfo;
	ol->inbuf = NULL;
	ol->inbuf_offset = 0;
	ol->fed_width = 0;
	ol->components = dinfo->output_components;

	cs = jpeg_cs_to_oil(dinfo->out_color_space);
	if (cs == OIL_CS_UNKNOWN) {
		return -1;
	}

	if (oil_required_input_rect(dinfo->output_height, dinfo->output_width,
		src_y, src_height, src_x, src_width,
		out_height, out_width,
		&fed_y, &fed_h, &fed_x, &fed_w) < 0) {
		return -1;
	}

	/* turbo's jpeg_crop_scanline can widen fed_x/fed_w to iMCU
	 * boundaries, so the scaler must be initialized with the
	 * post-rounding fed rect. Note: this is a decoder side effect on
	 * dinfo; any subsequent failure here leaves dinfo in cropped state
	 * and the caller must reset it before retrying init. */
#ifdef LIBJPEG_TURBO_VERSION
	crop_x = fed_x;
	crop_w = fed_w;
	jpeg_crop_scanline(dinfo, &crop_x, &crop_w);
	fed_x = crop_x;
	fed_w = crop_w;
	buf_w = fed_w;
#else
	buf_w = dinfo->output_width;
	ol->inbuf_offset = fed_x * dinfo->output_components;
#endif
	ol->fed_width = fed_w;

	ret = oil_scale_init_ex(&ol->os, fed_h, out_height, fed_w, out_width,
		src_y - fed_y, src_height,
		src_x - fed_x, src_width,
		cs);
	if (ret != 0) {
		return ret;
	}

	ol->inbuf = malloc((size_t)buf_w * dinfo->output_components);
	if (!ol->inbuf) {
		oil_scale_free(&ol->os);
		return -2;
	}

#ifdef LIBJPEG_TURBO_VERSION
	if (fed_y > 0) {
		jpeg_skip_scanlines(dinfo, fed_y);
	}
#else
	for (i = 0; i < fed_y; i++) {
		jpeg_read_scanlines(dinfo, &ol->inbuf, 1);
	}
#endif

	return 0;
}

void oil_libjpeg_free(struct oil_libjpeg *ol)
{
	if (ol->inbuf) {
		free(ol->inbuf);
	}
	oil_scale_free(&ol->os);
}

void oil_libjpeg_decode_row(struct oil_libjpeg *ol, unsigned char *dst)
{
#ifdef LIBJPEG_TURBO_VERSION
	jpeg_read_scanlines(ol->dinfo, &dst, 1);
#else
	jpeg_read_scanlines(ol->dinfo, &ol->inbuf, 1);
	memcpy(dst, ol->inbuf + ol->inbuf_offset,
		(size_t)ol->fed_width * ol->components);
#endif
}

void oil_libjpeg_read_scanline(struct oil_libjpeg *ol, unsigned char *outbuf)
{
	while (oil_scale_slots(&ol->os)) {
		jpeg_read_scanlines(ol->dinfo, &ol->inbuf, 1);
		oil_scale_in(&ol->os, ol->inbuf + ol->inbuf_offset);
	}
	oil_scale_out(&ol->os, outbuf);
}

enum oil_colorspace jpeg_cs_to_oil(J_COLOR_SPACE cs)
{
	switch(cs) {
	case JCS_GRAYSCALE:
		return OIL_CS_G;
	case JCS_RGB:
		return OIL_CS_RGB;
	case JCS_CMYK:
		return OIL_CS_CMYK;
#ifdef JCS_EXTENSIONS
	case JCS_EXT_RGBX:
	case JCS_EXT_BGRX:
		return OIL_CS_RGBX;
#endif
	default:
		return OIL_CS_UNKNOWN;
	}
}

J_COLOR_SPACE oil_cs_to_jpeg(enum oil_colorspace cs)
{
	switch(cs) {
	case OIL_CS_G:
		return JCS_GRAYSCALE;
	case OIL_CS_RGB:
	case OIL_CS_RGB_NOGAMMA:
		return JCS_RGB;
	case OIL_CS_CMYK:
		return JCS_CMYK;
	default:
		return JCS_UNKNOWN;
	}
}

