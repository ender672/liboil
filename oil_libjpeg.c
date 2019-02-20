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

int oil_libjpeg_init(struct oil_libjpeg *ol,
	struct jpeg_decompress_struct *dinfo, int out_width, int out_height)
{
	int ret;
	enum oil_colorspace cs;

	ol->dinfo = dinfo;

	cs = jpeg_cs_to_oil(dinfo->out_color_space);
	if (cs == OIL_CS_UNKNOWN) {
		return -1;
	}

	ol->inbuf = malloc(dinfo->output_width * dinfo->output_components);
	if (!ol->inbuf) {
		return -2;
	}

	ret = oil_scale_init(&ol->os, dinfo->output_height, out_height,
		dinfo->output_width, out_width, cs);
	if (ret!=0) {
		free(ol->inbuf);
		return ret;
	}

	return 0;
}

void oil_libjpeg_free(struct oil_libjpeg *ol)
{
	if (ol->inbuf) {
		free(ol->inbuf);
	}
	oil_scale_free(&ol->os);
}

void oil_libjpeg_read_scanline(struct oil_libjpeg *ol, unsigned char *outbuf)
{
	int i;

	for (i=oil_scale_slots(&ol->os); i>0; i--) {
		jpeg_read_scanlines(ol->dinfo, &ol->inbuf, 1);
		oil_scale_in(&ol->os, ol->inbuf);
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
		return JCS_RGB;
	case OIL_CS_CMYK:
		return JCS_CMYK;
	default:
		return JCS_UNKNOWN;
	}
}

