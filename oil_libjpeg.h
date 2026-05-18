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

#ifndef OIL_LIBJPEG_H
#define OIL_LIBJPEG_H

#include <stdio.h>
#include <jpeglib.h>
#include "oil_resample.h"

struct oil_libjpeg {
	struct oil_scale os;
	struct jpeg_decompress_struct *dinfo;
	unsigned char *inbuf;
	int inbuf_offset;
	int fed_width;
	int components;
};

/**
 * Initialize an oil_libjpeg struct.
 * @ol: Pointer to the struct to be initialized.
 * @dinfo: Pointer to a libjpeg decompress struct, with header already read.
 * @out_height: Desired height, in pixels, of the output image.
 * @out_width: Desired width, in pixels, of the output image.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 */
int oil_libjpeg_init(struct oil_libjpeg *ol,
	struct jpeg_decompress_struct *dinfo, int out_width, int out_height);

/**
 * Initialize an oil_libjpeg struct with a sub-pixel source rect.
 *
 * The wrapper computes the required fed input rect (with halo for the
 * Catmull-Rom kernel) via oil_required_input_rect and advances the libjpeg
 * decoder past rows outside the rect. When built against libjpeg-turbo, it
 * also uses jpeg_crop_scanline to restrict IDCT to the MCU columns covering
 * the fed rect.
 *
 * @ol: Pointer to the struct to be initialized.
 * @dinfo: Pointer to a libjpeg decompress struct, with header already read.
 * @out_width, @out_height: Desired output dimensions in pixels.
 * @src_x, @src_y, @src_width, @src_height: Source rect inside the full image,
 *     in source pixels (may be fractional). Must satisfy
 *     0 <= src_x, src_x + src_width <= dinfo->output_width, and the
 *     equivalent for the vertical axis.
 * @cs_override: If OIL_CS_UNKNOWN, the wrapper derives the scaler's
 *     colorspace from dinfo->out_color_space. Otherwise the override is
 *     passed to oil_scale_init_ex; this is how callers select the
 *     no-gamma variants (e.g., OIL_CS_RGB_NOGAMMA) when the decoded
 *     bytes are RGB but should be scaled in the file's native gamma.
 *     The override must have the same OIL_CMP as the derived colorspace.
 *
 * When built against libjpeg-turbo this function invokes jpeg_crop_scanline,
 * which mutates dinfo. On a -2 (memory) failure the decoder is left in
 * cropped state and the caller must reinitialize dinfo before retrying.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 */
int oil_libjpeg_init_ex(struct oil_libjpeg *ol,
	struct jpeg_decompress_struct *dinfo, int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override);

void oil_libjpeg_free(struct oil_libjpeg *ol);

/**
 * Decode the next input row from the JPEG into a caller-supplied buffer.
 *
 * @ol: Initialized wrapper.
 * @dst: Destination buffer of at least ol->fed_width * ol->components bytes.
 *     On success, holds one row of decoded pixels in the wrapper's
 *     colorspace, restricted to the cropped fed rect.
 *
 * Callers driving the scaler themselves (e.g., to use SIMD entry points or
 * to interpose a slot queue between decode and scale) use this in place of
 * the bundled oil_libjpeg_read_scanline.
 */
void oil_libjpeg_decode_row(struct oil_libjpeg *ol, unsigned char *dst);

void oil_libjpeg_read_scanline(struct oil_libjpeg *ol, unsigned char *outbuf);

enum oil_colorspace jpeg_cs_to_oil(J_COLOR_SPACE cs);

J_COLOR_SPACE oil_cs_to_jpeg(enum oil_colorspace cs);

#endif
