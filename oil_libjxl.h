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

#ifndef OIL_LIBJXL_H
#define OIL_LIBJXL_H

#include <pthread.h>
#include <jxl/decode.h>
#include <jxl/codestream_header.h>
#include "oil_resample.h"

/* Defined privately in oil_libjxl.c. */
struct oil_jxl_tile_buf;

struct oil_libjxl {
	struct oil_scale os;

	/* Decoder borrowed from the caller (see oil_libjxl_init_ex). The
	 * wrapper drives JxlDecoderProcessInput on its own producer thread;
	 * the caller still owns the decoder and parallel runner and must not
	 * touch the decoder until oil_libjxl_free returns. */
	JxlDecoder *dec;
	JxlPixelFormat fmt;

	/* Lock-free per-row buffer the decoder's worker threads coalesce
	 * scanlines into; the wrapper consumes finalized rows top-to-bottom. */
	struct oil_jxl_tile_buf *tb;
	pthread_t producer;
	int producer_started;

	unsigned char *inbuf;  /* fed_width*components fallback row on error */
	int in_vpos;           /* next tile-buffer row to consume */
	int have_row;          /* a row is currently checked out from the tb */

	int fed_x;
	int fed_y;
	int fed_width;
	int fed_height;
	int img_width;
	int img_height;
	int components;
	int error;             /* set if the producer thread reported a decode
	                        * failure; rows served afterward are zeroed */
};

/**
 * Initialize an oil_libjxl struct.
 * @ol: Pointer to the struct to be initialized.
 * @dec: A JxlDecoder, prepared as described in oil_libjxl_init_ex.
 * @info: Basic info already read from @dec.
 * @out_width, @out_height: Desired output dimensions in pixels.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 * Returns -3 if a decoder/thread setup call failed.
 */
int oil_libjxl_init(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height);

/**
 * Initialize an oil_libjxl struct with a sub-pixel source rect.
 *
 * libjxl has no incremental pull API: a single JxlDecoderProcessInput call
 * decodes the whole frame, dispatching partial scanlines to worker threads in
 * arbitrary order. The wrapper therefore runs the decode on its own producer
 * thread that routes those partials into a lock-free per-row tile buffer; the
 * calling thread pulls finalized scanlines top-to-bottom. The wrapper computes
 * the required fed input rect (with halo for the Catmull-Rom kernel) via
 * oil_required_input_rect; the tile buffer is scoped to that rect, so pixels
 * outside it (rows above/below, columns left/right) are neither buffered nor
 * copied even though libjxl still decodes the full frame.
 *
 * The caller must, before calling this function, have:
 *   - created @dec and a JxlThreadParallelRunner and bound them with
 *     JxlDecoderSetParallelRunner,
 *   - subscribed at least JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE,
 *   - supplied the entire codestream via JxlDecoderSetInput +
 *     JxlDecoderCloseInput,
 *   - driven JxlDecoderProcessInput until it returned JXL_DEC_BASIC_INFO and
 *     filled @info via JxlDecoderGetBasicInfo.
 * oil premultiplies alpha internally and therefore expects straight
 * (non-premultiplied) alpha input. For images whose alpha is stored as
 * associated/premultiplied (info->alpha_premultiplied), the caller must call
 * JxlDecoderSetUnpremultiplyAlpha(dec, JXL_TRUE) during the setup above,
 * before the first JxlDecoderProcessInput -- libjxl requires that option to be
 * set before decoding begins, which is already past by the time this function
 * runs. The call is a documented no-op for images without associated alpha.
 * The wrapper takes over the decoder from that point; the caller must not call
 * JxlDecoderProcessInput (or otherwise touch @dec) again until after
 * oil_libjxl_free. Ownership of @dec and the runner stays with the caller,
 * which destroys them after oil_libjxl_free.
 *
 * @src_x, @src_y, @src_width, @src_height: Source rect inside the full image,
 *     in source pixels (may be fractional). Must fit within the image bounds.
 * @cs_override: If OIL_CS_UNKNOWN, the wrapper derives the scaler's colorspace
 *     from @info. Otherwise the override is passed to oil_scale_init_ex; this
 *     is how callers select the no-gamma variants. The override must have the
 *     same OIL_CMP as the derived colorspace.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 * Returns -3 if a decoder/thread setup call failed.
 */
int oil_libjxl_init_ex(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override);

/**
 * Join the producer thread and free heap allocations associated with the
 * wrapper. Does not destroy the caller-owned decoder or parallel runner.
 */
void oil_libjxl_free(struct oil_libjxl *ol);

/**
 * Decode the next input row into a caller-supplied buffer.
 *
 * @ol: Initialized wrapper.
 * @dst: Destination buffer of at least ol->fed_width * ol->components bytes.
 *     On success, holds one row of decoded pixels in the wrapper's colorspace,
 *     restricted to the cropped fed rect.
 *
 * Callers driving the scaler themselves (e.g., to use SIMD entry points or to
 * interpose a slot queue between decode and scale) use this in place of the
 * bundled oil_libjxl_read_scanline. If the producer thread reported a decode
 * failure, ol->error is set and the row is zero-filled.
 */
void oil_libjxl_decode_row(struct oil_libjxl *ol, unsigned char *dst);

void oil_libjxl_read_scanline(struct oil_libjxl *ol, unsigned char *outbuf);

enum oil_colorspace jxl_cs_to_oil(const JxlBasicInfo *info);

#endif
