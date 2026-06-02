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

#ifndef OIL_JXL_H
#define OIL_JXL_H

#include <stddef.h>
#include <jxl/decode.h>
#include <jxl/codestream_header.h>
#include "oil_resample.h"
#include "oil_jxl_rowbuf.h"
#include "oil_jxl_threads.h"

/*
 * libjxl helpers for liboil.
 *
 * A kit of composable helpers for decoding and resizing a libjxl codestream.
 * One JxlDecoderProcessInput decodes the whole frame, dispatching partial
 * scanlines to workers out of order, so the helpers assemble around that: the
 * reorder buffer oil_jxl_rowbuf (oil_jxl_rowbuf.h), the optional pthreads
 * pieces oil_jxl_runner / oil_jxl_condvar_waiter (oil_jxl_threads.h), the
 * resampler oil_scale (oil_resample.h), and the three declared below.
 * oil_jxl_resample bundles them into a one-call convenience; a caller needing
 * to own the threading, runner, or allocator composes them directly (imgscale.c
 * shows the streaming case).
 *
 * Every entry point assumes the caller owns @dec: create it, bind a runner
 * (stock JxlThreadParallelRunner, oil_jxl_runner, or its own), subscribe
 * JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE, supply the codestream (SetInput +
 * CloseInput), and drive ProcessInput to JXL_DEC_BASIC_INFO. oil expects
 * straight alpha, so for premultiplied alpha also set
 * JxlDecoderSetUnpremultiplyAlpha(dec, JXL_TRUE) first.
 */

/**
 * Map a decoded image's JxlBasicInfo to an oil colorspace (G/GA/RGB/RGBA), or
 * OIL_CS_UNKNOWN if the channel layout is unsupported.
 */
enum oil_colorspace jxl_cs_to_oil(const JxlBasicInfo *info);

/**
 * Run a libjxl decode to completion on the calling thread, feeding finalized
 * scanlines into @rb. Wires the multithreaded image-out callback itself; @fmt
 * is the pixel format the rowbuf was sized for.
 *
 * Because JxlDecoderProcessInput decodes the whole frame in one blocking call,
 * a streaming caller runs this on a thread it owns while another thread pulls
 * rows via oil_jxl_rowbuf_wait_row. On a decode error, truncation, or a decode
 * that completes without an image-out pass (no rows published -- e.g. the
 * caller did not subscribe JXL_DEC_FULL_IMAGE) it aborts @rb (so a blocked
 * consumer is released) and returns nonzero; returns 0 on a complete decode.
 */
int oil_jxl_run_decode(JxlDecoder *dec, const JxlPixelFormat *fmt,
	struct oil_jxl_rowbuf *rb);

/**
 * One-call convenience: decode @dec (driven to JXL_DEC_BASIC_INFO with @info
 * filled) and resize the (possibly sub-pixel) source rect to
 * @out_width x @out_height, writing the result into @out -- rows at @out_stride
 * bytes, each out_width*components.
 *
 * The batteries-included entry point, composed entirely from the helpers above:
 * it creates the reorder buffer + condvar waiter, spawns one thread to drive the
 * decode, and pulls/scales rows on the calling thread. It is the only entry here
 * that spawns a thread for you; a caller wanting to own the threading, the
 * parallel runner, or the memory manager composes those helpers directly.
 *
 * @cs_override: OIL_CS_UNKNOWN derives the colorspace from @info; otherwise it
 *     must share the derived OIL_CMP (this selects the no-gamma variants).
 *
 * Returns 0 on a complete decode+resize; -1 bad argument or partial decode
 * (output past the failure point is zero-filled); -2 allocation failure; -3
 * thread-spawn failure.
 */
int oil_jxl_resample(JxlDecoder *dec, const JxlBasicInfo *info,
	int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override,
	unsigned char *out, size_t out_stride);

#endif
