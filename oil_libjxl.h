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
#include <stddef.h>
#include <jxl/decode.h>
#include <jxl/codestream_header.h>
#include <jxl/parallel_runner.h>
#include "oil_resample.h"
#include "oil_jxl_rowbuf.h"

struct oil_libjxl {
	struct oil_scale os;

	/* Borrowed from the caller; the wrapper's producer thread drives it, so the
	 * caller must not touch @dec until oil_libjxl_free returns. */
	JxlDecoder *dec;
	JxlPixelFormat fmt;

	/* Lock-free per-row buffer the decoder's worker threads coalesce
	 * scanlines into; the wrapper consumes finalized rows top-to-bottom. */
	struct oil_jxl_rowbuf *tb;
	pthread_t producer;
	int producer_started;

	/* Optional oil_libjxl_runner_create handle, set by the caller after init to
	 * enable prompt cancellation; NULL = non-cancellable runner, so cancel stops
	 * only the consumer side and free waits out the decode. */
	void *runner;

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
 * libjxl has no incremental pull API (one JxlDecoderProcessInput decodes the
 * whole frame, dispatching partials to workers out of order), so the wrapper
 * drives the decode on its own producer thread feeding a per-row tile buffer
 * scoped to the fed rect; the caller pulls finalized rows top-to-bottom.
 *
 * Before calling, the caller must have created @dec, bound a parallel runner,
 * subscribed >= JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE, supplied the whole
 * codestream (SetInput + CloseInput), and driven ProcessInput to
 * JXL_DEC_BASIC_INFO with @info filled. oil expects straight alpha, so for
 * associated/premultiplied alpha the caller must also have set
 * JxlDecoderSetUnpremultiplyAlpha(dec, JXL_TRUE) (must precede the first
 * ProcessInput; a no-op otherwise). The wrapper then owns @dec until
 * oil_libjxl_free -- the caller must not touch it meanwhile -- but still
 * destroys @dec and the runner afterwards.
 *
 * @src_*: source rect in (possibly fractional) source pixels, within bounds.
 * @cs_override: OIL_CS_UNKNOWN derives the colorspace from @info; otherwise it
 *     is passed to oil_scale_init_ex (how callers pick no-gamma variants) and
 *     must share the derived OIL_CMP.
 *
 * Returns 0, or -1 (bad arg) / -2 (alloc) / -3 (decoder/thread setup).
 */
int oil_libjxl_init_ex(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override);

/**
 * Join the producer thread and free the wrapper's allocations (not the
 * caller-owned decoder or runner). If ol->runner is set, cancels the decode
 * first so free returns promptly instead of waiting out the frame.
 */
void oil_libjxl_free(struct oil_libjxl *ol);

/**
 * Cancellable parallel runner for libjxl. libjxl offers no way to interrupt the
 * single ProcessInput that decodes a frame; this thread pool checks a cancel
 * flag before each work item so an interactive caller can abandon a superseded
 * decode mid-frame and release libjxl's frame state.
 *
 * Usage: create one, bind it before the pre-init BASIC_INFO decode, then point
 * ol->runner at it after oil_libjxl_init_ex:
 *     void *r = oil_libjxl_runner_create(0);            // 0 = default count
 *     JxlDecoderSetParallelRunner(dec, oil_libjxl_parallel_runner, r);
 *     ... drive to BASIC_INFO, oil_libjxl_init_ex(...) ...
 *     ol.runner = r;
 * Cancel from any thread via oil_libjxl_cancel(&ol). A cancelled runner
 * (including by oil_libjxl_free) needs oil_libjxl_runner_reset before reuse;
 * destroy only when no decode is in flight.
 */
void *oil_libjxl_runner_create(size_t num_threads);
void  oil_libjxl_runner_destroy(void *runner);
void  oil_libjxl_runner_reset(void *runner);
JxlParallelRetCode oil_libjxl_parallel_runner(void *runner, void *jxl,
	JxlParallelRunInit init, JxlParallelRunFunction func,
	uint32_t start_range, uint32_t end_range);

/**
 * Abandon the in-progress decode so the producer can be joined quickly: rows
 * requested afterward read the zeroed fallback with ol->error set. With
 * ol->runner NULL it cancels only the consumer side. Idempotent.
 */
void oil_libjxl_cancel(struct oil_libjxl *ol);

/**
 * Decode the next fed row into @dst (>= ol->fed_width * ol->components bytes),
 * for callers driving the scaler themselves instead of oil_libjxl_read_scanline
 * (e.g. SIMD entry points, or a slot queue between decode and scale). On decode
 * failure ol->error is set and the row is zero-filled.
 */
void oil_libjxl_decode_row(struct oil_libjxl *ol, unsigned char *dst);

void oil_libjxl_read_scanline(struct oil_libjxl *ol, unsigned char *outbuf);

/**
 * High-water mark of finalized-but-unconsumed rows the wrapper buffered: its
 * peak heap footprint beyond libjxl's, in rows (x fed_width*components = bytes).
 */
size_t oil_libjxl_peak_buffered_rows(const struct oil_libjxl *ol);

/**
 * Starvation / window self-tuning instrumentation:
 *   consumer_waits      - times the resize thread blocked on an unready row
 *                         (many are legitimate decode lag).
 *   induced_starvations - the subset where a producer was paused by
 *                         back-pressure; should settle at zero, else thrashing.
 *   window_grows        - window enlargements.
 *   window              - current (settled) window size.
 */
size_t oil_libjxl_consumer_waits(const struct oil_libjxl *ol);
size_t oil_libjxl_induced_starvations(const struct oil_libjxl *ol);
size_t oil_libjxl_window_grows(const struct oil_libjxl *ol);
size_t oil_libjxl_window(const struct oil_libjxl *ol);

enum oil_colorspace jxl_cs_to_oil(const JxlBasicInfo *info);

#endif
