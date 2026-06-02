/* SPDX-License-Identifier: MIT */

#ifndef OIL_JXL_THREADS_H
#define OIL_JXL_THREADS_H

#include <stddef.h>
#include <stdint.h>
#include <jxl/parallel_runner.h>
#include "oil_jxl_waiter.h"

/**
 * Optional pthreads helpers for driving libjxl. This is the only part of the
 * libjxl integration that depends on pthreads; a consumer who supplies their
 * own JxlParallelRunner (e.g. the stock JxlThreadParallelRunner, or their own
 * executor) need not use it.
 *
 * Cancellable parallel runner for libjxl. libjxl offers no way to interrupt the
 * single ProcessInput that decodes a frame; this thread pool checks a cancel
 * flag before each work item so an interactive caller can abandon a superseded
 * decode mid-frame and release libjxl's frame state.
 *
 * Usage: create one and bind it to the decoder before the BASIC_INFO decode:
 *     void *r = oil_jxl_runner_create(0);            // 0 = default count
 *     JxlDecoderSetParallelRunner(dec, oil_jxl_parallel_runner, r);
 *     ... drive to BASIC_INFO, then oil_jxl_run_decode(dec, &fmt, rb) ...
 * Cancel from any thread via oil_jxl_runner_cancel(r), pairing it with
 * oil_jxl_rowbuf_abort to also release a back-pressure-parked worker. A
 * cancelled runner needs oil_jxl_runner_reset before reuse; destroy only when
 * no decode is in flight.
 */
void *oil_jxl_runner_create(size_t num_threads);
void  oil_jxl_runner_destroy(void *runner);
void  oil_jxl_runner_reset(void *runner);
JxlParallelRetCode oil_jxl_parallel_runner(void *runner, void *jxl,
	JxlParallelRunInit init, JxlParallelRunFunction func,
	uint32_t start_range, uint32_t end_range);

/**
 * Request cancellation of @runner: in-flight and subsequent runner calls return
 * an error so JxlDecoderProcessInput unwinds. Safe from any thread; cleared by
 * oil_jxl_runner_reset. NULL is a no-op.
 */
void oil_jxl_runner_cancel(void *runner);

/**
 * An efficient oil_jxl_waiter backed by a pthreads mutex + two condition
 * variables, for use with oil_jxl_rowbuf_create. Returns NULL on allocation
 * failure. The returned waiter is owned by the caller and must outlive the
 * rowbuf using it; free it with oil_jxl_condvar_waiter_destroy.
 */
struct oil_jxl_waiter *oil_jxl_condvar_waiter_create(void);
void oil_jxl_condvar_waiter_destroy(struct oil_jxl_waiter *waiter);

#endif
