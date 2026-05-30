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

#ifndef OIL_JXL_ROWBUF_H
#define OIL_JXL_ROWBUF_H

#include <stddef.h>
#include "oil_jxl_waiter.h"

/**
 * Lock-free-ish reorder buffer for out-of-order partial scanlines.
 *
 * A decoder (e.g. libjxl) delivers partial scanline segments (x,y,n,pixels)
 * from worker threads, out of order. The rowbuf reassembles them into finalized
 * scanlines that a single consumer pulls top-to-bottom, scoped to a crop rect
 * and bounded in heap use by a self-tuning back-pressure window. It is
 * decoder-agnostic: it knows only pixel segments in full-image coordinates plus
 * the bytes-per-pixel.
 *
 * All reassembly is done with C11 atomics; the two genuine blocking points (the
 * consumer waiting for a row, a back-pressured producer waiting for window room)
 * are delegated to a caller-supplied oil_jxl_waiter, so this core makes no
 * platform threading calls of its own.
 *
 * The window self-tunes by default; the OIL_JXL_WINDOW env var pins it (N rows,
 * N=0 = unbounded), disabling self-tuning.
 */
struct oil_jxl_rowbuf;

/**
 * Create a rowbuf scoped to the crop rect [x0,x0+w) x [y0,y0+h) in full-image
 * coordinates, carrying @bpp bytes per pixel and using @tile_w-wide tile slots.
 *
 * @waiter supplies the blocking primitive (see oil_jxl_waiter.h) and is
 * borrowed -- the caller owns it and must keep it alive until after
 * oil_jxl_rowbuf_destroy. oil_jxl_threads provides a pthreads condvar waiter;
 * a single-threaded caller may pass a trivial no-op waiter.
 *
 * Returns NULL on bad arguments (tile_w 0 or > 65535, or @waiter NULL) or
 * allocation failure.
 */
struct oil_jxl_rowbuf *oil_jxl_rowbuf_create(size_t x0, size_t y0,
	size_t w, size_t h, size_t bpp, size_t tile_w,
	const struct oil_jxl_waiter *waiter);

void oil_jxl_rowbuf_destroy(struct oil_jxl_rowbuf *rb);

/**
 * Write a partial scanline segment: @n pixels at full-image (@x,@y) from
 * @pixels (n*bpp bytes). Segments outside the crop are clipped/ignored. Safe to
 * call concurrently from many threads. The thread that completes a row's last
 * tile coalesces and publishes it.
 */
void oil_jxl_rowbuf_write_segment(struct oil_jxl_rowbuf *rb,
	size_t x, size_t y, size_t n, const void *pixels);

/**
 * Non-blocking: the finalized crop-local scanline @y, or NULL if it is not
 * ready yet (or the producer aborted).
 */
unsigned char *oil_jxl_rowbuf_try_row(struct oil_jxl_rowbuf *rb, size_t y);

/**
 * Blocking: the finalized crop-local scanline @y, or NULL if the producer
 * aborted before that row completed.
 */
unsigned char *oil_jxl_rowbuf_wait_row(struct oil_jxl_rowbuf *rb, size_t y);

/**
 * Release the crop-local scanline @y previously obtained from wait_row/try_row,
 * sliding the back-pressure window up so producers may run further ahead.
 */
void oil_jxl_rowbuf_release_row(struct oil_jxl_rowbuf *rb, size_t y);

/**
 * Abort: wake all waiters so they observe the abort instead of deadlocking.
 * wait_row for any not-yet-finalized row then returns NULL.
 */
void oil_jxl_rowbuf_abort(struct oil_jxl_rowbuf *rb);

/* Instrumentation (see oil_libjxl.h for the precise semantics). */
size_t oil_jxl_rowbuf_peak_rows(const struct oil_jxl_rowbuf *rb);
size_t oil_jxl_rowbuf_consumer_waits(const struct oil_jxl_rowbuf *rb);
size_t oil_jxl_rowbuf_induced_starvations(const struct oil_jxl_rowbuf *rb);
size_t oil_jxl_rowbuf_window_grows(const struct oil_jxl_rowbuf *rb);
size_t oil_jxl_rowbuf_window(const struct oil_jxl_rowbuf *rb);

#endif
