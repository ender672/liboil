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

#ifndef OIL_JXL_WAITER_H
#define OIL_JXL_WAITER_H

/* The reorder buffer (oil_jxl_rowbuf) does all its lock-free work with C11
 * atomics, but two points genuinely have to block: the consumer waiting for a
 * row, and a back-pressured producer waiting for window room. Rather than hard-
 * code a platform threading primitive, the buffer delegates that blocking to a
 * caller-supplied waiter, so the buffer core itself makes no pthread (or any
 * platform) calls. oil_jxl_threads supplies an efficient pthreads condvar
 * implementation; a single-threaded caller can supply a trivial no-op waiter. */

/* Wait channels. The two conditions share one lock (so the consumer's and a
 * producer's starvation-detection reads/writes are mutually exclusive -- they
 * race otherwise) but block on separate condition variables. */
enum oil_jxl_wait_channel {
	OIL_JXL_WAIT_ROW = 0,     /* consumer waits for a finalized row */
	OIL_JXL_WAIT_WINDOW = 1,  /* back-pressured producer waits for room */
	OIL_JXL_WAIT_CHANNELS = 2
};

/**
 * A mutual-exclusion lock guarding the buffer's small coordination state, plus
 * two condition channels keyed by enum oil_jxl_wait_channel.
 *
 * Contract:
 *  - lock/unlock bracket the coordination critical sections.
 *  - wait(channel) is always called with the lock held; it atomically releases
 *    the lock, blocks until a matching wake(), and reacquires the lock before
 *    returning. Callers re-check their predicate in a loop, so spurious wakeups
 *    are permitted.
 *  - wake(channel, all): all=0 may wake just one waiter on the channel (used to
 *    avoid a thundering herd of back-pressured producers); all=1 wakes all.
 *
 * The waiter is borrowed: the buffer never frees it. The caller owns its
 * lifetime and must keep it alive until after oil_jxl_rowbuf_destroy.
 */
struct oil_jxl_waiter {
	void (*lock)(void *opaque);
	void (*unlock)(void *opaque);
	void (*wait)(void *opaque, int channel);
	void (*wake)(void *opaque, int channel, int all);
	void *opaque;
};

#endif
