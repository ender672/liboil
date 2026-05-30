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

#include "oil_jxl_threads.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <jxl/thread_parallel_runner.h>   /* default worker-count helper */

/* ---------- cancellable parallel runner ----------
 *
 * A fixed-size thread pool implementing JxlParallelRunner that, unlike
 * JxlThreadParallelRunner, checks a cancel flag before each work item so a
 * decode can be abandoned mid-frame: the in-flight runner call returns an error
 * and JxlDecoderProcessInput unwinds instead of finishing the frame. This lets
 * oil_libjxl_cancel/_free drop libjxl's large frame state promptly.
 *
 * Cancel reaches workers spinning in run_job via the flag, and workers parked
 * on back-pressure via oil_jxl_rowbuf_abort (oil_libjxl_cancel does both). libjxl
 * calls the runner once per parallel section, so one job runs at a time. */

struct oil_jxl_worker_arg { struct oil_jxl_runner *r; size_t id; };

struct oil_jxl_runner {
	size_t n;                  /* worker count == concurrency told to libjxl */
	pthread_t *workers;
	struct oil_jxl_worker_arg *args;
	size_t n_started;

	pthread_mutex_t lock;
	pthread_cond_t  cv_start;  /* workers wait here for a job */
	pthread_cond_t  cv_done;   /* dispatcher waits here for completion */

	/* Current job; stable while running > 0. */
	JxlParallelRunFunction func;
	void *jxl;
	uint32_t end;
	_Atomic uint32_t cursor;   /* next index to claim */
	size_t running;            /* workers still in the job */
	uint64_t generation;       /* bumped per job so workers detect new work */

	_Atomic int cancel;
	int shutdown;
};

static void run_job(struct oil_jxl_runner *r, size_t tid)
{
	for (;;) {
		uint32_t i;
		if (atomic_load_explicit(&r->cancel, memory_order_acquire))
			return;
		i = atomic_fetch_add_explicit(&r->cursor, 1,
		                               memory_order_relaxed);
		if (i >= r->end)
			return;
		r->func(r->jxl, i, tid);
	}
}

static void *oil_jxl_worker(void *arg)
{
	struct oil_jxl_worker_arg *wa = arg;
	struct oil_jxl_runner *r = wa->r;
	size_t tid = wa->id;
	uint64_t seen;

	pthread_mutex_lock(&r->lock);
	/* Baseline at generation 0, not the current value: a worker that hasn't run
	 * yet when the first job is dispatched would otherwise capture the
	 * already-bumped generation, park, and never run it -- yet the dispatcher
	 * counted it in `running`, so it would wait forever. Startup deadlock. */
	seen = 0;
	for (;;) {
		while (!r->shutdown && r->generation == seen)
			pthread_cond_wait(&r->cv_start, &r->lock);
		if (r->shutdown)
			break;
		seen = r->generation;
		pthread_mutex_unlock(&r->lock);

		run_job(r, tid);

		pthread_mutex_lock(&r->lock);
		if (--r->running == 0)
			pthread_cond_broadcast(&r->cv_done);
	}
	pthread_mutex_unlock(&r->lock);
	return NULL;
}

JxlParallelRetCode oil_libjxl_parallel_runner(void *opaque, void *jxl,
	JxlParallelRunInit init, JxlParallelRunFunction func,
	uint32_t start_range, uint32_t end_range)
{
	struct oil_jxl_runner *r = opaque;
	int rc;

	if (atomic_load_explicit(&r->cancel, memory_order_acquire))
		return -1;
	rc = init(jxl, r->n);
	if (rc != 0)
		return rc;
	if (end_range <= start_range)
		return 0;

	pthread_mutex_lock(&r->lock);
	r->func = func;
	r->jxl = jxl;
	r->end = end_range;
	atomic_store_explicit(&r->cursor, start_range, memory_order_relaxed);
	r->running = r->n;
	r->generation++;
	pthread_cond_broadcast(&r->cv_start);
	while (r->running != 0)
		pthread_cond_wait(&r->cv_done, &r->lock);
	pthread_mutex_unlock(&r->lock);

	return atomic_load_explicit(&r->cancel, memory_order_acquire) ? -1 : 0;
}

void *oil_libjxl_runner_create(size_t num_threads)
{
	struct oil_jxl_runner *r;
	size_t i;

	if (num_threads == 0)
		num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads();
	if (num_threads < 1)
		num_threads = 1;

	r = calloc(1, sizeof(*r));
	if (!r)
		return NULL;
	r->n = num_threads;
	r->workers = calloc(num_threads, sizeof(*r->workers));
	r->args = calloc(num_threads, sizeof(*r->args));
	if (!r->workers || !r->args) {
		free(r->workers);
		free(r->args);
		free(r);
		return NULL;
	}
	pthread_mutex_init(&r->lock, NULL);
	pthread_cond_init(&r->cv_start, NULL);
	pthread_cond_init(&r->cv_done, NULL);

	for (i = 0; i < num_threads; i++) {
		r->args[i].r = r;
		r->args[i].id = i;
		if (pthread_create(&r->workers[i], NULL, oil_jxl_worker,
		                   &r->args[i]) != 0)
			break;
		r->n_started++;
	}
	if (r->n_started != num_threads) {
		oil_libjxl_runner_destroy(r);
		return NULL;
	}
	return r;
}

void oil_libjxl_runner_destroy(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	size_t i;
	if (!r)
		return;
	pthread_mutex_lock(&r->lock);
	r->shutdown = 1;
	pthread_cond_broadcast(&r->cv_start);
	pthread_mutex_unlock(&r->lock);
	for (i = 0; i < r->n_started; i++)
		pthread_join(r->workers[i], NULL);
	pthread_cond_destroy(&r->cv_start);
	pthread_cond_destroy(&r->cv_done);
	pthread_mutex_destroy(&r->lock);
	free(r->args);
	free(r->workers);
	free(r);
}

void oil_libjxl_runner_reset(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 0, memory_order_release);
}

void oil_libjxl_runner_cancel(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 1, memory_order_release);
}
