/* SPDX-License-Identifier: MIT */

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
 * and JxlDecoderProcessInput unwinds instead of finishing the frame, so a
 * superseded decode can drop libjxl's large frame state promptly.
 *
 * Cancel reaches workers spinning in run_job via the flag; a worker parked on
 * back-pressure is released by oil_jxl_rowbuf_abort (cancel both to abandon a
 * decode mid-frame). libjxl calls the runner once per parallel section, so one
 * job runs at a time. */

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

JxlParallelRetCode oil_jxl_parallel_runner(void *opaque, void *jxl,
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

void *oil_jxl_runner_create(size_t num_threads)
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
		oil_jxl_runner_destroy(r);
		return NULL;
	}
	return r;
}

void oil_jxl_runner_destroy(void *opaque)
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

void oil_jxl_runner_reset(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 0, memory_order_release);
}

void oil_jxl_runner_cancel(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 1, memory_order_release);
}

/* ---------- condvar-backed oil_jxl_waiter ----------
 *
 * One mutex + two condition variables (ROW, WINDOW) implementing the
 * oil_jxl_waiter contract for oil_jxl_rowbuf. The struct oil_jxl_waiter is the
 * first member so its opaque can point back at the container. */

struct oil_jxl_cv_waiter {
	struct oil_jxl_waiter w;
	pthread_mutex_t mutex;
	pthread_cond_t  cv[OIL_JXL_WAIT_CHANNELS];
};

static void cvw_lock(void *o)
{
	pthread_mutex_lock(&((struct oil_jxl_cv_waiter *)o)->mutex);
}
static void cvw_unlock(void *o)
{
	pthread_mutex_unlock(&((struct oil_jxl_cv_waiter *)o)->mutex);
}
static void cvw_wait(void *o, int channel)
{
	struct oil_jxl_cv_waiter *c = o;
	pthread_cond_wait(&c->cv[channel], &c->mutex);
}
static void cvw_wake(void *o, int channel, int all)
{
	struct oil_jxl_cv_waiter *c = o;
	if (all)
		pthread_cond_broadcast(&c->cv[channel]);
	else
		pthread_cond_signal(&c->cv[channel]);
}

struct oil_jxl_waiter *oil_jxl_condvar_waiter_create(void)
{
	struct oil_jxl_cv_waiter *c = calloc(1, sizeof(*c));
	int i;
	if (!c)
		return NULL;
	/* pthread_mutex_init / pthread_cond_init with default (NULL) attributes
	 * do not allocate and cannot fail on the platforms liboil targets, so
	 * their return codes are not checked; allocation is the only failure
	 * mode this constructor reports (NULL above). */
	pthread_mutex_init(&c->mutex, NULL);
	for (i = 0; i < OIL_JXL_WAIT_CHANNELS; i++)
		pthread_cond_init(&c->cv[i], NULL);
	c->w.lock   = cvw_lock;
	c->w.unlock = cvw_unlock;
	c->w.wait   = cvw_wait;
	c->w.wake   = cvw_wake;
	c->w.opaque = c;
	return &c->w;
}

void oil_jxl_condvar_waiter_destroy(struct oil_jxl_waiter *waiter)
{
	struct oil_jxl_cv_waiter *c;
	int i;
	if (!waiter)
		return;
	c = waiter->opaque;
	pthread_mutex_destroy(&c->mutex);
	for (i = 0; i < OIL_JXL_WAIT_CHANNELS; i++)
		pthread_cond_destroy(&c->cv[i]);
	free(c);
}
