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

#include "oil_libjxl.h"
#include "oil_jxl_rowbuf.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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

static void runner_request_cancel(void *opaque)
{
	struct oil_jxl_runner *r = opaque;
	if (r)
		atomic_store_explicit(&r->cancel, 1, memory_order_release);
}

/* ---------- libjxl image-out callbacks ---------- */

static void *jxl_init_cb(void *init_opaque, size_t nt, size_t npp)
{
	(void)nt; (void)npp;
	return init_opaque;  /* forward rowbuf* to run_cb */
}
static void jxl_destroy_cb(void *o) { (void)o; }
static void jxl_run_cb(void *run_opaque, size_t tid,
                       size_t x, size_t y, size_t n, const void *pixels)
{
	(void)tid;
	oil_jxl_rowbuf_write_segment(run_opaque, x, y, n, pixels);
}

/* ---------- producer thread ---------- */

static void *jxl_producer(void *arg)
{
	struct oil_libjxl *ol = arg;

	for (;;) {
		JxlDecoderStatus s = JxlDecoderProcessInput(ol->dec);
		if (s == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
			if (JxlDecoderSetMultithreadedImageOutCallback(ol->dec,
			        &ol->fmt, jxl_init_cb, jxl_run_cb,
			        jxl_destroy_cb, ol->tb) != JXL_DEC_SUCCESS)
				break;
			continue;
		}
		if (s == JXL_DEC_FULL_IMAGE) continue;
		if (s == JXL_DEC_SUCCESS) return NULL;
		/* JXL_DEC_ERROR, JXL_DEC_NEED_MORE_INPUT (truncated), or any
		 * unexpected status: decode cannot complete. */
		break;
	}

	oil_jxl_rowbuf_abort(ol->tb);
	return NULL;
}

/* ---------- oil wrapper ---------- */

enum oil_colorspace jxl_cs_to_oil(const JxlBasicInfo *info)
{
	int alpha = info->alpha_bits > 0;
	switch (info->num_color_channels) {
	case 1:
		return alpha ? OIL_CS_GA : OIL_CS_G;
	case 3:
		return alpha ? OIL_CS_RGBA : OIL_CS_RGB;
	default:
		return OIL_CS_UNKNOWN;
	}
}

int oil_libjxl_init(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height)
{
	return oil_libjxl_init_ex(ol, dec, info, out_width, out_height,
		0.0, 0.0, (double)info->xsize, (double)info->ysize,
		OIL_CS_UNKNOWN);
}

int oil_libjxl_init_ex(struct oil_libjxl *ol, JxlDecoder *dec,
	const JxlBasicInfo *info, int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override)
{
	int ret, cmp;
	int fed_x, fed_y, fed_w, fed_h;
	enum oil_colorspace cs;

	ol->dec = dec;
	ol->tb = NULL;
	ol->runner = NULL;
	ol->producer_started = 0;
	ol->inbuf = NULL;
	ol->in_vpos = 0;
	ol->have_row = 0;
	ol->fed_x = ol->fed_y = ol->fed_width = ol->fed_height = 0;
	ol->img_width = info->xsize;
	ol->img_height = info->ysize;
	ol->components = 0;
	ol->error = 0;

	cs = jxl_cs_to_oil(info);
	if (cs == OIL_CS_UNKNOWN) {
		return -1;
	}
	cmp = OIL_CMP(cs);
	if (cs_override != OIL_CS_UNKNOWN) {
		if (OIL_CMP(cs_override) != cmp) {
			return -1;
		}
		cs = cs_override;
	}

	if (oil_required_input_rect(info->ysize, info->xsize,
		src_y, src_height, src_x, src_width,
		out_height, out_width,
		&fed_y, &fed_h, &fed_x, &fed_w) < 0) {
		return -1;
	}
	ol->fed_x = fed_x;
	ol->fed_y = fed_y;
	ol->fed_width = fed_w;
	ol->fed_height = fed_h;
	ol->components = cmp;

	ret = oil_scale_init_ex(&ol->os, fed_h, out_height, fed_w, out_width,
		src_y - fed_y, src_height,
		src_x - fed_x, src_width,
		cs);
	if (ret != 0) {
		return ret;
	}

	/* Fallback row returned to the scaler if the decode fails partway. */
	ol->inbuf = calloc((size_t)fed_w * cmp, 1);
	if (!ol->inbuf) {
		oil_scale_free(&ol->os);
		return -2;
	}

	ol->fmt.num_channels = cmp;
	ol->fmt.data_type    = JXL_TYPE_UINT8;
	ol->fmt.endianness   = JXL_NATIVE_ENDIAN;
	ol->fmt.align        = 0;

	ol->tb = oil_jxl_rowbuf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256);
	if (!ol->tb) {
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -2;
	}

	if (pthread_create(&ol->producer, NULL, jxl_producer, ol) != 0) {
		oil_jxl_rowbuf_destroy(ol->tb);
		ol->tb = NULL;
		free(ol->inbuf);
		oil_scale_free(&ol->os);
		return -3;
	}
	ol->producer_started = 1;

	return 0;
}

size_t oil_libjxl_peak_buffered_rows(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_peak_rows(ol->tb) : 0;
}

size_t oil_libjxl_consumer_waits(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_consumer_waits(ol->tb) : 0;
}

size_t oil_libjxl_induced_starvations(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_induced_starvations(ol->tb) : 0;
}

size_t oil_libjxl_window_grows(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_window_grows(ol->tb) : 0;
}

size_t oil_libjxl_window(const struct oil_libjxl *ol)
{
	return ol->tb ? oil_jxl_rowbuf_window(ol->tb) : 0;
}

void oil_libjxl_cancel(struct oil_libjxl *ol)
{
	if (ol->tb)
		oil_jxl_rowbuf_abort(ol->tb);       /* release back-pressure-parked workers */
	if (ol->runner)
		runner_request_cancel(ol->runner);  /* stop issuing work -> ProcessInput unwinds */
}

void oil_libjxl_free(struct oil_libjxl *ol)
{
	if (ol->producer_started) {
		/* Cancel so the producer can exit and be joined: without it a free
		 * before draining every row would deadlock the join (parked workers
		 * keep ProcessInput from returning). Idempotent on a drained decode. */
		oil_libjxl_cancel(ol);
		pthread_join(ol->producer, NULL);
	}
	if (ol->tb) {
		oil_jxl_rowbuf_destroy(ol->tb);
	}
	if (ol->inbuf) {
		free(ol->inbuf);
	}
	oil_scale_free(&ol->os);
}

/* Release the previously checked-out row, then return the next fed scanline
 * (in_vpos counts fed rows from 0). On producer abort, returns a zeroed
 * fallback row and sets ol->error. */
static unsigned char *jxl_next_row(struct oil_libjxl *ol)
{
	uint8_t *row;

	if (ol->have_row) {
		oil_jxl_rowbuf_release_row(ol->tb, ol->in_vpos);
		ol->in_vpos++;
		ol->have_row = 0;
	}

	row = oil_jxl_rowbuf_wait_row(ol->tb, ol->in_vpos);
	if (!row) {
		ol->error = 1;
		return ol->inbuf;
	}
	ol->have_row = 1;
	return row;
}

void oil_libjxl_decode_row(struct oil_libjxl *ol, unsigned char *dst)
{
	unsigned char *row = jxl_next_row(ol);
	memcpy(dst, row, (size_t)ol->fed_width * ol->components);
}

void oil_libjxl_read_scanline(struct oil_libjxl *ol, unsigned char *outbuf)
{
	while (oil_scale_slots(&ol->os)) {
		oil_scale_in(&ol->os, jxl_next_row(ol));
	}
	oil_scale_out(&ol->os, outbuf);
}
