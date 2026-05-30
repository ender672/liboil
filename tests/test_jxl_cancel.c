/*
 * Cancellation tests (see the numbered cases in main): tearing down before
 * draining must not hang; with the cancellable runner an early teardown must
 * abandon rather than wait out the decode; and a reset runner is reusable. A
 * watchdog turns any deadlock into a failure.
 *
 * Driven through the Path-B helpers: the consumer owns the decode thread, and
 * teardown == oil_jxl_rowbuf_abort (release back-pressure-parked workers) +
 * oil_jxl_runner_cancel (abandon the decode, if a cancellable runner) + join.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <jxl/decode.h>
#include <jxl/thread_parallel_runner.h>
#include "oil_resample.h"
#include "oil_jxl.h"
#include "jxl_testutil.h"

static void *watchdog(void *arg)
{
	unsigned secs = *(unsigned *)arg;
	sleep(secs);
	fprintf(stderr, "FAIL: timed out (%us) -- a cancel/teardown deadlocked\n",
		secs);
	_exit(1);
	return NULL;
}

struct cd_drive { JxlDecoder *dec; const JxlPixelFormat *fmt;
	struct oil_jxl_rowbuf *rb; };
static void *cd_drive_thread(void *arg)
{
	struct cd_drive *d = arg;
	oil_jxl_run_decode(d->dec, d->fmt, d->rb);
	return NULL;
}

/* Decode with runner cr (NULL => an internal default JxlThreadParallelRunner,
 * which cannot be cancelled). Pull `consume` fed rows (<0 = all), then tear
 * down. Returns wall seconds from first read to joined. */
static double run(unsigned char *jxl, size_t jxl_size, int in_w, int in_h,
	void *cr, int consume)
{
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	JxlParallelRunner runner_fn;
	enum oil_colorspace cs;
	struct oil_jxl_waiter *waiter;
	struct oil_jxl_rowbuf *rb;
	struct cd_drive drv;
	pthread_t driver;
	JxlPixelFormat fmt;
	int cmp, fed_x, fed_y, fed_w, fed_h;
	size_t vpos;
	double t0, t1;

	if (cr) {
		runner = cr;
		runner_fn = oil_jxl_parallel_runner;
	} else {
		runner = JxlThreadParallelRunnerCreate(NULL,
			JxlThreadParallelRunnerDefaultNumWorkerThreads());
		runner_fn = JxlThreadParallelRunner;
	}
	assert(runner);
	dec = JxlDecoderCreate(NULL);
	assert(dec);
	assert(JxlDecoderSetParallelRunner(dec, runner_fn, runner)
		== JXL_DEC_SUCCESS);
	assert(JxlDecoderSubscribeEvents(dec,
		JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS);
	JxlDecoderSetInput(dec, jxl, jxl_size);
	JxlDecoderCloseInput(dec);
	assert(JxlDecoderProcessInput(dec) == JXL_DEC_BASIC_INFO);
	assert(JxlDecoderGetBasicInfo(dec, &info) == JXL_DEC_SUCCESS);

	cs = jxl_cs_to_oil(&info);
	assert(cs != OIL_CS_UNKNOWN);
	cmp = OIL_CMP(cs);
	assert(oil_required_input_rect(info.ysize, info.xsize,
		0.0, (double)info.ysize, 0.0, (double)info.xsize,
		in_h / 2, in_w / 2, &fed_y, &fed_h, &fed_x, &fed_w) == 0);

	waiter = oil_jxl_condvar_waiter_create();
	assert(waiter);
	rb = oil_jxl_rowbuf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256, waiter);
	assert(rb);
	fmt.num_channels = cmp;
	fmt.data_type = JXL_TYPE_UINT8;
	fmt.endianness = JXL_NATIVE_ENDIAN;
	fmt.align = 0;
	drv.dec = dec;
	drv.fmt = &fmt;
	drv.rb = rb;
	assert(pthread_create(&driver, NULL, cd_drive_thread, &drv) == 0);

	t0 = now_sec();
	for (vpos = 0; (consume < 0 || (int)vpos < consume) &&
	     vpos < (size_t)fed_h; vpos++) {
		unsigned char *row = oil_jxl_rowbuf_wait_row(rb, vpos);
		if (!row)
			break;   /* aborted */
		oil_jxl_rowbuf_release_row(rb, vpos);
	}

	/* Teardown (the old oil_libjxl_free): release any back-pressure-parked
	 * worker, abandon the decode if the runner is cancellable, then join. With
	 * the stock runner the abort only stops buffering -- the decode still runs
	 * to completion (waits out) but the join returns. */
	oil_jxl_rowbuf_abort(rb);
	if (cr)
		oil_jxl_runner_cancel(cr);
	pthread_join(driver, NULL);
	t1 = now_sec();

	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	JxlDecoderDestroy(dec);
	if (cr)
		oil_jxl_runner_reset(cr);   /* ready for reuse */
	else
		JxlThreadParallelRunnerDestroy(runner);
	return t1 - t0;
}

int main(void)
{
	unsigned char *jxl;
	size_t jxl_size;
	const int in_w = 1024, in_h = 8192;
	unsigned timeout = 30;
	pthread_t wd;
	void *cr;
	double t_full, t_cancel;

	pthread_create(&wd, NULL, watchdog, &timeout);
	encode_gradient_jxl(in_w, in_h, &jxl, &jxl_size);

	/* 1. Default (non-cancellable) runner: tearing down before draining must
	 *    not hang -- it waits out the decode but returns. */
	run(jxl, jxl_size, in_w, in_h, NULL, 8);
	printf("  default runner, teardown after 8 rows: ok\n");
	run(jxl, jxl_size, in_w, in_h, NULL, 0);
	printf("  default runner, teardown before reading: ok\n");

	/* 2. Cancellable runner: a full drain establishes the decode cost; an
	 *    early teardown must come back markedly faster (decode abandoned). */
	cr = oil_jxl_runner_create(0);
	assert(cr);
	t_full = run(jxl, jxl_size, in_w, in_h, cr, -1);   /* full drain */
	printf("  cancellable runner, full decode: %.3f s\n", t_full);
	t_cancel = run(jxl, jxl_size, in_w, in_h, cr, 8);  /* abandon early */
	printf("  cancellable runner, abandon after 8 rows: %.3f s\n", t_cancel);

	/* 3. The runner was reused across both decodes (reset between) -> reuse
	 *    works. And the early teardown must be a real abandon, not a wait-out. */
	if (t_cancel >= t_full) {
		fprintf(stderr, "FAIL: cancel (%.3f s) not faster than full "
			"decode (%.3f s) -- decode was not abandoned\n",
			t_cancel, t_full);
		return 1;
	}
	printf("  cancel abandoned the decode (%.1fx faster) and runner reuse "
		"works\n", t_full / (t_cancel > 1e-6 ? t_cancel : 1e-6));

	oil_jxl_runner_destroy(cr);
	free(jxl);
	printf("All cancel tests pass.\n");
	return 0;
}
