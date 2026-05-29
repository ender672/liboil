/*
 * Cancellation tests (see the numbered cases in main): free-before-drain must
 * not hang, oil_libjxl_cancel must abandon rather than wait out a decode, and a
 * reset runner is reusable. A watchdog turns any deadlock into a failure.
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
#include "oil_libjxl.h"
#include "jxl_testutil.h"

static void *watchdog(void *arg)
{
	unsigned secs = *(unsigned *)arg;
	sleep(secs);
	fprintf(stderr, "FAIL: timed out (%us) -- a cancel/free deadlocked\n",
		secs);
	_exit(1);
	return NULL;
}

/* Decode with runner cr (NULL => an internal default JxlThreadParallelRunner).
 * Consume `consume` fed rows (<0 = all), optionally cancel, then free. Returns
 * wall seconds from first read to freed. */
static double run(unsigned char *jxl, size_t jxl_size, int in_w, int in_h,
	void *cr, int consume, int do_cancel)
{
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	struct oil_libjxl ol;
	unsigned char *row;
	double t0, t1;
	int y;
	JxlParallelRunner runner_fn;

	if (cr) {
		runner = cr;
		runner_fn = oil_libjxl_parallel_runner;
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

	assert(oil_libjxl_init_ex(&ol, dec, &info, in_w / 2, in_h / 2,
		0.0, 0.0, (double)in_w, (double)in_h, OIL_CS_UNKNOWN) == 0);
	ol.runner = cr;  /* NULL for the default runner -> cancel = consumer only */

	row = malloc((size_t)ol.fed_width * ol.components);
	assert(row);
	t0 = now_sec();
	for (y = 0; (consume < 0 || y < consume) && y < ol.fed_height; y++)
		oil_libjxl_decode_row(&ol, row);
	if (do_cancel)
		oil_libjxl_cancel(&ol);
	free(row);
	oil_libjxl_free(&ol);
	t1 = now_sec();

	JxlDecoderDestroy(dec);
	if (cr)
		oil_libjxl_runner_reset(cr);   /* ready for reuse */
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

	/* 1. Default (non-cancellable) runner: free before draining must not
	 *    hang -- it waits out the decode but returns. */
	run(jxl, jxl_size, in_w, in_h, NULL, 8, 0);
	printf("  default runner, free after 8 rows: ok\n");
	run(jxl, jxl_size, in_w, in_h, NULL, 0, 0);
	printf("  default runner, free before reading: ok\n");

	/* 2. Cancellable runner: a full drain establishes the decode cost; an
	 *    early cancel must come back markedly faster (decode abandoned). */
	cr = oil_libjxl_runner_create(0);
	assert(cr);
	t_full = run(jxl, jxl_size, in_w, in_h, cr, -1, 0);   /* full drain */
	printf("  cancellable runner, full decode: %.3f s\n", t_full);
	t_cancel = run(jxl, jxl_size, in_w, in_h, cr, 8, 1);  /* cancel early */
	printf("  cancellable runner, cancel after 8 rows: %.3f s\n", t_cancel);

	/* 3. The runner was reused across both decodes (reset between) -> reuse
	 *    works. And cancel must be a real abandon, not a wait-out. */
	if (t_cancel >= t_full) {
		fprintf(stderr, "FAIL: cancel (%.3f s) not faster than full "
			"decode (%.3f s) -- decode was not abandoned\n",
			t_cancel, t_full);
		return 1;
	}
	printf("  cancel abandoned the decode (%.1fx faster) and runner reuse "
		"works\n", t_full / (t_cancel > 1e-6 ? t_cancel : 1e-6));

	oil_libjxl_runner_destroy(cr);
	free(jxl);
	printf("All cancel tests pass.\n");
	return 0;
}
