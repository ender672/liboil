/*
 * Regression tests for two concurrency fixes, each looped to make the race
 * likely rather than relying on one lucky interleaving:
 *
 *  A. peak_rows accounting. live_rows used to be incremented after the row was
 *     published, so a fast consumer could decrement first and underflow it,
 *     latching a huge peak.
 *  B. runner worker startup. A worker not yet running when the first job was
 *     dispatched used to miss it and park forever, deadlocking the dispatcher.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
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
	fprintf(stderr, "FAIL: timed out (%us) -- a decode/runner deadlocked\n",
		secs);
	_exit(1);
	return NULL;
}

/* Test A: peak_rows must stay within [1, fed_height]; the underflow latched a
 * huge value. Any runner works -- the fix is in the tile buffer. */
static void test_peak_accounting(unsigned char *jxl, size_t jxl_size,
	int in_w, int in_h, int iters)
{
	int i, y;

	for (i = 0; i < iters; i++) {
		JxlDecoder *dec;
		void *runner;
		JxlBasicInfo info;
		struct oil_libjxl ol;
		unsigned char *row;
		size_t peak;

		dec = open_jxl(jxl, jxl_size, &runner, &info);
		assert(oil_libjxl_init_ex(&ol, dec, &info, in_w / 2, in_h / 2,
			0.0, 0.0, (double)in_w, (double)in_h, OIL_CS_UNKNOWN) == 0);

		row = malloc((size_t)ol.fed_width * ol.components);
		assert(row);
		for (y = 0; y < ol.fed_height; y++)   /* unthrottled: race publish vs release */
			oil_libjxl_decode_row(&ol, row);
		assert(!ol.error);

		peak = oil_libjxl_peak_buffered_rows(&ol);
		if (peak < 1 || peak > (size_t)ol.fed_height) {
			fprintf(stderr, "FAIL: iter %d peak_buffered_rows=%zu out of "
				"range [1, %d] -- live_rows accounting corrupt\n",
				i, peak, ol.fed_height);
			exit(1);
		}

		free(row);
		oil_libjxl_free(&ol);
		JxlDecoderDestroy(dec);
		JxlThreadParallelRunnerDestroy(runner);
	}
	printf("  A: peak accounting stayed in range over %d decodes: ok\n", iters);
}

/* Test B: a fresh runner per iteration, exercising the first-dispatch startup
 * path repeatedly under the watchdog. */
static void test_runner_startup(unsigned char *jxl, size_t jxl_size,
	int in_w, int in_h, int iters)
{
	int i, y;

	for (i = 0; i < iters; i++) {
		JxlDecoder *dec;
		void *r;
		JxlBasicInfo info;
		struct oil_libjxl ol;
		unsigned char *row;

		r = oil_libjxl_runner_create(0);
		assert(r);
		dec = JxlDecoderCreate(NULL);
		assert(dec);
		assert(JxlDecoderSetParallelRunner(dec, oil_libjxl_parallel_runner, r)
			== JXL_DEC_SUCCESS);
		assert(JxlDecoderSubscribeEvents(dec,
			JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS);
		JxlDecoderSetInput(dec, jxl, jxl_size);
		JxlDecoderCloseInput(dec);
		assert(JxlDecoderProcessInput(dec) == JXL_DEC_BASIC_INFO);
		assert(JxlDecoderGetBasicInfo(dec, &info) == JXL_DEC_SUCCESS);

		assert(oil_libjxl_init_ex(&ol, dec, &info, in_w / 2, in_h / 2,
			0.0, 0.0, (double)in_w, (double)in_h, OIL_CS_UNKNOWN) == 0);
		ol.runner = r;

		row = malloc((size_t)ol.fed_width * ol.components);
		assert(row);
		for (y = 0; y < ol.fed_height; y++)
			oil_libjxl_decode_row(&ol, row);
		assert(!ol.error);

		free(row);
		oil_libjxl_free(&ol);
		JxlDecoderDestroy(dec);
		oil_libjxl_runner_destroy(r);
	}
	printf("  B: %d fresh-runner decodes completed without startup deadlock: ok\n",
		iters);
}

int main(void)
{
	const int in_w = 200, in_h = 4096;   /* single-tile width, many rows */
	unsigned char *jxl;
	size_t jxl_size;
	unsigned timeout = 90;
	pthread_t wd;

	pthread_create(&wd, NULL, watchdog, &timeout);

	encode_gradient_jxl(in_w, in_h, &jxl, &jxl_size);
	printf("encoded %dx%d gradient: %zu bytes\n", in_w, in_h, jxl_size);

	test_peak_accounting(jxl, jxl_size, in_w, in_h, 150);
	test_runner_startup(jxl, jxl_size, in_w, in_h, 80);

	free(jxl);
	printf("All regression tests pass.\n");
	return 0;
}
