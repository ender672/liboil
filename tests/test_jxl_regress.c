/*
 * Regression tests for two concurrency fixes, each looped to make the race
 * likely rather than relying on one lucky interleaving:
 *
 *  A. peak_rows accounting. live_rows used to be incremented after the row was
 *     published, so a fast consumer could decrement first and underflow it,
 *     latching a huge peak.
 *  B. runner worker startup. A worker not yet running when the first job was
 *     dispatched used to miss it and park forever, deadlocking the dispatcher.
 *
 * Both drive the decode through the Path-B helpers (rowbuf + run_decode), the
 * same composition oil_jxl_resample and imgscale use.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
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
	fprintf(stderr, "FAIL: timed out (%us) -- a decode/runner deadlocked\n",
		secs);
	_exit(1);
	return NULL;
}

struct rd_drive { JxlDecoder *dec; const JxlPixelFormat *fmt;
	struct oil_jxl_rowbuf *rb; };
static void *rd_drive_thread(void *arg)
{
	struct rd_drive *d = arg;
	oil_jxl_run_decode(d->dec, d->fmt, d->rb);
	return NULL;
}

/* Drive a full-image decode of @dec (runner already bound) into a rowbuf sized
 * for an out_w x out_h resize, pull every fed row unthrottled (racing publish
 * against release), and return the rowbuf's peak buffered-row count; *fed_h_out
 * gets the fed height. */
static size_t decode_all_rows(JxlDecoder *dec, const JxlBasicInfo *info,
	int out_w, int out_h, int *fed_h_out)
{
	enum oil_colorspace cs = jxl_cs_to_oil(info);
	int cmp, fed_x, fed_y, fed_w, fed_h;
	struct oil_jxl_waiter *waiter;
	struct oil_jxl_rowbuf *rb;
	struct rd_drive drv;
	pthread_t driver;
	JxlPixelFormat fmt;
	size_t vpos, peak;

	assert(cs != OIL_CS_UNKNOWN);
	cmp = OIL_CMP(cs);
	assert(oil_required_input_rect(info->ysize, info->xsize,
		0.0, (double)info->ysize, 0.0, (double)info->xsize,
		out_h, out_w, &fed_y, &fed_h, &fed_x, &fed_w) == 0);

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
	assert(pthread_create(&driver, NULL, rd_drive_thread, &drv) == 0);

	for (vpos = 0; vpos < (size_t)fed_h; vpos++) {
		unsigned char *row = oil_jxl_rowbuf_wait_row(rb, vpos);
		assert(row);   /* clean decode: every fed row must arrive */
		oil_jxl_rowbuf_release_row(rb, vpos);
	}
	pthread_join(driver, NULL);

	peak = oil_jxl_rowbuf_peak_rows(rb);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	*fed_h_out = fed_h;
	return peak;
}

/* Test A: peak_rows must stay within [1, fed_height]; the underflow latched a
 * huge value. Any runner works -- the fix is in the rowbuf. */
static void test_peak_accounting(unsigned char *jxl, size_t jxl_size,
	int in_w, int in_h, int iters)
{
	int i;

	for (i = 0; i < iters; i++) {
		JxlDecoder *dec;
		void *runner;
		JxlBasicInfo info;
		size_t peak;
		int fed_h;

		dec = open_jxl(jxl, jxl_size, &runner, &info);
		peak = decode_all_rows(dec, &info, in_w / 2, in_h / 2, &fed_h);
		if (peak < 1 || peak > (size_t)fed_h) {
			fprintf(stderr, "FAIL: iter %d peak_buffered_rows=%zu out of "
				"range [1, %d] -- live_rows accounting corrupt\n",
				i, peak, fed_h);
			exit(1);
		}
		JxlDecoderDestroy(dec);
		JxlThreadParallelRunnerDestroy(runner);
	}
	printf("  A: peak accounting stayed in range over %d decodes: ok\n", iters);
}

/* Test B: a fresh oil_jxl_runner per iteration, exercising the first-dispatch
 * startup path repeatedly under the watchdog. */
static void test_runner_startup(unsigned char *jxl, size_t jxl_size,
	int in_w, int in_h, int iters)
{
	int i, fed_h;

	for (i = 0; i < iters; i++) {
		JxlDecoder *dec;
		void *r;
		JxlBasicInfo info;

		r = oil_jxl_runner_create(0);
		assert(r);
		dec = JxlDecoderCreate(NULL);
		assert(dec);
		assert(JxlDecoderSetParallelRunner(dec, oil_jxl_parallel_runner, r)
			== JXL_DEC_SUCCESS);
		assert(JxlDecoderSubscribeEvents(dec,
			JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS);
		JxlDecoderSetInput(dec, jxl, jxl_size);
		JxlDecoderCloseInput(dec);
		assert(JxlDecoderProcessInput(dec) == JXL_DEC_BASIC_INFO);
		assert(JxlDecoderGetBasicInfo(dec, &info) == JXL_DEC_SUCCESS);

		decode_all_rows(dec, &info, in_w / 2, in_h / 2, &fed_h);

		JxlDecoderDestroy(dec);
		oil_jxl_runner_destroy(r);
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
