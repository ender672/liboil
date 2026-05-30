/*
 * Starvation benchmark: the window caps memory (see jxlmembench) without
 * starving or thrashing the resize thread. Drives the real resize path and
 * checks induced_starvations stays small and does NOT scale with height -- i.e.
 * the window converges.
 *
 * Usage: ./jxlstarvebench [width] [height] [throttle_us]
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

struct drive { JxlDecoder *dec; const JxlPixelFormat *fmt;
	struct oil_jxl_rowbuf *rb; };
static void *drive_thread(void *arg)
{
	struct drive *d = arg;
	oil_jxl_run_decode(d->dec, d->fmt, d->rb);
	return NULL;
}

int main(int argc, char **argv)
{
	int in_w = argc > 1 ? atoi(argv[1]) : 512;
	int in_h = argc > 2 ? atoi(argv[2]) : 8192;
	int throttle_us = argc > 3 ? atoi(argv[3]) : 0;
	int out_w = in_w / 2, out_h = in_h / 2;
	int cmp, fed_x, fed_y, fed_w, fed_h, y, fail = 0;
	unsigned char *jxl, *out;
	size_t jxl_size, waits, induced, grows, window, induced_budget, vpos = 0;
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	enum oil_colorspace cs;
	struct oil_scale os;
	struct oil_jxl_waiter *waiter;
	struct oil_jxl_rowbuf *rb;
	struct drive drv;
	pthread_t driver;
	JxlPixelFormat fmt;

	printf("encoding %dx%d gradient JXL...\n", in_w, in_h);
	encode_gradient_jxl(in_w, in_h, &jxl, &jxl_size);
	printf("  codestream: %zu bytes\n", jxl_size);

	dec = open_jxl(jxl, jxl_size, &runner, &info);
	cs = jxl_cs_to_oil(&info);
	assert(cs != OIL_CS_UNKNOWN);
	cmp = OIL_CMP(cs);
	assert(oil_required_input_rect(info.ysize, info.xsize,
		0.0, (double)info.ysize, 0.0, (double)info.xsize,
		out_h, out_w, &fed_y, &fed_h, &fed_x, &fed_w) == 0);
	assert(oil_scale_init_ex(&os, fed_h, out_h, fed_w, out_w,
		0.0 - fed_y, (double)info.ysize, 0.0 - fed_x, (double)info.xsize,
		cs) == 0);

	waiter = oil_jxl_condvar_waiter_create();
	rb = oil_jxl_rowbuf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256, waiter);
	assert(waiter && rb);
	fmt.num_channels = cmp;
	fmt.data_type = JXL_TYPE_UINT8;
	fmt.endianness = JXL_NATIVE_ENDIAN;
	fmt.align = 0;
	drv.dec = dec;
	drv.fmt = &fmt;
	drv.rb = rb;
	assert(pthread_create(&driver, NULL, drive_thread, &drv) == 0);

	printf("resizing to %dx%d, fed rect %dx%d, %d worker threads, "
		"consumer throttle %dus/out-row\n",
		out_w, out_h, fed_w, fed_h,
		(int)JxlThreadParallelRunnerDefaultNumWorkerThreads(),
		throttle_us);

	out = malloc((size_t)out_w * out_h * cmp);
	assert(out);
	for (y = 0; y < out_h; y++) {
		while (oil_scale_slots(&os)) {
			unsigned char *r = oil_jxl_rowbuf_wait_row(rb, vpos);
			assert(r);
			oil_scale_in(&os, r);
			oil_jxl_rowbuf_release_row(rb, vpos);
			vpos++;
		}
		oil_scale_out(&os, out + (size_t)y * out_w * cmp);
		if (throttle_us > 0)
			usleep(throttle_us);
	}

	waits   = oil_jxl_rowbuf_consumer_waits(rb);
	induced = oil_jxl_rowbuf_induced_starvations(rb);
	grows   = oil_jxl_rowbuf_window_grows(rb);
	window  = oil_jxl_rowbuf_window(rb);

	/* A fixed warmup cost, so the budget is constant -- independent of fed_height. */
	induced_budget = 64;

	printf("\n");
	printf("  fed height:           %d rows\n", fed_h);
	printf("  consumer wait events: %zu  (%.2f%% of fed rows)\n",
		waits, 100.0 * waits / fed_h);
	printf("  induced starvations:  %zu  (budget %zu, must not scale with H)\n",
		induced, induced_budget);
	printf("  window grows:         %zu\n", grows);
	printf("  settled window:       %zu rows\n", window);

	oil_jxl_rowbuf_abort(rb);
	pthread_join(driver, NULL);
	free(out);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	oil_scale_free(&os);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	free(jxl);

	if (induced > induced_budget) {
		printf("\nRESULT: FAIL - %zu induced starvations exceeds the "
			"constant budget; window is thrashing, not converging\n",
			induced);
		fail = 1;
	}
	if (!fail)
		printf("\nRESULT: PASS - resize thread not starved by our pauses; "
			"window converged (%zu growths)\n", grows);
	return fail;
}
