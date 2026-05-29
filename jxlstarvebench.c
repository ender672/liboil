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
#include <jxl/decode.h>
#include <jxl/thread_parallel_runner.h>
#include "oil_resample.h"
#include "oil_libjxl.h"
#include "jxl_testutil.h"

int main(int argc, char **argv)
{
	int in_w = argc > 1 ? atoi(argv[1]) : 512;
	int in_h = argc > 2 ? atoi(argv[2]) : 8192;
	int throttle_us = argc > 3 ? atoi(argv[3]) : 0;
	int out_w = in_w / 2, out_h = in_h / 2;
	unsigned char *jxl, *out;
	size_t jxl_size, waits, induced, grows, window;
	size_t induced_budget;
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	struct oil_libjxl ol;
	int y, fail = 0;

	printf("encoding %dx%d gradient JXL...\n", in_w, in_h);
	encode_gradient_jxl(in_w, in_h, &jxl, &jxl_size);
	printf("  codestream: %zu bytes\n", jxl_size);

	dec = open_jxl(jxl, jxl_size, &runner, &info);

	assert(oil_libjxl_init_ex(&ol, dec, &info, out_w, out_h,
		0.0, 0.0, (double)in_w, (double)in_h, OIL_CS_UNKNOWN) == 0);

	printf("resizing to %dx%d, fed rect %dx%d, %d worker threads, "
		"consumer throttle %dus/out-row\n",
		out_w, out_h, ol.fed_width, ol.fed_height,
		(int)JxlThreadParallelRunnerDefaultNumWorkerThreads(),
		throttle_us);

	out = malloc((size_t)out_w * out_h * ol.components);
	assert(out);
	for (y = 0; y < out_h; y++) {
		oil_libjxl_read_scanline(&ol,
			out + (size_t)y * out_w * ol.components);
		if (throttle_us > 0)
			usleep(throttle_us);
	}
	assert(!ol.error);

	waits   = oil_libjxl_consumer_waits(&ol);
	induced = oil_libjxl_induced_starvations(&ol);
	grows   = oil_libjxl_window_grows(&ol);
	window  = oil_libjxl_window(&ol);

	/* A fixed warmup cost, so the budget is constant -- independent of fed_height. */
	induced_budget = 64;

	printf("\n");
	printf("  fed height:           %d rows\n", ol.fed_height);
	printf("  consumer wait events: %zu  (%.2f%% of fed rows)\n",
		waits, 100.0 * waits / ol.fed_height);
	printf("  induced starvations:  %zu  (budget %zu, must not scale with H)\n",
		induced, induced_budget);
	printf("  window grows:         %zu\n", grows);
	printf("  settled window:       %zu rows\n", window);

	free(out);
	oil_libjxl_free(&ol);
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
