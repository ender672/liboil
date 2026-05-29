/*
 * Memory regression check for the wrapper's row buffer. Decodes a tall image
 * with a throttled (slower-than-decode) consumer and reports peak
 * finalized-but-unconsumed rows. Unbounded approaches the full fed height; the
 * window stays near its size. Fails if peak exceeds TARGET_FRACTION of the image.
 *
 * Usage: ./jxlmembench [width] [height] [throttle_us]
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

/* Fraction of fed height, not an absolute count: holds across widths and
 * tolerates transient spikes when the cap lifts. Unbounded ~99%, bounded well under. */
#define TARGET_FRACTION 0.50

int main(int argc, char **argv)
{
	int in_w = argc > 1 ? atoi(argv[1]) : 512;
	int in_h = argc > 2 ? atoi(argv[2]) : 8192;
	int throttle_us = argc > 3 ? atoi(argv[3]) : 40;
	int out_w, out_h;
	unsigned char *jxl, *row;
	size_t jxl_size;
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	struct oil_libjxl ol;
	size_t peak, peak_bytes, fed_bytes, fed_h_for_check;
	double t0, t1;
	int y;

	/* Downscale 2x so the fed rect spans nearly the full image height. */
	out_w = in_w / 2;
	out_h = in_h / 2;

	printf("encoding %dx%d gradient JXL...\n", in_w, in_h);
	encode_gradient_jxl(in_w, in_h, &jxl, &jxl_size);
	printf("  codestream: %zu bytes\n", jxl_size);

	dec = open_jxl(jxl, jxl_size, &runner, &info);

	assert(oil_libjxl_init_ex(&ol, dec, &info, out_w, out_h,
		0.0, 0.0, (double)in_w, (double)in_h, OIL_CS_UNKNOWN) == 0);

	printf("decoding to %dx%d, fed rect %dx%d, %d worker threads, "
		"consumer throttle %dus/row\n",
		out_w, out_h, ol.fed_width, ol.fed_height,
		(int)JxlThreadParallelRunnerDefaultNumWorkerThreads(),
		throttle_us);

	/* Pull fed rows directly (not through the scaler) to measure the buffer alone. */
	row = malloc((size_t)ol.fed_width * ol.components);
	assert(row);
	t0 = now_sec();
	for (y = 0; y < ol.fed_height; y++) {
		oil_libjxl_decode_row(&ol, row);
		if (throttle_us > 0)
			usleep(throttle_us);
	}
	t1 = now_sec();
	assert(!ol.error);

	peak = oil_libjxl_peak_buffered_rows(&ol);
	fed_bytes = (size_t)ol.fed_width * ol.components;
	peak_bytes = peak * fed_bytes;
	fed_h_for_check = (size_t)ol.fed_height;

	printf("\n");
	printf("  elapsed:            %.3f s\n", t1 - t0);
	printf("  fed height:         %d rows\n", ol.fed_height);
	printf("  peak buffered rows: %zu  (%.1f%% of fed height)\n",
		peak, 100.0 * peak / ol.fed_height);
	printf("  peak buffer bytes:  %.2f MiB\n",
		peak_bytes / (1024.0 * 1024.0));
	printf("  target:             < %.0f%% of fed height (%zu rows)\n",
		TARGET_FRACTION * 100.0,
		(size_t)(TARGET_FRACTION * ol.fed_height));

	free(row);
	oil_libjxl_free(&ol);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	free(jxl);

	if (peak >= (size_t)(TARGET_FRACTION * fed_h_for_check)) {
		printf("\nRESULT: FAIL - producer raced %zu rows ahead of the "
			"consumer (buffer not bounded)\n", peak);
		return 1;
	}
	printf("\nRESULT: PASS - buffer stayed well within the bound\n");
	return 0;
}
