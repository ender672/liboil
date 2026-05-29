/*
 * Throughput benchmark: full decode+resize, no throttle -- checks
 * back-pressure costs no throughput vs the unbounded buffer. OIL_JXL_WINDOW
 * picks the config: 0 = unbounded, unset = adaptive window, N = pinned to N rows.
 *
 * Usage: ./jxlperfbench [in_w] [in_h] [out_w] [out_h] [iters]
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jxl/decode.h>
#include <jxl/thread_parallel_runner.h>
#include "oil_resample.h"
#include "oil_libjxl.h"
#include "jxl_testutil.h"

/* One decode+resize pass; returns wall seconds, reports buffer stats. */
static double run_once(unsigned char *jxl, size_t jxl_size,
	int in_w, int in_h, int out_w, int out_h,
	size_t *peak, size_t *window, size_t *induced)
{
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	struct oil_libjxl ol;
	unsigned char *out;
	double t0, t1;
	int y;

	dec = open_jxl(jxl, jxl_size, &runner, &info);

	t0 = now_sec();
	assert(oil_libjxl_init_ex(&ol, dec, &info, out_w, out_h,
		0.0, 0.0, (double)in_w, (double)in_h, OIL_CS_UNKNOWN) == 0);
	out = malloc((size_t)out_w * out_h * ol.components);
	assert(out);
	for (y = 0; y < out_h; y++)
		oil_libjxl_read_scanline(&ol,
			out + (size_t)y * out_w * ol.components);
	t1 = now_sec();
	assert(!ol.error);

	*peak = oil_libjxl_peak_buffered_rows(&ol);
	*window = oil_libjxl_window(&ol);
	*induced = oil_libjxl_induced_starvations(&ol);

	free(out);
	oil_libjxl_free(&ol);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	return t1 - t0;
}

int main(int argc, char **argv)
{
	int in_w  = argc > 1 ? atoi(argv[1]) : 4096;
	int in_h  = argc > 2 ? atoi(argv[2]) : 4096;
	int out_w = argc > 3 ? atoi(argv[3]) : 1024;
	int out_h = argc > 4 ? atoi(argv[4]) : 1024;
	int iters = argc > 5 ? atoi(argv[5]) : 7;
	unsigned char *jxl;
	size_t jxl_size, peak = 0, window = 0, induced = 0;
	double *times, best, med, mpix;
	const char *env = getenv("OIL_JXL_WINDOW");
	int i;

	printf("encoding %dx%d gradient JXL (one-time)...\n", in_w, in_h);
	encode_gradient_jxl(in_w, in_h, &jxl, &jxl_size);
	printf("  codestream: %zu bytes\n", jxl_size);
	printf("config: OIL_JXL_WINDOW=%s, %dx%d -> %dx%d, %d threads, %d iters\n",
		env ? env : "(unset/adaptive)", in_w, in_h, out_w, out_h,
		(int)JxlThreadParallelRunnerDefaultNumWorkerThreads(), iters);

	times = malloc(sizeof(*times) * iters);
	assert(times);
	for (i = 0; i < iters; i++) {
		times[i] = run_once(jxl, jxl_size, in_w, in_h, out_w, out_h,
			&peak, &window, &induced);
		printf("  iter %d: %.4f s\n", i, times[i]);
	}

	qsort(times, iters, sizeof(*times), cmp_double);
	best = times[0];
	med = times[iters / 2];
	mpix = ((double)in_w * in_h / 1e6) / best;  /* input Mpix decoded */

	printf("\n");
	printf("  best:   %.4f s   (%.1f input Mpix/s)\n", best, mpix);
	printf("  median: %.4f s\n", med);
	printf("  peak buffered rows: %zu\n", peak);
	printf("  final window:       %zu rows\n", window);
	printf("  induced starvations:%zu\n", induced);

	free(times);
	free(jxl);
	return 0;
}
