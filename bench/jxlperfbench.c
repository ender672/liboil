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

/* One decode+resize pass; returns wall seconds, reports buffer stats. */
static double run_once(unsigned char *jxl, size_t jxl_size,
	int in_w, int in_h, int out_w, int out_h,
	size_t *peak, size_t *window, size_t *induced)
{
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
	unsigned char *out;
	int cmp, fed_x, fed_y, fed_w, fed_h, y;
	size_t vpos = 0;
	double t0, t1;

	(void)in_w; (void)in_h;
	dec = open_jxl(jxl, jxl_size, &runner, &info);
	cs = jxl_cs_to_oil(&info);
	cmp = OIL_CMP(cs);

	t0 = now_sec();
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
	}
	t1 = now_sec();

	*peak = oil_jxl_rowbuf_peak_rows(rb);
	*window = oil_jxl_rowbuf_window(rb);
	*induced = oil_jxl_rowbuf_induced_starvations(rb);

	oil_jxl_rowbuf_abort(rb);
	pthread_join(driver, NULL);
	free(out);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	oil_scale_free(&os);
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
