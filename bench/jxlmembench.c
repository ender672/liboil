/*
 * Memory regression check for the reorder buffer. Decodes a tall image with a
 * throttled (slower-than-decode) consumer and reports peak finalized-but-
 * unconsumed rows. Unbounded approaches the full fed height; the window stays
 * near its size. Fails if peak exceeds TARGET_FRACTION of the image.
 *
 * Usage: ./jxlmembench [width] [height] [throttle_us]
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

/* Fraction of fed height, not an absolute count: holds across widths and
 * tolerates transient spikes when the cap lifts. Unbounded ~99%, bounded well under. */
#define TARGET_FRACTION 0.50

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
	int throttle_us = argc > 3 ? atoi(argv[3]) : 40;
	int out_w, out_h, cmp, fed_x, fed_y, fed_w, fed_h, y;
	unsigned char *jxl;
	size_t jxl_size, peak, peak_bytes, fed_bytes;
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	enum oil_colorspace cs;
	struct oil_jxl_waiter *waiter;
	struct oil_jxl_rowbuf *rb;
	struct drive drv;
	pthread_t driver;
	JxlPixelFormat fmt;
	double t0, t1;

	/* Downscale 2x so the fed rect spans nearly the full image height. */
	out_w = in_w / 2;
	out_h = in_h / 2;

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

	printf("decoding to %dx%d, fed rect %dx%d, %d worker threads, "
		"consumer throttle %dus/row\n",
		out_w, out_h, fed_w, fed_h,
		(int)JxlThreadParallelRunnerDefaultNumWorkerThreads(),
		throttle_us);

	/* Pull fed rows directly (not through the scaler) to measure the buffer alone. */
	t0 = now_sec();
	for (y = 0; y < fed_h; y++) {
		unsigned char *r = oil_jxl_rowbuf_wait_row(rb, y);
		assert(r);
		oil_jxl_rowbuf_release_row(rb, y);
		if (throttle_us > 0)
			usleep(throttle_us);
	}
	t1 = now_sec();

	peak = oil_jxl_rowbuf_peak_rows(rb);
	fed_bytes = (size_t)fed_w * cmp;
	peak_bytes = peak * fed_bytes;

	printf("\n");
	printf("  elapsed:            %.3f s\n", t1 - t0);
	printf("  fed height:         %d rows\n", fed_h);
	printf("  peak buffered rows: %zu  (%.1f%% of fed height)\n",
		peak, 100.0 * peak / fed_h);
	printf("  peak buffer bytes:  %.2f MiB\n",
		peak_bytes / (1024.0 * 1024.0));
	printf("  target:             < %.0f%% of fed height (%zu rows)\n",
		TARGET_FRACTION * 100.0, (size_t)(TARGET_FRACTION * fed_h));

	/* Release any back-pressure-parked producer so the decode thread can join
	 * even if the consumer read fewer fed rows than were buffered. */
	oil_jxl_rowbuf_abort(rb);
	pthread_join(driver, NULL);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	free(jxl);

	if (peak >= (size_t)(TARGET_FRACTION * fed_h)) {
		printf("\nRESULT: FAIL - producer raced %zu rows ahead of the "
			"consumer (buffer not bounded)\n", peak);
		return 1;
	}
	printf("\nRESULT: PASS - buffer stayed well within the bound\n");
	return 0;
}
