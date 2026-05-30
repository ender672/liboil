/*
 * Tests the libjxl helper paths.
 *
 * Encodes a deterministic RGB gradient to an in-memory (lossless) JPEG XL
 * codestream, then decodes+resizes it two ways:
 *
 *   1. oil_jxl_resample - the one-call convenience.
 *   2. A manual Path-B composition: a rowbuf fed by an oil_jxl_run_decode
 *      thread, pulled and scaled by the caller (what oil_jxl_resample does
 *      internally, and what imgscale does to stream to an encoder).
 *
 * Asserts the two agree byte-for-byte (the easy button matches hand
 * composition), plus an independent crop-alignment check: at integer-aligned
 * 1:1 scaling the Catmull-Rom kernel is an identity, so an interior crop must
 * reproduce the matching region of a full-image decode.
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

/* IN_W is wider than the rowbuf's 256px tile_w so the full-image and wide-crop
 * decodes span multiple tiles, exercising the multi-tile coalescing and
 * crop-local clipping paths in oil_jxl_rowbuf_write_segment. */
#define IN_W 600
#define IN_H 192

static unsigned char *decode_resample(unsigned char *data, size_t size,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	unsigned char *out;

	dec = open_jxl(data, size, &runner, &info);
	out = malloc((size_t)out_w * out_h * 3);
	assert(oil_jxl_resample(dec, &info, out_w, out_h,
		src_x, src_y, src_w, src_h, OIL_CS_UNKNOWN,
		out, (size_t)out_w * 3) == 0);

	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	return out;
}

struct drive { JxlDecoder *dec; const JxlPixelFormat *fmt;
	struct oil_jxl_rowbuf *rb; };
static void *drive_thread(void *arg)
{
	struct drive *d = arg;
	oil_jxl_run_decode(d->dec, d->fmt, d->rb);
	return NULL;
}

/* Hand-composed Path B: own the decode thread, pull rows from the rowbuf and
 * drive the scaler. */
static unsigned char *decode_manual(unsigned char *data, size_t size,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
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

	dec = open_jxl(data, size, &runner, &info);
	cs = jxl_cs_to_oil(&info);
	assert(cs != OIL_CS_UNKNOWN);
	cmp = OIL_CMP(cs);
	assert(oil_required_input_rect(info.ysize, info.xsize,
		src_y, src_h, src_x, src_w, out_h, out_w,
		&fed_y, &fed_h, &fed_x, &fed_w) == 0);
	assert(oil_scale_init_ex(&os, fed_h, out_h, fed_w, out_w,
		src_y - fed_y, src_h, src_x - fed_x, src_w, cs) == 0);

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

	oil_jxl_rowbuf_abort(rb);
	pthread_join(driver, NULL);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	oil_scale_free(&os);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	return out;
}

static void expect_eq(const char *label, const char *which,
	const unsigned char *a, const unsigned char *b, size_t n)
{
	if (memcmp(a, b, n) != 0) {
		size_t i;
		for (i = 0; i < n; i++) {
			if (a[i] != b[i]) {
				fprintf(stderr, "  %s: %s mismatch at byte %zu: "
					"resample=%u %s=%u\n",
					label, which, i, a[i], which, b[i]);
				break;
			}
		}
		assert(0);
	}
}

static void check_case(unsigned char *data, size_t size,
	const char *label, int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	unsigned char *a, *b;
	size_t n = (size_t)out_w * out_h * 3;
	a = decode_resample(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	b = decode_manual(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	expect_eq(label, "manual", a, b, n);
	printf("  %s: %dx%d ok\n", label, out_w, out_h);
	free(a);
	free(b);
}

/* At integer-aligned 1:1 scale the Catmull-Rom kernel is an identity, so an
 * interior crop (>=2 px from the edge) must reproduce exactly the matching
 * region of a full-image decode. The crop is wider than the 256px tile_w and
 * starts at a non-tile-aligned column, exercising the multi-tile clip-and-shift
 * path where a wrong tile index or offset would corrupt interior pixels. */
static void check_crop_alignment(unsigned char *data, size_t size)
{
	const int cx = 96, cy = 48, cw = 384, ch = 96;
	const int margin = 2;
	unsigned char *full, *crop;
	int x, y;

	full = decode_resample(data, size, IN_W, IN_H,
		0.0, 0.0, (double)IN_W, (double)IN_H);
	crop = decode_resample(data, size, cw, ch,
		(double)cx, (double)cy, (double)cw, (double)ch);

	for (y = margin; y < ch - margin; y++) {
		for (x = margin; x < cw - margin; x++) {
			unsigned char *c = crop + ((size_t)y * cw + x) * 3;
			unsigned char *f = full +
				((size_t)(y + cy) * IN_W + (x + cx)) * 3;
			if (memcmp(c, f, 3) != 0) {
				fprintf(stderr, "  crop alignment: mismatch at "
					"crop (%d,%d): crop=%u,%u,%u full=%u,%u,%u\n",
					x, y, c[0], c[1], c[2], f[0], f[1], f[2]);
				assert(0);
			}
		}
	}
	printf("  crop alignment: interior %dx%d matches full decode\n",
		cw - 2 * margin, ch - 2 * margin);
	free(full);
	free(crop);
}

int main(void)
{
	unsigned char *jxl;
	size_t jxl_size;

	encode_gradient_jxl(IN_W, IN_H, &jxl, &jxl_size);
	printf("encoded test JXL: %zu bytes (%dx%d gradient)\n",
		jxl_size, IN_W, IN_H);

	check_case(jxl, jxl_size, "full image",       100,  75,
		0.0, 0.0, (double)IN_W, (double)IN_H);
	check_case(jxl, jxl_size, "integer crop",      80,  60,
		16.0, 32.0, 128.0, 96.0);
	check_case(jxl, jxl_size, "sub-pixel crop",   100,  75,
		10.25, 5.5, 200.75, 150.25);
	check_case(jxl, jxl_size, "tiny output",        4,   3,
		0.0, 0.0, (double)IN_W, (double)IN_H);
	check_case(jxl, jxl_size, "upscale crop",     200, 150,
		50.0, 40.0, 32.0, 24.0);

	check_crop_alignment(jxl, jxl_size);

	free(jxl);
	printf("All tests pass.\n");
	return 0;
}
