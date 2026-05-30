/*
 * Demonstrates the split decode/feed API on oil_libjxl.
 *
 * Encodes a deterministic RGB gradient to an in-memory (lossless) JPEG XL
 * codestream, then decodes the same image three ways:
 *
 *   1. The bundled all-in-one path, oil_libjxl_read_scanline.
 *   2. The split path: oil_libjxl_decode_row into a caller-owned buffer,
 *      then oil_scale_in / oil_scale_out driven by the caller.
 *   3. The one-call convenience oil_jxl_resample.
 *
 * Asserts all three outputs are byte-identical. The split path is what a caller
 * that wants to interpose a slot queue between decode and scale relies on;
 * matching oil_jxl_resample confirms the easy button agrees with the wrapper.
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

/* IN_W is wider than the tile buffer's 256px tile_w so the full-image and
 * wide-crop decodes span multiple tiles, exercising the multi-tile coalescing
 * and crop-local clipping paths (a single-tile image would never iterate the
 * per-tile loop in oil_jxl_rowbuf_write_segment). */
#define IN_W 600
#define IN_H 192

static unsigned char *decode_bundled(unsigned char *data, size_t size,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	struct oil_libjxl ol;
	unsigned char *out;
	int y;

	dec = open_jxl(data, size, &runner, &info);
	assert(oil_libjxl_init_ex(&ol, dec, &info, out_w, out_h,
		src_x, src_y, src_w, src_h, OIL_CS_UNKNOWN) == 0);

	out = malloc((size_t)out_w * out_h * ol.components);
	for (y = 0; y < out_h; y++) {
		oil_libjxl_read_scanline(&ol,
			out + (size_t)y * out_w * ol.components);
	}
	assert(!ol.error);

	oil_libjxl_free(&ol);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	return out;
}

static unsigned char *decode_split(unsigned char *data, size_t size,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	JxlDecoder *dec;
	void *runner;
	JxlBasicInfo info;
	struct oil_libjxl ol;
	unsigned char *out, *row;
	int y;

	dec = open_jxl(data, size, &runner, &info);
	assert(oil_libjxl_init_ex(&ol, dec, &info, out_w, out_h,
		src_x, src_y, src_w, src_h, OIL_CS_UNKNOWN) == 0);

	row = malloc((size_t)ol.fed_width * ol.components);
	out = malloc((size_t)out_w * out_h * ol.components);
	for (y = 0; y < out_h; y++) {
		while (oil_scale_slots(&ol.os) > 0) {
			oil_libjxl_decode_row(&ol, row);
			oil_scale_in(&ol.os, row);
		}
		oil_scale_out(&ol.os, out + (size_t)y * out_w * ol.components);
	}
	assert(!ol.error);
	free(row);

	oil_libjxl_free(&ol);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	return out;
}

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

static void expect_eq(const char *label, const char *which,
	const unsigned char *a, const unsigned char *b, size_t n)
{
	if (memcmp(a, b, n) != 0) {
		size_t i;
		for (i = 0; i < n; i++) {
			if (a[i] != b[i]) {
				fprintf(stderr, "  %s: %s mismatch at byte %zu: "
					"bundled=%u %s=%u\n",
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
	unsigned char *a, *b, *c;
	size_t n = (size_t)out_w * out_h * 3;
	a = decode_bundled(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	b = decode_split(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	c = decode_resample(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	expect_eq(label, "split", a, b, n);
	expect_eq(label, "resample", a, c, n);
	printf("  %s: %dx%d ok\n", label, out_w, out_h);
	free(a);
	free(b);
	free(c);
}

/* Decode at integer-aligned 1:1 scale (out dims == src rect), then compare an
 * interior sub-rectangle of a cropped decode against the same region of a
 * full-image decode. Interior pixels (>=2 px from the crop edge) have all
 * Catmull-Rom taps inside both fed rects, so they must be byte-identical.
 *
 * The crop is wider than the 256px tile_w and starts at a non-tile-aligned
 * column, so the fed rect spans a crop-local tile boundary: this exercises
 * the multi-tile clip-and-shift path, where a mistaken tile index or column
 * offset would corrupt interior pixels that this check would then catch. */
static void check_crop_alignment(unsigned char *data, size_t size)
{
	const int cx = 96, cy = 48, cw = 384, ch = 96;
	const int margin = 2;
	unsigned char *full, *crop;
	int x, y;

	full = decode_bundled(data, size, IN_W, IN_H,
		0.0, 0.0, (double)IN_W, (double)IN_H);
	crop = decode_bundled(data, size, cw, ch,
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

	/* Full image, simple downscale. */
	check_case(jxl, jxl_size, "full image",       100,  75,
		0.0, 0.0, (double)IN_W, (double)IN_H);

	/* Integer-aligned crop. */
	check_case(jxl, jxl_size, "integer crop",      80,  60,
		16.0, 32.0, 128.0, 96.0);

	/* Sub-pixel src rect. */
	check_case(jxl, jxl_size, "sub-pixel crop",   100,  75,
		10.25, 5.5, 200.75, 150.25);

	/* Tiny output. */
	check_case(jxl, jxl_size, "tiny output",        4,   3,
		0.0, 0.0, (double)IN_W, (double)IN_H);

	/* Upscale a small region. */
	check_case(jxl, jxl_size, "upscale crop",     200, 150,
		50.0, 40.0, 32.0, 24.0);

	/* Independent crop-alignment check: at integer-aligned 1:1 scaling the
	 * Catmull-Rom kernel is an identity, so an interior crop must reproduce
	 * exactly the same pixels as the matching region of a full-image decode.
	 * Both paths share the gamma roundtrip, so any rounding cancels. This
	 * catches a crop region that is shifted, unlike check_case (which only
	 * compares the two decode paths, both of which use the same code). */
	check_crop_alignment(jxl, jxl_size);

	free(jxl);
	printf("All tests pass.\n");
	return 0;
}
