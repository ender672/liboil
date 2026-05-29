/*
 * Demonstrates the split decode/feed API on oil_libjxl.
 *
 * Encodes a deterministic RGB gradient to an in-memory (lossless) JPEG XL
 * codestream, then decodes the same image two ways:
 *
 *   1. The bundled all-in-one path, oil_libjxl_read_scanline.
 *   2. The split path: oil_libjxl_decode_row into a caller-owned buffer,
 *      then oil_scale_in / oil_scale_out driven by the caller.
 *
 * Asserts the two outputs are byte-identical. This is the property any caller
 * that wants to interpose a slot queue between decode and scale relies on.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>
#include "oil_resample.h"
#include "oil_libjxl.h"

#define IN_W 256
#define IN_H 192

static void encode_gradient_jxl(unsigned char **out, size_t *out_size)
{
	JxlEncoder *enc;
	JxlEncoderFrameSettings *fs;
	JxlBasicInfo info;
	JxlColorEncoding color;
	JxlPixelFormat fmt = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
	unsigned char *pixels;
	size_t cap, used;
	int x, y;

	pixels = malloc((size_t)IN_W * IN_H * 3);
	assert(pixels);
	for (y = 0; y < IN_H; y++) {
		for (x = 0; x < IN_W; x++) {
			unsigned char *p = pixels + ((size_t)y * IN_W + x) * 3;
			p[0] = (unsigned char)x;
			p[1] = (unsigned char)y;
			p[2] = (unsigned char)((x + y) >> 1);
		}
	}

	enc = JxlEncoderCreate(NULL);
	assert(enc);

	JxlEncoderInitBasicInfo(&info);
	info.xsize = IN_W;
	info.ysize = IN_H;
	info.bits_per_sample = 8;
	info.exponent_bits_per_sample = 0;
	info.num_color_channels = 3;
	info.alpha_bits = 0;
	info.uses_original_profile = JXL_TRUE;
	assert(JxlEncoderSetBasicInfo(enc, &info) == JXL_ENC_SUCCESS);

	JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
	assert(JxlEncoderSetColorEncoding(enc, &color) == JXL_ENC_SUCCESS);

	fs = JxlEncoderFrameSettingsCreate(enc, NULL);
	assert(fs);
	assert(JxlEncoderSetFrameLossless(fs, JXL_TRUE) == JXL_ENC_SUCCESS);

	assert(JxlEncoderAddImageFrame(fs, &fmt, pixels,
		(size_t)IN_W * IN_H * 3) == JXL_ENC_SUCCESS);
	JxlEncoderCloseInput(enc);

	cap = 1 << 20;
	*out = malloc(cap);
	assert(*out);
	used = 0;
	for (;;) {
		uint8_t *next = *out + used;
		size_t avail = cap - used;
		JxlEncoderStatus s = JxlEncoderProcessOutput(enc, &next, &avail);
		used = cap - avail;
		if (s == JXL_ENC_SUCCESS) break;
		assert(s == JXL_ENC_NEED_MORE_OUTPUT);
		cap *= 2;
		*out = realloc(*out, cap);
		assert(*out);
	}
	*out_size = used;

	JxlEncoderDestroy(enc);
	free(pixels);
}

/* Create a decoder, advance to JXL_DEC_BASIC_INFO, and fill *info. The runner
 * is returned via *runner_out for the caller to destroy. */
static JxlDecoder *open_jxl(unsigned char *data, size_t size,
	void **runner_out, JxlBasicInfo *info)
{
	JxlDecoder *dec;
	void *runner;

	runner = JxlThreadParallelRunnerCreate(NULL,
		JxlThreadParallelRunnerDefaultNumWorkerThreads());
	assert(runner);
	dec = JxlDecoderCreate(NULL);
	assert(dec);
	assert(JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner)
		== JXL_DEC_SUCCESS);
	assert(JxlDecoderSubscribeEvents(dec,
		JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) == JXL_DEC_SUCCESS);
	JxlDecoderSetInput(dec, data, size);
	JxlDecoderCloseInput(dec);

	assert(JxlDecoderProcessInput(dec) == JXL_DEC_BASIC_INFO);
	assert(JxlDecoderGetBasicInfo(dec, info) == JXL_DEC_SUCCESS);

	*runner_out = runner;
	return dec;
}

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

static void check_case(unsigned char *data, size_t size,
	const char *label, int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	unsigned char *a, *b;
	size_t n = (size_t)out_w * out_h * 3;
	a = decode_bundled(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	b = decode_split(data, size, out_w, out_h, src_x, src_y, src_w, src_h);
	if (memcmp(a, b, n) != 0) {
		size_t i;
		for (i = 0; i < n; i++) {
			if (a[i] != b[i]) {
				fprintf(stderr, "  %s: mismatch at byte %zu: bundled=%u split=%u\n",
					label, i, a[i], b[i]);
				break;
			}
		}
		assert(0);
	}
	printf("  %s: %dx%d ok\n", label, out_w, out_h);
	free(a);
	free(b);
}

/* Decode at integer-aligned 1:1 scale (out dims == src rect), then compare an
 * interior sub-rectangle of a cropped decode against the same region of a
 * full-image decode. Interior pixels (>=2 px from the crop edge) have all
 * Catmull-Rom taps inside both fed rects, so they must be byte-identical. */
static void check_crop_alignment(unsigned char *data, size_t size)
{
	const int cx = 64, cy = 48, cw = 128, ch = 96;
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

	encode_gradient_jxl(&jxl, &jxl_size);
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
