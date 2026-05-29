/* Shared helpers for the JXL wrapper's tests and benchmarks. See jxl_testutil.h. */
#include "jxl_testutil.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>

double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int cmp_double(const void *a, const void *b)
{
	double da = *(const double *)a, db = *(const double *)b;
	return (da > db) - (da < db);
}

void encode_gradient_jxl(int w, int h, unsigned char **out, size_t *out_size)
{
	JxlEncoder *enc;
	JxlEncoderFrameSettings *fs;
	JxlBasicInfo info;
	JxlColorEncoding color;
	JxlPixelFormat fmt = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
	unsigned char *pixels;
	size_t cap, used, npix = (size_t)w * h;
	int x, y;

	pixels = malloc(npix * 3);
	assert(pixels);
	for (y = 0; y < h; y++) {
		for (x = 0; x < w; x++) {
			unsigned char *p = pixels + ((size_t)y * w + x) * 3;
			p[0] = (unsigned char)x;
			p[1] = (unsigned char)y;
			p[2] = (unsigned char)((x + y) >> 1);
		}
	}

	enc = JxlEncoderCreate(NULL);
	assert(enc);
	JxlEncoderInitBasicInfo(&info);
	info.xsize = w;
	info.ysize = h;
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
	/* Lowest effort: these care about decode speed, not encode. */
	JxlEncoderFrameSettingsSetOption(fs, JXL_ENC_FRAME_SETTING_EFFORT, 1);

	assert(JxlEncoderAddImageFrame(fs, &fmt, pixels, npix * 3)
		== JXL_ENC_SUCCESS);
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

JxlDecoder *open_jxl(unsigned char *data, size_t size,
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
