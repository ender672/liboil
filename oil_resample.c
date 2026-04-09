/**
 * Copyright (c) 2014-2019 Timothy Elliott
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "oil_resample.h"
#include "oil_resample_internal.h"
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>

/**
 * When shrinking a 10 million pixel wide scanline down to a single pixel, we
 * reach the limits of single-precision floats. Limit input dimensions to one
 * million by one million pixels to avoid this issue as well as overflow issues
 * with 32-bit ints.
 */
#define MAX_DIMENSION 1000000

/**
 * Bicubic interpolation. 2 base taps on either side.
 */
#define TAPS 4

static int max(int a, int b)
{
	return a > b ? a : b;
}

static int min(int a, int b)
{
	return a < b ? a : b;
}

/**
 * Clamp a float between 0 and 1.
 */
static float clampf(float x)
{
	if (x > 1.0f) {
		return 1.0f;
	} else if (x < 0.0f) {
		return 0.0f;
	}
	return x;
}

/**
 * Convert a float to an int. When compiling on x86 without march=native, this
 * performs much better than roundf().
 */
static int f2i(float x)
{
	return x + 0.5f;
}

/**
 * Convert a float to 8-bit integer.
 */
/**
 * Map from the discreet dest coordinate pos to a continuous source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
static double map(int dim_in, int dim_out, int pos)
{
	return (pos + 0.5) * ((double)dim_in / dim_out) - 0.5;
}

/**
 * Returns the mapped input position and put the sub-pixel remainder in rest.
 */
static int split_map(int dim_in, int dim_out, int pos, float *rest)
{
	double smp;
	int smp_i;

	smp = map(dim_in, dim_out, pos);
	smp_i = smp < 0 ? -1 : smp;
	*rest = smp - smp_i;
	return smp_i;
}

/**
 * Given input and output dimension, calculate the total number of taps that
 * will be needed to calculate an output sample.
 *
 * When we reduce an image by a factor of two, we need to scale our resampling
 * function by two as well in order to avoid aliasing.
 */
static int calc_taps(int dim_in, int dim_out)
{
	int tmp;
	if (dim_out > dim_in) {
		return TAPS;
	}
	tmp = TAPS * dim_in / dim_out;
	return tmp - (tmp & 1);
}

/**
 * Catmull-Rom interpolator.
 */
static float catrom(float x)
{
	if (x<1) {
		return (1.5f*x - 2.5f)*x*x + 1;
	}
	return (((5 - x)*x - 8)*x + 4) / 2;
}

/**
 * Given an offset tx, calculate taps coefficients.
 */
static void calc_coeffs(float *coeffs, float tx, int taps, int ltrim, int rtrim)
{
	int i;
	float tmp, tap_mult, fudge;

	tap_mult = (float)taps / TAPS;
	tx = 1 - tx - taps / 2 + ltrim;
	fudge = 0.0f;

	for (i=ltrim; i<taps-rtrim; i++) {
		tmp = catrom(fabsf(tx) / tap_mult) / tap_mult;
		fudge += tmp;
		coeffs[i] = tmp;
		tx += 1;
	}
	fudge = 1 / fudge;
	for (i=ltrim; i<taps-rtrim; i++) {
		coeffs[i] *= fudge;
	}
}

/**
 * Takes a sample value, an array of 4 coefficients & 4 accumulators, and
 * adds the product of sample * coeffs[n] to each accumulator.
 */
static void add_sample_to_sum_f(float sample, float *coeffs, float *sum)
{
	int i;
	for (i=0; i<4; i++) {
		sum[i] += sample * coeffs[i];
	}
}

/**
 * Takes an array of 4 floats and shifts them left. The rightmost element is
 * set to 0.0.
 */
static void shift_left_f(float *f)
{
	f[0] = f[1];
	f[1] = f[2];
	f[2] = f[3];
	f[3] = 0.0f;
}

static void yscale_out_rgba_nogamma(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, j, tap_off;
	float alpha, val;

	tap_off = tap * 4;
	for (i=0; i<width; i++) {
		alpha = clampf(sums[tap_off + 3]);
		for (j=0; j<3; j++) {
			val = sums[tap_off + j];
			if (alpha != 0) {
				val /= alpha;
			}
			out[j] = f2i(clampf(val) * 255.0f);
			sums[tap_off + j] = 0.0f;
		}
		out[3] = round(alpha * 255.0f);
		sums[tap_off + 3] = 0.0f;
		sums += 16;
		out += 4;
	}
}

static void yscale_out_rgbx_nogamma(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, j, tap_off;

	tap_off = tap * 4;
	for (i=0; i<width; i++) {
		for (j=0; j<3; j++) {
			out[j] = f2i(clampf(sums[tap_off + j]) * 255.0f);
			sums[tap_off + j] = 0.0f;
		}
		out[3] = 255;
		sums[tap_off + 3] = 0.0f;
		sums += 16;
		out += 4;
	}
}

static void yscale_out(float *sums, int width, unsigned char *out,
	enum oil_colorspace cs, int tap)
{
	switch(cs) {
	case OIL_CS_RGBA_NOGAMMA:
		yscale_out_rgba_nogamma(sums, width, out, tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		yscale_out_rgbx_nogamma(sums, width, out, tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

/* horizontal scaling */

float i2f_map[256];

static void build_i2f(void)
{
	int i;

	for (i=0; i<=255; i++) {
		i2f_map[i] = i / 255.0f;
	}
}

/**
 * Given input & output dimensions, populate a buffer of coefficients and border counters.
 *
 * This method assumes that in_dim >= out_dim.
 *
 * It generates 4 * in_dim coefficients -- 4 for every input sample.
 *
 * It generates out_dim border counters, these indicate how many input samples to process before
 * the next output sample is finished.
 */
static void scale_down_coeffs(int in_dim, int out_dim, float *coeff_buf, int *border_buf,
	float *tmp_coeffs)
{
	int smp_i, i, j, taps, offset, pos, ltrim, rtrim, smp_end, smp_start, ends[4];
	float tx;

	taps = calc_taps(in_dim, out_dim);
	for (i=0; i<4; i++) {
		ends[i] = -1;
	}

	for (i=0; i<out_dim; i++) {
		smp_i = split_map(in_dim, out_dim, i, &tx);

		smp_start = smp_i - (taps/2 - 1);
		smp_end = smp_i + taps/2;
		if (smp_end >= in_dim) {
			smp_end = in_dim - 1;
		}
		ends[i%4] = smp_end;
		border_buf[i] = smp_end - ends[(i+3)%4];

		ltrim = 0;
		if (smp_start < 0) {
			ltrim = -1 * smp_start;
		}
		rtrim = smp_start + (taps - 1) - smp_end;
		calc_coeffs(tmp_coeffs, tx, taps, ltrim, rtrim);

		for (j=ltrim; j<taps - rtrim; j++) {
			pos = smp_start + j;

			offset = 3;
			if (pos > ends[(i+3)%4]) {
				offset = 0;
			} else if (pos > ends[(i+2)%4]) {
				offset = 1;
			} else if (pos > ends[(i+1)%4]) {
				offset = 2;
			}

			coeff_buf[pos * 4 + offset] = tmp_coeffs[j];
		}
	}
}


static void scale_down_rgba_nogamma(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y, int tap)
{
	int i, j, k;
	float alpha, sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			alpha = i2f_map[in[3]];
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(i2f_map[in[k]] * alpha, coeffs_x, sum[k]);
			}
			add_sample_to_sum_f(alpha, coeffs_x, sum[3]);
			in += 4;
			coeffs_x += 4;
		}

		{
			float samples[4];
			for (j=0; j<4; j++) {
				samples[j] = sum[j][0];
				shift_left_f(sum[j]);
			}
			for (j=0; j<4; j++) {
				float cy = coeffs_y[j];
				int off = ((tap + j) & 3) * 4;
				sums_y[off + 0] += samples[0] * cy;
				sums_y[off + 1] += samples[1] * cy;
				sums_y[off + 2] += samples[2] * cy;
				sums_y[off + 3] += samples[3] * cy;
			}
			sums_y += 16;
		}
	}
}

static void scale_down_rgbx_nogamma(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y, int tap)
{
	int i, j, k;
	float sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(i2f_map[in[k]], coeffs_x, sum[k]);
			}
			add_sample_to_sum_f(1.0f, coeffs_x, sum[3]);
			in += 4;
			coeffs_x += 4;
		}

		{
			float samples[4];
			for (j=0; j<4; j++) {
				samples[j] = sum[j][0];
				shift_left_f(sum[j]);
			}
			for (j=0; j<4; j++) {
				float cy = coeffs_y[j];
				int off = ((tap + j) & 3) * 4;
				sums_y[off + 0] += samples[0] * cy;
				sums_y[off + 1] += samples[1] * cy;
				sums_y[off + 2] += samples[2] * cy;
				sums_y[off + 3] += samples[3] * cy;
			}
			sums_y += 16;
		}
	}
}

/* Global functions */
void oil_global_init(void)
{
	build_i2f();
}

#define ALIGN16(x) (((x) + 15) & ~15)

static int calc_coeffs_len(int in_dim, int out_dim)
{
	return TAPS * max(in_dim, out_dim) * sizeof(float);
}

static int calc_borders_len(int in_dim, int out_dim)
{
	return min(in_dim, out_dim) * sizeof(int);
}

static int downscale_alloc_size(int in_height, int out_height, int in_width,
	int out_width, enum oil_colorspace cs)
{
	int taps_x, taps_y;

	taps_x = calc_taps(in_width, out_width);
	taps_y = calc_taps(in_height, out_height);

	return ALIGN16(calc_coeffs_len(in_width, out_width))
		+ ALIGN16(calc_borders_len(in_width, out_width))
		+ ALIGN16(calc_coeffs_len(in_height, out_height))
		+ ALIGN16(calc_borders_len(in_height, out_height))
		+ ALIGN16(max(taps_x, taps_y) * sizeof(float))
		+ ALIGN16(out_width * OIL_CMP(cs) * TAPS * sizeof(float));
}

static void downscale_init(struct oil_scale *os)
{
	int coeffs_x_len, coeffs_y_len, borders_x_len, borders_y_len, sums_len;
	char *p;

	coeffs_x_len = ALIGN16(calc_coeffs_len(os->in_width, os->out_width));
	borders_x_len = ALIGN16(calc_borders_len(os->in_width, os->out_width));
	coeffs_y_len = ALIGN16(calc_coeffs_len(os->in_height, os->out_height));
	borders_y_len = ALIGN16(calc_borders_len(os->in_height, os->out_height));
	sums_len = ALIGN16(os->out_width * OIL_CMP(os->cs) * TAPS * sizeof(float));

	p = os->buf;
	os->coeffs_x = (float *)p;		p += coeffs_x_len;
	os->borders_x = (int *)p;		p += borders_x_len;
	os->coeffs_y = (float *)p;		p += coeffs_y_len;
	os->borders_y = (int *)p;		p += borders_y_len;
	os->sums_y = (float *)p;		p += sums_len;
	os->tmp_coeffs = (float *)p;

	scale_down_coeffs(os->in_width, os->out_width, os->coeffs_x, os->borders_x,
		os->tmp_coeffs);
	scale_down_coeffs(os->in_height, os->out_height, os->coeffs_y, os->borders_y,
		os->tmp_coeffs);
}

int oil_scale_alloc_size(int in_height, int out_height, int in_width,
	int out_width, enum oil_colorspace cs)
{
	return downscale_alloc_size(in_height, out_height, in_width,
		out_width, cs);
}

int oil_scale_init_allocated(struct oil_scale *os, int in_height,
	int out_height, int in_width, int out_width, enum oil_colorspace cs,
	void *buf)
{
	/* sanity check on arguments */
	if (!os || !buf || in_height > MAX_DIMENSION || out_height > MAX_DIMENSION ||
		in_height < 1 || out_height < 1 ||
		in_width > MAX_DIMENSION || out_width > MAX_DIMENSION ||
		in_width < 1 || out_width < 1) {
		return -1;
	}

	/* only downscaling is supported */
	if (out_height > in_height || out_width > in_width) {
		return -1;
	}

	// Lazy perform global init, in case oil_global_ini() hasn't been
	// called yet.
	if (!i2f_map[128]) {
		oil_global_init();
	}

	memset(os, 0, sizeof(struct oil_scale));
	os->in_height = in_height;
	os->out_height = out_height;
	os->in_width = in_width;
	os->out_width = out_width;
	os->cs = cs;
	os->buf = buf;

	downscale_init(os);

	return 0;
}

int oil_scale_init(struct oil_scale *os, int in_height, int out_height,
	int in_width, int out_width, enum oil_colorspace cs)
{
	int alloc_size, ret;
	void *buf;

	alloc_size = oil_scale_alloc_size(in_height, out_height, in_width,
		out_width, cs);
	buf = calloc(1, alloc_size);
	if (!buf) {
		return -2;
	}

	ret = oil_scale_init_allocated(os, in_height, out_height, in_width,
		out_width, cs, buf);
	if (ret) {
		free(buf);
		return ret;
	}

	return 0;
}

void oil_scale_restart(struct oil_scale *os)
{
	os->in_pos = os->out_pos = 0;
	os->sums_y_tap = 0;
}

void oil_scale_free(struct oil_scale *os)
{
	if (!os) {
		return;
	}

	free(os->buf);
	os->buf = NULL;
	os->coeffs_x = NULL;
	os->borders_x = NULL;
	os->coeffs_y = NULL;
	os->borders_y = NULL;
	os->sums_y = NULL;
	os->tmp_coeffs = NULL;
}

int oil_scale_slots(struct oil_scale *ys)
{
	return ys->borders_y[ys->out_pos];
}

static void down_scale_in(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	switch(os->cs) {
	case OIL_CS_RGBA_NOGAMMA:
		scale_down_rgba_nogamma(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		scale_down_rgbx_nogamma(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}

	os->borders_y[os->out_pos] -= 1;
	os->in_pos++;
}

int oil_scale_in(struct oil_scale *os, unsigned char *in)
{
	if (oil_scale_slots(os) == 0) {
		return -1;
	}
	down_scale_in(os, in);
	return 0;
}

int oil_scale_out(struct oil_scale *os, unsigned char *out)
{
	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	yscale_out(os->sums_y, os->out_width, out, os->cs, os->sums_y_tap);
	os->sums_y_tap = (os->sums_y_tap + 1) & 3;

	os->out_pos++;
	return 0;
}
