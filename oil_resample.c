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
#include <math.h>
#include <stdlib.h>
#include <limits.h>

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

/**
 * Clamp a float between 0 and 1.
 */
static float clampf(float x) {
	if (x > 1.0f) {
		return 1.0f;
	} else if (x < 0.0f) {
		return 0.0f;
	}
	return x;
}

/**
 * Convert a float to 8-bit integer.
 */
static int clamp8(float x)
{
	return round(clampf(x) * 255.0f);
}

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
	if (x>2) {
		return 0;
	}
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
 * Holds pre-calculated table of linear float to srgb char mappings.
 * Initialized via build_l2s_rights();
 */
static float l2s_rights[256];

/**
 * Populates l2s_rights.
 */
static void build_l2s_rights(void)
{
	int i;
	double srgb_f, tmp, val;

	for (i=0; i<255; i++) {
		srgb_f = (i + 0.5)/255.0;
		if (srgb_f <= 0.0404482362771082) {
			val = srgb_f / 12.92;
		} else {
			tmp = (srgb_f + 0.055)/1.055;
			val = pow(tmp, 2.4);
		}
		l2s_rights[i] = val;
	}
	l2s_rights[i] = 256.0f;
}

/**
 * Maps the given linear RGB float to sRGB integer.
 * 
 * Performs a binary search on l2s_rights.
 */
static int linear_sample_to_srgb(float in)
{
	int offs, i;
	offs = 0;
	for (i=128; i>0; i >>= 1) {
		if (in > l2s_rights[offs + i]) {
			offs += i;
		}
	}
	return in > l2s_rights[offs] ? offs + 1 : offs;
}

/**
 * Takes an array of 4 floats and shifts them left. The rightmost element is
 * set to the given value.
 */
static void push_f(float *f, float val)
{
	f[0] = f[1];
	f[1] = f[2];
	f[2] = f[3];
	f[3] = val;
}

/**
 * Takes an array of 4 floats and shifts them left. The rightmost element is
 * set to 0.0.
 */
static void shift_left_f(float *f)
{
	push_f(f, 0.0f);
}

/**
 * Resizes a strip of RGBX scanlines to a single scanline.
 */
static void strip_scale_rgbx(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs)
{
	int i, j;
	double sum[3];

	for (i=0; i<len; i+=4) {
		sum[0] = sum[1] = sum[2] = 0;
		for (j=0; j<strip_height; j++) {
			sum[0] += coeffs[j] * in[j][i];
			sum[1] += coeffs[j] * in[j][i + 1];
			sum[2] += coeffs[j] * in[j][i + 2];
		}
		out[0] = linear_sample_to_srgb(sum[0]);
		out[1] = linear_sample_to_srgb(sum[1]);
		out[2] = linear_sample_to_srgb(sum[2]);
		out[3] = 0;
		out += 4;
	}
}

/**
 * Resizes a strip of RGB scanlines to a single scanline.
 */
static void strip_scale_rgb(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs)
{
	int i, j;
	double sum[3];

	for (i=0; i<len; i+=3) {
		sum[0] = sum[1] = sum[2] = 0;
		for (j=0; j<strip_height; j++) {
			sum[0] += coeffs[j] * in[j][i];
			sum[1] += coeffs[j] * in[j][i + 1];
			sum[2] += coeffs[j] * in[j][i + 2];
		}
		out[0] = linear_sample_to_srgb(sum[0]);
		out[1] = linear_sample_to_srgb(sum[1]);
		out[2] = linear_sample_to_srgb(sum[2]);
		out += 3;
	}
}

/**
 * Resizes a strip of greyscale scanlines to a single scanline.
 */
static void strip_scale_g(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs)
{
	int i, j;
	double sum;

	for (i=0; i<len; i++) {
		sum = 0;
		for (j=0; j<strip_height; j++) {
			sum += coeffs[j] * in[j][i];
		}
		out[i] = clamp8(sum);
	}
}

/**
 * Resizes a strip of greyscale-alpha scanlines to a single scanline.
 */
static void strip_scale_ga(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs)
{
	int i, j;
	double sum[2], alpha;

	for (i=0; i<len; i+=2) {
		sum[0] = sum[1] = 0;
		for (j=0; j<strip_height; j++) {
			sum[0] += coeffs[j] * in[j][i];
			sum[1] += coeffs[j] * in[j][i + 1];
		}
		alpha = clampf(sum[1]);
		if (alpha != 0) {
			sum[0] /= alpha;
		}
		out[0] = clamp8(sum[0]);
		out[1] = round(alpha * 255.0f);
		out += 2;
	}
}

/**
 * Resizes a strip of RGB-alpha scanlines to a single scanline.
 */
static void strip_scale_rgba(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs)
{
	int i, j;
	double sum[4], alpha;

	for (i=0; i<len; i+=4) {
		sum[0] = sum[1] = sum[2] = sum[3] = 0;
		for (j=0; j<strip_height; j++) {
			sum[0] += coeffs[j] * in[j][i];
			sum[1] += coeffs[j] * in[j][i + 1];
			sum[2] += coeffs[j] * in[j][i + 2];
			sum[3] += coeffs[j] * in[j][i + 3];
		}
		alpha = clampf(sum[3]);
		if (alpha != 0) {
			sum[0] /= alpha;
			sum[1] /= alpha;
			sum[2] /= alpha;
		}
		out[0] = linear_sample_to_srgb(sum[0]);
		out[1] = linear_sample_to_srgb(sum[1]);
		out[2] = linear_sample_to_srgb(sum[2]);
		out[3] = round(alpha * 255.0f);
		out += 4;
	}
}

/**
 * Resizes a strip of CMYK scanlines to a single scanline.
 */
static void strip_scale_cmyk(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs)
{
	int i, j;
	double sum[4];

	for (i=0; i<len; i+=4) {
		sum[0] = sum[1] = sum[2] = sum[3] = 0;
		for (j=0; j<strip_height; j++) {
			sum[0] += coeffs[j] * in[j][i];
			sum[1] += coeffs[j] * in[j][i + 1];
			sum[2] += coeffs[j] * in[j][i + 2];
			sum[3] += coeffs[j] * in[j][i + 3];
		}
		out[0] = clamp8(sum[0]);
		out[1] = clamp8(sum[1]);
		out[2] = clamp8(sum[2]);
		out[3] = clamp8(sum[3]);
		out += 4;
	}
}

/**
 * Scale a strip of scanlines. Branches to the correct interpolator using the
 * given colorspace.
 */
static void strip_scale(float **in, int strip_height, int len,
	unsigned char *out, float *coeffs, float ty, enum oil_colorspace cs)
{
	calc_coeffs(coeffs, ty, strip_height, 0, 0);

	switch(cs) {
	case OIL_CS_G:
		strip_scale_g(in, strip_height, len, out, coeffs);
		break;
	case OIL_CS_GA:
		strip_scale_ga(in, strip_height, len, out, coeffs);
		break;
	case OIL_CS_RGB:
		strip_scale_rgb(in, strip_height, len, out, coeffs);
		break;
	case OIL_CS_RGBX:
		strip_scale_rgbx(in, strip_height, len, out, coeffs);
		break;
	case OIL_CS_RGBA:
		strip_scale_rgba(in, strip_height, len, out, coeffs);
		break;
	case OIL_CS_CMYK:
		strip_scale_cmyk(in, strip_height, len, out, coeffs);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

/* horizontal scaling */

/**
 * Holds pre-calculated mapping of sRGB chars to linear RGB floating point
 * values.
 */
static float s2l_map_f[256];

/**
 * Populates s2l_map_f.
 */
static void build_s2l(void)
{
	int input;
	double in_f, tmp, val;

	for (input=0; input<=255; input++) {
		in_f = input / 255.0;
		if (in_f <= 0.040448236277) {
			val = in_f / 12.92;
		} else {
			tmp = ((in_f + 0.055)/1.055);
			val = pow(tmp, 2.4);
		}
		s2l_map_f[input] = val;
	}
}

/**
 * Given input & output dimensions, populate a buffer of coefficients and
 * border counters.
 *
 * This method assumes that in_width >= out_width.
 *
 * It generates 4 * in_width coefficients -- 4 for every input sample.
 *
 * It generates out_width border counters, these indicate how many input
 * samples to process before the next output sample is finished.
 */
static void xscale_calc_coeffs(int in_width, int out_width, float *coeff_buf,
	int *border_buf, float *tmp_coeffs)
{
	int smp_i, i, j, taps, offset, pos, ltrim, rtrim, smp_end, smp_start,
		ends[4];
	float tx;

	taps = calc_taps(in_width, out_width);
	for (i=0; i<4; i++) {
		ends[i] = -1;
	}

	for (i=0; i<out_width; i++) {
		smp_i = split_map(in_width, out_width, i, &tx);

		smp_start = smp_i - (taps/2 - 1);
		smp_end = smp_i + taps/2;
		if (smp_end >= in_width) {
			smp_end = in_width - 1;
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

/**
 * Precalculate coefficients and borders for an upscale.
 *
 * coeff_buf will be populated with 4 input coefficients for every output
 * sample.
 *
 * border_buf will be populated with the number of output samples to produce
 * for every input sample.
 *
 * users of coeff_buf & border_buf are expected to keep a buffer of the last 4
 * input samples, and multiply them with each output sample's coefficients.
 */
static void scale_up_coeffs(int in_width, int out_width, float *coeff_buf,
	int *border_buf)
{
	int i, smp_i, start, end, ltrim, rtrim, safe_end, max_pos;
	float tx;

	max_pos = in_width - 1;
	for (i=0; i<out_width; i++) {
		smp_i = split_map(in_width, out_width, i, &tx);
		start = smp_i - 1;
		end = smp_i + 2;

		// This is the border position at which we will tell the
		// interpolator to calculate the output sample.
		safe_end = end > max_pos ? max_pos : end;

		ltrim = 0;
		rtrim = 0;
		if (start < 0) {
			ltrim = -1 * start;
		}
		if (end > max_pos) {
			rtrim = end - max_pos;
		}

		border_buf[safe_end] += 1;

		// we offset coeff_buf by rtrim because the interpolator won't
		// be pushing any more samples into its sample buffer.
		calc_coeffs(coeff_buf + rtrim, tx, 4, ltrim, rtrim);

		coeff_buf += 4;
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
 * Takes an array of n 4-element source arrays, writes the first element to the
 * next n positions of the output address, and shifts the source arrays.
 */
static void dump_out(float *out, float sum[][4], int n)
{
	int i;
	for (i=0; i<n; i++) {
		out[i] = sum[i][0];
		shift_left_f(sum[i]);
	}
}

static void xscale_down_rgbx(unsigned char *in, float *out,
	int out_width, float *coeff_buf, int *border_buf)
{
	int i, j, k;
	float sum[3][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(s2l_map_f[in[k]], coeff_buf, sum[k]);
			}
			in += 4;
			coeff_buf += 4;
		}
		dump_out(out, sum, 3);
		out[3] = 0;
		out += 4;
	}
}

static void xscale_down_rgb(unsigned char *in, float *out,
	int out_width, float *coeff_buf, int *border_buf)
{
	int i, j, k;
	float sum[3][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(s2l_map_f[in[k]], coeff_buf, sum[k]);
			}
			in += 3;
			coeff_buf += 4;
		}
		dump_out(out, sum, 3);
		out += 3;
	}
}

static void xscale_down_g(unsigned char *in, float *out,
	int out_width, float *coeff_buf, int *border_buf)
{
	int i, j;
	float sum[4] = { 0.0f };

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			add_sample_to_sum_f(in[0] / 255.0f, coeff_buf, sum);
			in += 1;
			coeff_buf += 4;
		}
		out[0] = sum[0];
		shift_left_f(sum);
		out += 1;
	}
}

static void xscale_down_cmyk(unsigned char *in, float *out,
	int out_width, float *coeff_buf, int *border_buf)
{
	int i, j, k;
	float sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			for (k=0; k<4; k++) {
				add_sample_to_sum_f(in[k] / 255.0f, coeff_buf, sum[k]);
			}
			in += 4;
			coeff_buf += 4;
		}
		dump_out(out, sum, 4);
		out += 4;
	}
}

static void xscale_down_rgba(unsigned char *in, float *out,
	int out_width, float *coeff_buf, int *border_buf)
{
	int i, j, k;
	float alpha, sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			alpha = in[3] / 255.0f;
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(s2l_map_f[in[k]] * alpha, coeff_buf, sum[k]);
			}
			add_sample_to_sum_f(alpha, coeff_buf, sum[3]);
			in += 4;
			coeff_buf += 4;
		}
		dump_out(out, sum, 4);
		out += 4;
	}
}

static void xscale_down_ga(unsigned char *in, float *out,
	int out_width, float *coeff_buf, int *border_buf)
{
	int i, j;
	float alpha, sum[2][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			alpha = in[1] / 255.0f;
			add_sample_to_sum_f(in[0] / 255.0f * alpha, coeff_buf, sum[0]);
			add_sample_to_sum_f(alpha, coeff_buf, sum[1]);
			in += 2;
			coeff_buf += 4;
		}
		dump_out(out, sum, 2);
		out += 2;
	}
}

static void oil_xscale_down(unsigned char *in, float *out,
	int width_out, enum oil_colorspace cs_in, float *coeff_buf,
	int *border_buf)
{
	switch(cs_in) {
	case OIL_CS_RGBX:
		xscale_down_rgbx(in, out, width_out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGB:
		xscale_down_rgb(in, out, width_out, coeff_buf, border_buf);
		break;
	case OIL_CS_G:
		xscale_down_g(in, out, width_out, coeff_buf, border_buf);
		break;
	case OIL_CS_CMYK:
		xscale_down_cmyk(in, out, width_out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA:
		xscale_down_rgba(in, out, width_out, coeff_buf, border_buf);
		break;
	case OIL_CS_GA:
		xscale_down_ga(in, out, width_out, coeff_buf, border_buf);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void xscale_up_reduce_n(float in[][4], float *out, float *coeffs,
	int cmp)
{
	int i, j;

	for (i=0; i<cmp; i++) {
		out[i] = 0;
		for (j=0; j<4; j++) {
			out[i] += in[i][j] * coeffs[j];
		}
	}
}

static void xscale_up_rgbx(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[3][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<3; j++) {
			push_f(smp[j], s2l_map_f[in[j]]);
		}
		for (j=border_buf[i]; j>0; j--) {
			xscale_up_reduce_n(smp, out, coeff_buf, 3);
			out[3] = 0;
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void xscale_up_rgb(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[3][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<3; j++) {
			push_f(smp[j], s2l_map_f[in[j]]);
		}
		for (j=border_buf[i]; j>0; j--) {
			xscale_up_reduce_n(smp, out, coeff_buf, 3);
			out += 3;
			coeff_buf += 4;
		}
		in += 3;
	}
}

static void xscale_up_cmyk(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<4; j++) {
			push_f(smp[j], in[j] / 255.0f);
		}
		for (j=border_buf[i]; j>0; j--) {
			xscale_up_reduce_n(smp, out, coeff_buf, 4);
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void xscale_up_rgba(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4][4] = {{0}};

	for (i=0; i<width_in; i++) {
		push_f(smp[3], in[3] / 255.0f);
		for (j=0; j<3; j++) {
			push_f(smp[j], smp[3][3] * s2l_map_f[in[j]]);
		}
		for (j=border_buf[i]; j>0; j--) {
			xscale_up_reduce_n(smp, out, coeff_buf, 4);
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void xscale_up_ga(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[2][4] = {{0}};

	for (i=0; i<width_in; i++) {
		push_f(smp[1], in[1] / 255.0f);
		push_f(smp[0], smp[1][3] * in[0] / 255.0f);
		for (j=border_buf[i]; j>0; j--) {
			xscale_up_reduce_n(smp, out, coeff_buf, 2);
			out += 2;
			coeff_buf += 4;
		}
		in += 2;
	}
}

static void xscale_up_g(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j, k;
	float smp[4] = {0};

	for (i=0; i<width_in; i++) {
		push_f(smp, in[0] / 255.0f);
		for (j=border_buf[i]; j>0; j--) {
			out[0] = 0;
			for (k=0; k<4; k++) {
				out[0] += smp[k] * coeff_buf[k];
			}
			out += 1;
			coeff_buf += 4;
		}
		in += 1;
	}
}

static void oil_xscale_up(unsigned char *in, int width_in, float *out,
	enum oil_colorspace cs_in, float *coeff_buf, int *border_buf)
{
	switch(cs_in) {
	case OIL_CS_RGBX:
		xscale_up_rgbx(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGB:
		xscale_up_rgb(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_G:
		xscale_up_g(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_CMYK:
		xscale_up_cmyk(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA:
		xscale_up_rgba(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_GA:
		xscale_up_ga(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

/* Global function helpers */

/**
 * Given an oil_scale struct, map the next output scanline to a position &
 * offset in the input image.
 */
static int yscaler_map_pos(struct oil_scale *ys, float *ty)
{
	int target;
	target = split_map(ys->in_height, ys->out_height, ys->out_pos, ty);
	return target + ys->taps / 2;
}

/**
 * Return the index of the buffered scanline to use for the tap at position
 * pos.
 */
static int oil_yscaler_safe_idx(struct oil_scale *ys, int pos)
{
	int ret, max_height;

	max_height = ys->in_height - 1;
	ret = ys->target - ys->taps + 1 + pos;
	if (ret < 0) {
		return 0;
	} else if (ret > max_height) {
		return max_height;
	}
	return ret;
}

/* Global functions */
void oil_global_init()
{
	build_s2l();
	build_l2s_rights();
}

int oil_scale_init(struct oil_scale *os, int in_height, int out_height,
	int in_width, int out_width, enum oil_colorspace cs)
{
	float *tmp_coeffs;

	if (!os || in_height > MAX_DIMENSION || out_height > MAX_DIMENSION ||
		in_height < 1 || out_height < 1 ||
		in_width > MAX_DIMENSION || out_width > MAX_DIMENSION ||
		in_width < 1 || out_width < 1) {
		return -1;
	}

	/* Lazy perform global init */
	if (!s2l_map_f[128]) {
		oil_global_init();
	}

	os->in_height = in_height;
	os->out_height = out_height;
	os->in_width = in_width;
	os->out_width = out_width;
	os->cs = cs;
	os->in_pos = 0;
	os->out_pos = 0;
	os->taps = calc_taps(in_height, out_height);
	os->target = yscaler_map_pos(os, &os->ty);
	os->sl_len = out_width * OIL_CMP(cs);
	os->coeffs_y = NULL;
	os->coeffs_x = NULL;
	os->borders = NULL;
	os->rb = NULL;
	os->virt = NULL;

	/**
	 * If we are horizontally shrinking, then allocate & pre-calculate
	 * coefficients.
	 */
	if (out_width <= in_width) {
		os->coeffs_x = malloc(128 * in_width);
		os->borders = malloc(sizeof(int) * out_width);
		if (!os->coeffs_x || !os->borders) {
			oil_scale_free(os);
			return -2;
		}

		tmp_coeffs = malloc(sizeof(float) * os->taps);
		xscale_calc_coeffs(in_width, out_width, os->coeffs_x,
			os->borders, tmp_coeffs);
		free(tmp_coeffs);
	} else {
		os->coeffs_x = calloc(1, 4 * sizeof(float) * out_width);
		os->borders = calloc(1, sizeof(int) * in_width);
		if (!os->coeffs_x || !os->borders) {
			oil_scale_free(os);
			return -2;
		}
		scale_up_coeffs(in_width, out_width, os->coeffs_x,
			os->borders);
	}

	os->rb = malloc((long)os->sl_len * os->taps * sizeof(float));
	os->virt = malloc(os->taps * sizeof(float*));
	os->coeffs_y = malloc(os->taps * sizeof(float));
	if (!os->rb || !os->virt || !os->coeffs_y) {
		oil_scale_free(os);
		return -2;
	}

	return 0;
}

void oil_scale_free(struct oil_scale *os)
{
	if (!os) {
		return;
	}
	if (os->virt) {
		free(os->virt);
		os->virt = NULL;
	}
	if (os->rb) {
		free(os->rb);
		os->rb = NULL;
	}
	if (os->coeffs_y) {
		free(os->coeffs_y);
		os->coeffs_y = NULL;
	}

	if (os->coeffs_x) {
		free(os->coeffs_x);
		os->coeffs_x = NULL;
	}
	if (os->borders) {
		free(os->borders);
		os->borders = NULL;
	}
}

int oil_scale_slots(struct oil_scale *ys)
{
	int tmp, safe_target;
	tmp = ys->target + 1;
	safe_target = tmp > ys->in_height ? ys->in_height : tmp;
	return safe_target - ys->in_pos;
}

void oil_scale_in(struct oil_scale *os, unsigned char *in)
{
	float *tmp;

	tmp = os->rb + (os->in_pos % os->taps) * os->sl_len;
	os->in_pos++;
	if (os->out_height <= os->in_height) {
		oil_xscale_down(in, tmp, os->out_width, os->cs, os->coeffs_x,
			os->borders);
	} else {
		oil_xscale_up(in, os->in_width, tmp, os->cs, os->coeffs_x,
			os->borders);
	}
}

void oil_scale_out(struct oil_scale *ys, unsigned char *out)
{
	int i, idx;

	if (!ys || !out) {
		return;
	}

	for (i=0; i<ys->taps; i++) {
		idx = oil_yscaler_safe_idx(ys, i);
		ys->virt[i] = ys->rb + (idx % ys->taps) * ys->sl_len;
	}
	strip_scale(ys->virt, ys->taps, ys->sl_len, out, ys->coeffs_y, ys->ty,
		ys->cs);
	ys->out_pos++;
	ys->target = yscaler_map_pos(ys, &ys->ty);
}

int oil_fix_ratio(int src_width, int src_height, int *out_width,
	int *out_height)
{
	double width_ratio, height_ratio, tmp;
	int *adjust_dim;

	if (src_width < 1 || src_height < 1 || *out_width < 1 || *out_height < 1) {
		return -1; // bad argument
	}

	width_ratio = *out_width / (double)src_width;
	height_ratio = *out_height / (double)src_height;
	if (width_ratio < height_ratio) {
		tmp = round(width_ratio * src_height);
		adjust_dim = out_height;
	} else {
		tmp = round(height_ratio * src_width);
		adjust_dim = out_width;
	}
	if (tmp > INT_MAX) {
		return -2; // adjusted dimension out of range
	}
	*adjust_dim = tmp ? tmp : 1;
	return 0;
}
