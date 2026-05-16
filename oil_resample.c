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
#include <stdint.h>
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
static int clamp8(float x)
{
	return f2i(clampf(x) * 255.0f);
}

/**
 * Return the maximum number of input samples that can contribute to any single
 * output sample. Used only for buffer allocation.
 */
static int max_taps(int dim_in, int dim_out)
{
	if (dim_out >= dim_in) {
		return TAPS;
	}
	return (int)ceil(4.0 * dim_in / dim_out) + 1;
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
 * Pre-calculated linear-to-sRGB table. All entries map the [0, 1] linear
 * range; callers clamp their input before indexing. Catmull-Rom's negative
 * lobe can drive intermediates outside [0, 1] - and RGBA/ARGB unpremul can
 * send R_pre / alpha arbitrarily large - so every gamma-aware output path
 * clamps before the lookup.
 */
#define L2S_ALL_LEN 22000
static unsigned char l2s_map_storage[L2S_ALL_LEN];
int l2s_len = L2S_ALL_LEN;
unsigned char *l2s_map = l2s_map_storage;

static void build_l2s(void)
{
	int i;
	double srgb_f, tmp, val;

	for (i=0; i<L2S_ALL_LEN; i++) {
		srgb_f = (i + 0.5)/(L2S_ALL_LEN - 1);
		if (srgb_f <= 0.00313) {
			val = srgb_f * 12.92;
		} else {
			tmp = pow(srgb_f, 1/2.4);
			val = 1.055 * tmp - 0.055;
		}

		l2s_map[i] = round(val * 255);
	}
}

/**
 * Maps the given linear RGB float to sRGB integer.
 */
static unsigned char linear_sample_to_srgb(float in)
{
	return l2s_map[(int)(in * (l2s_len - 1))];
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
 * set to the given value.
 */
static void push_f(float *f, float val)
{
	int j;
	for (j=0; j<3; j++) {
		f[j] = f[j + 1];
	}
	f[3] = val;
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

static void yscale_out_linear(float *sums, int len, unsigned char *out)
{
	int i;
	for (i=0; i<len; i++) {
		out[i] = linear_sample_to_srgb(clampf(*sums));
		shift_left_f(sums);
		sums += 4;
	}
}

static void yscale_out_nonlinear(float *sums, int sl_len, unsigned char *out)
{
	int i;

	for (i=0; i<sl_len; i++) {
		out[i] = clamp8(*sums);
		shift_left_f(sums);
		sums += 4;
	}
}


static void yscale_out_ga(float *sums, int width, unsigned char *out)
{
	int i;
	float alpha;

	for (i=0; i<width; i++) {
		alpha = clampf(sums[4]);
		if (alpha != 0) {
			sums[0] /= alpha;
		}
		out[0] = clamp8(sums[0]);
		shift_left_f(sums);
		out[1] = f2i(alpha * 255.0f);
		shift_left_f(sums + 4);
		sums += 8;
		out += 2;
	}
}

static void yscale_out_cmyk(float *sums, int width, unsigned char *out, int tap)
{
	int i, j, tap_off;

	tap_off = tap * 4;
	for (i=0; i<width; i++) {
		for (j=0; j<4; j++) {
			out[j] = clamp8(sums[tap_off + j]);
			sums[tap_off + j] = 0.0f;
		}
		sums += 16;
		out += 4;
	}
}

static void yscale_out_rgba(float *sums, int width, unsigned char *out, int tap)
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
			out[j] = linear_sample_to_srgb(clampf(val));
			sums[tap_off + j] = 0.0f;
		}
		out[3] = round(alpha * 255.0f);
		sums[tap_off + 3] = 0.0f;
		sums += 16;
		out += 4;
	}
}


static void oil_yscale_out_argb(float *sums, int width, unsigned char *out, int tap)
{
	int i, j, tap_off;
	float alpha, val;

	tap_off = tap * 4;
	for (i=0; i<width; i++) {
		alpha = clampf(sums[tap_off + 3]);
		out[0] = round(alpha * 255.0f);
		for (j=0; j<3; j++) {
			val = sums[tap_off + j];
			if (alpha != 0) {
				val /= alpha;
			}
			out[j + 1] = linear_sample_to_srgb(clampf(val));
			sums[tap_off + j] = 0.0f;
		}
		sums[tap_off + 3] = 0.0f;
		sums += 16;
		out += 4;
	}
}

static void yscale_out_rgbx(float *sums, int width, unsigned char *out, int tap)
{
	int i, j, tap_off;

	tap_off = tap * 4;
	for (i=0; i<width; i++) {
		for (j=0; j<3; j++) {
			out[j] = linear_sample_to_srgb(clampf(sums[tap_off + j]));
			sums[tap_off + j] = 0.0f;
		}
		out[3] = 255;
		sums[tap_off + 3] = 0.0f;
		sums += 16;
		out += 4;
	}
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
	int sl_len;

	sl_len = width * OIL_CMP(cs);

	switch(cs) {
	case OIL_CS_G:
		yscale_out_nonlinear(sums, sl_len, out);
		break;
	case OIL_CS_CMYK:
		yscale_out_cmyk(sums, width, out, tap);
		break;
	case OIL_CS_GA:
		yscale_out_ga(sums, width, out);
		break;
	case OIL_CS_RGB:
		yscale_out_linear(sums, sl_len, out);
		break;
	case OIL_CS_RGBA:
		yscale_out_rgba(sums, width, out, tap);
		break;
	case OIL_CS_ARGB:
		oil_yscale_out_argb(sums, width, out, tap);
		break;
	case OIL_CS_RGBX:
		yscale_out_rgbx(sums, width, out, tap);
		break;
	case OIL_CS_RGB_NOGAMMA:
		yscale_out_nonlinear(sums, sl_len, out);
		break;
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

static void yscale_up_g_cmyk(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float sum;

	for (i=0; i<len; i++) {
		sum = coeffs[0] * in[0][i] +
			coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] +
			coeffs[3] * in[3][i];
		out[i] = clamp8(sum);
	}
}

static void yscale_up_ga(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i, j;
	float alpha, sums[2];

	for (i=0; i<len; i+=2) {
		for (j=0; j<2; j++) {
			sums[j] = coeffs[0] * in[0][i + j] +
				coeffs[1] * in[1][i + j] +
				coeffs[2] * in[2][i + j] +
				coeffs[3] * in[3][i + j];
		}
		alpha = clampf(sums[1]);
		if (alpha != 0) {
			sums[0] /= alpha;
		}
		out[i] = clamp8(sums[0]);
		out[i + 1] = f2i(alpha * 255.0f);
	}
}

static void yscale_up_rgb(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float sum;

	for (i=0; i<len; i++) {
		sum = coeffs[0] * in[0][i] +
			coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] +
			coeffs[3] * in[3][i];
		out[i] = linear_sample_to_srgb(clampf(sum));
	}
}

static void yscale_up_rgba(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i, j;
	float alpha, sums[4];

	for (i=0; i<len; i+=4) {
		for (j=0; j<4; j++) {
			sums[j] = coeffs[0] * in[0][i + j] +
				coeffs[1] * in[1][i + j] +
				coeffs[2] * in[2][i + j] +
				coeffs[3] * in[3][i + j];
		}
		alpha = clampf(sums[3]);
		for (j=0; j<3; j++) {
			if (alpha != 0 && alpha != 1.0f) {
				sums[j] /= alpha;
			}
			out[i + j] = linear_sample_to_srgb(clampf(sums[j]));
		}
		out[i + 3] = f2i(alpha * 255.0f);
	}
}

static void oil_yscale_up_argb(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i, j;
	float alpha, sums[4];

	for (i=0; i<len; i+=4) {
		for (j=0; j<4; j++) {
			sums[j] = coeffs[0] * in[0][i + j] +
				coeffs[1] * in[1][i + j] +
				coeffs[2] * in[2][i + j] +
				coeffs[3] * in[3][i + j];
		}
		alpha = clampf(sums[3]);
		out[i] = f2i(alpha * 255.0f);
		for (j=0; j<3; j++) {
			if (alpha != 0 && alpha != 1.0f) {
				sums[j] /= alpha;
			}
			out[i + j + 1] = linear_sample_to_srgb(clampf(sums[j]));
		}
	}
}

static void yscale_up_rgbx(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i, j;
	float sums[3];

	for (i=0; i<len; i+=4) {
		for (j=0; j<3; j++) {
			sums[j] = coeffs[0] * in[0][i + j] +
				coeffs[1] * in[1][i + j] +
				coeffs[2] * in[2][i + j] +
				coeffs[3] * in[3][i + j];
		}
		for (j=0; j<3; j++) {
			out[i + j] = linear_sample_to_srgb(clampf(sums[j]));
		}
		out[i + 3] = 255;
	}
}

static void yscale_up_rgba_nogamma(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i, j;
	float alpha, sums[4];

	for (i=0; i<len; i+=4) {
		for (j=0; j<4; j++) {
			sums[j] = coeffs[0] * in[0][i + j] +
				coeffs[1] * in[1][i + j] +
				coeffs[2] * in[2][i + j] +
				coeffs[3] * in[3][i + j];
		}
		alpha = clampf(sums[3]);
		for (j=0; j<3; j++) {
			if (alpha != 0 && alpha != 1.0f) {
				sums[j] /= alpha;
				sums[j] = clampf(sums[j]);
			}
			out[i + j] = f2i(clampf(sums[j]) * 255.0f);
		}
		out[i + 3] = f2i(alpha * 255.0f);
	}
}

static void yscale_up_rgbx_nogamma(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i, j;
	float sums[3];

	for (i=0; i<len; i+=4) {
		for (j=0; j<3; j++) {
			sums[j] = coeffs[0] * in[0][i + j] +
				coeffs[1] * in[1][i + j] +
				coeffs[2] * in[2][i + j] +
				coeffs[3] * in[3][i + j];
		}
		for (j=0; j<3; j++) {
			out[i + j] = f2i(clampf(sums[j]) * 255.0f);
		}
		out[i + 3] = 255;
	}
}

/**
 * Upscale a strip of scanlines. Branches to the correct interpolator using
 * the given colorspace.
 */
static void yscale_up(float **in, int len, float *coeffs, unsigned char *out,
	enum oil_colorspace cs)
{
	switch(cs) {
	case OIL_CS_G:
	case OIL_CS_CMYK:
		yscale_up_g_cmyk(in, len, coeffs, out);
		break;
	case OIL_CS_GA:
		yscale_up_ga(in, len, coeffs, out);
		break;
	case OIL_CS_RGB:
		yscale_up_rgb(in, len, coeffs, out);
		break;
	case OIL_CS_RGBA:
		yscale_up_rgba(in, len, coeffs, out);
		break;
	case OIL_CS_ARGB:
		oil_yscale_up_argb(in, len, coeffs, out);
		break;
	case OIL_CS_RGBX:
		yscale_up_rgbx(in, len, coeffs, out);
		break;
	case OIL_CS_RGB_NOGAMMA:
		yscale_up_g_cmyk(in, len, coeffs, out);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		yscale_up_rgba_nogamma(in, len, coeffs, out);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		yscale_up_rgbx_nogamma(in, len, coeffs, out);
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
float s2l_map[256];

/**
 * Populates s2l_map.
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
		s2l_map[input] = val;
	}
}

float i2f_map[256];

static void build_i2f(void)
{
	int i;

	for (i=0; i<=255; i++) {
		i2f_map[i] = i / 255.0f;
	}
}

/**
 * Given fed buffer width, logical source width, source offset, and output
 * width, populate a buffer of coefficients and border counters.
 *
 * src_dim drives the scale factor (tap_mult = src_dim / out_dim) and src_off
 * shifts the reverse map within the fed buffer to express a sub-pixel crop.
 *
 * Generates 4 * fed_dim coefficients -- 4 for every fed input sample. Samples
 * outside [src_off-halo, src_off+src_dim+halo] receive zero coefficients (the
 * caller zeros coeff_buf before calling); they are still consumed by the
 * border counters so the resampler advances through the full fed buffer.
 *
 * Generates out_dim border counters that sum to fed_dim, indicating how many
 * fed samples to consume before each output is finished. The final border is
 * extended to cover any trailing halo so the caller can feed all fed_dim rows.
 */
static void scale_down_coeffs(int fed_dim, double src_dim, double src_off,
	int out_dim, float *coeff_buf, int *border_buf, float *tmp_coeffs)
{
	int i, j, offset, pos, smp_end, smp_start, n_samples, ends[4];
	float fudge;
	double tap_mult, radius, center, left_edge, right_edge, dist;

	tap_mult = src_dim / out_dim;
	radius = 2.0 * tap_mult;
	center = src_off + 0.5 * tap_mult - 0.5;

	for (i=0; i<4; i++) {
		ends[i] = -1;
	}

	for (i=0; i<out_dim; i++) {
		left_edge = center - radius;
		right_edge = center + radius;
		/* floor(left)+1 / ceil(right)-1 collapse the original ceil/floor
		 * + equality-bump pair into one rounding op each. Excludes
		 * samples at exactly distance +/-2 (catrom = 0 there). */
		smp_start = (int)floor(left_edge) + 1;
		smp_end = (int)ceil(right_edge) - 1;
		if (smp_start < 0) {
			smp_start = 0;
		}
		if (smp_end >= fed_dim) {
			smp_end = fed_dim - 1;
		}
		n_samples = smp_end - smp_start + 1;

		ends[i%4] = smp_end;
		border_buf[i] = smp_end - ends[(i+3)%4];

		fudge = 0.0f;
		for (j=0; j<n_samples; j++) {
			dist = fabs((double)(smp_start + j) - center);
			tmp_coeffs[j] = catrom((float)(dist / tap_mult)) / (float)tap_mult;
			fudge += tmp_coeffs[j];
		}
		fudge = 1.0f / fudge;
		for (j=0; j<n_samples; j++) {
			tmp_coeffs[j] *= fudge;
		}

		for (j=0; j<n_samples; j++) {
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

		center += tap_mult;
	}

	/* If trailing halo extends past the last sampled position, fold it
	 * into the final border so the caller's fed_dim scanlines are all
	 * consumed. Those samples have zero coefficients (no output sampled
	 * them) so they contribute nothing to the result. */
	if (out_dim > 0 && ends[(out_dim - 1) % 4] < fed_dim - 1) {
		border_buf[out_dim - 1] += fed_dim - 1 - ends[(out_dim - 1) % 4];
	}
}

/**
 * Precalculate coefficients and borders for an upscale.
 *
 * src_dim drives the scale factor (step = src_dim / out_dim) and src_off
 * shifts the reverse map within the fed buffer. coeff_buf gets 4 entries per
 * output sample; border_buf is fed_dim entries (indices outside the sampled
 * region remain 0).
 */
static void scale_up_coeffs(int fed_dim, double src_dim, double src_off,
	int out_dim, float *coeff_buf, int *border_buf)
{
	int i, smp_i, start, end, ltrim, rtrim, safe_end, max_pos;
	double pos_d, step;
	float tx;

	max_pos = fed_dim - 1;
	step = src_dim / out_dim;
	pos_d = src_off + 0.5 * step - 0.5;

	for (i=0; i<out_dim; i++) {
		/* split_map inlined: smp_i is floor toward -inf, but pos_d in
		 * (-0.5, 0) maps to smp_i = -1. */
		if (pos_d < 0.0) {
			smp_i = -1;
			tx = (float)(pos_d + 1.0);
		} else {
			smp_i = (int)pos_d;
			tx = (float)(pos_d - smp_i);
		}
		start = smp_i - 1;
		end = smp_i + 2;

		// This is the border position at which we will tell the
		// interpolator to calculate the output sample.
		safe_end = min(end, max_pos);
		if (safe_end < 0) {
			safe_end = 0;
		}

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
		pos_d += step;
	}
}


static void scale_down_rgb(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y)
{
	int i, j, k;
	float sum[3][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(s2l_map[in[k]], coeffs_x, sum[k]);
			}
			in += 3;
			coeffs_x += 4;
		}

		for (j=0; j<3; j++) {
			add_sample_to_sum_f(sum[j][0], coeffs_y, sums_y);
			shift_left_f(sum[j]);
			sums_y += 4;
		}
	}
}

static void scale_down_g(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y)
{
	int i, j;
	float sum[4] = { 0.0f };

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			add_sample_to_sum_f(i2f_map[in[0]], coeffs_x, sum);
			in += 1;
			coeffs_x += 4;
		}
		add_sample_to_sum_f(sum[0], coeffs_y, sums_y);
		shift_left_f(sum);
		sums_y += 4;
	}
}

static void scale_down_cmyk(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y, int tap)
{
	int i, j;
	float sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			unsigned int px;
			memcpy(&px, in, 4);
			add_sample_to_sum_f(i2f_map[px & 0xFF], coeffs_x, sum[0]);
			add_sample_to_sum_f(i2f_map[(px >> 8) & 0xFF], coeffs_x, sum[1]);
			add_sample_to_sum_f(i2f_map[(px >> 16) & 0xFF], coeffs_x, sum[2]);
			add_sample_to_sum_f(i2f_map[px >> 24], coeffs_x, sum[3]);
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

static void scale_down_rgba(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y, int tap)
{
	int i, j, k;
	float alpha, sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			alpha = i2f_map[in[3]];
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(s2l_map[in[k]] * alpha, coeffs_x, sum[k]);
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

static void scale_down_ga(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y)
{
	int i, j;
	float alpha, sum[2][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			alpha = i2f_map[in[1]];
			add_sample_to_sum_f(i2f_map[in[0]] * alpha, coeffs_x, sum[0]);
			add_sample_to_sum_f(alpha, coeffs_x, sum[1]);
			in += 2;
			coeffs_x += 4;
		}

		for (j=0; j<2; j++) {
			add_sample_to_sum_f(sum[j][0], coeffs_y, sums_y);
			shift_left_f(sum[j]);
			sums_y += 4;
		}
	}
}

static void scale_down_rgb_nogamma(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y)
{
	int i, j, k;
	float sum[3][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(i2f_map[in[k]], coeffs_x, sum[k]);
			}
			in += 3;
			coeffs_x += 4;
		}

		for (j=0; j<3; j++) {
			add_sample_to_sum_f(sum[j][0], coeffs_y, sums_y);
			shift_left_f(sum[j]);
			sums_y += 4;
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

static void oil_scale_down_argb(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y, int tap)
{
	int i, j, k;
	float alpha, sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			alpha = i2f_map[in[0]];
			for (k=0; k<3; k++) {
				add_sample_to_sum_f(s2l_map[in[k + 1]] * alpha, coeffs_x, sum[k]);
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

static void scale_down_rgbx(unsigned char *in, float *sums_y, int out_width, float *coeffs_x,
	int *border_buf, float *coeffs_y, int tap)
{
	int i, j;
	float sum[4][4] = {{ 0.0f }};

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			unsigned int px;
			memcpy(&px, in, 4);
			add_sample_to_sum_f(s2l_map[px & 0xFF], coeffs_x, sum[0]);
			add_sample_to_sum_f(s2l_map[(px >> 8) & 0xFF], coeffs_x, sum[1]);
			add_sample_to_sum_f(s2l_map[(px >> 16) & 0xFF], coeffs_x, sum[2]);
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

static void xscale_up_reduce_n(float in[][4], float *out, float *coeffs,
	int cmp)
{
	int i;

	for (i=0; i<cmp; i++) {
		out[i] = in[i][0] * coeffs[0] +
			in[i][1] * coeffs[1] +
			in[i][2] * coeffs[2] +
			in[i][3] * coeffs[3];
	}
}

static void xscale_up_rgb(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[3][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<3; j++) {
			push_f(smp[j], s2l_map[in[j]]);
		}
		for (j=0; j<border_buf[i]; j++) {
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
		for (j=0; j<border_buf[i]; j++) {
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
			push_f(smp[j], smp[3][3] * s2l_map[in[j]]);
		}
		for (j=0; j<border_buf[i]; j++) {
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
		push_f(smp[0], smp[1][3] * i2f_map[in[0]]);
		for (j=0; j<border_buf[i]; j++) {
			xscale_up_reduce_n(smp, out, coeff_buf, 2);
			out += 2;
			coeff_buf += 4;
		}
		in += 2;
	}
}

static void oil_xscale_up_argb(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4][4] = {{0}};

	for (i=0; i<width_in; i++) {
		push_f(smp[3], in[0] / 255.0f);
		for (j=0; j<3; j++) {
			push_f(smp[j], smp[3][3] * s2l_map[in[j + 1]]);
		}
		for (j=0; j<border_buf[i]; j++) {
			xscale_up_reduce_n(smp, out, coeff_buf, 4);
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void xscale_up_rgbx(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<3; j++) {
			push_f(smp[j], s2l_map[in[j]]);
		}
		push_f(smp[3], 1.0f);
		for (j=0; j<border_buf[i]; j++) {
			xscale_up_reduce_n(smp, out, coeff_buf, 4);
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void xscale_up_g(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4] = {0};

	for (i=0; i<width_in; i++) {
		push_f(smp, in[i] / 255.0f);
		for (j=0; j<border_buf[i]; j++) {
			out[0] = smp[0] * coeff_buf[0] +
				smp[1] * coeff_buf[1] +
				smp[2] * coeff_buf[2] +
				smp[3] * coeff_buf[3];
			out += 1;
			coeff_buf += 4;
		}
	}
}

static void xscale_up_rgb_nogamma(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[3][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<3; j++) {
			push_f(smp[j], i2f_map[in[j]]);
		}
		for (j=0; j<border_buf[i]; j++) {
			xscale_up_reduce_n(smp, out, coeff_buf, 3);
			out += 3;
			coeff_buf += 4;
		}
		in += 3;
	}
}

static void xscale_up_rgba_nogamma(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4][4] = {{0}};

	for (i=0; i<width_in; i++) {
		push_f(smp[3], in[3] / 255.0f);
		for (j=0; j<3; j++) {
			push_f(smp[j], smp[3][3] * i2f_map[in[j]]);
		}
		for (j=0; j<border_buf[i]; j++) {
			xscale_up_reduce_n(smp, out, coeff_buf, 4);
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void xscale_up_rgbx_nogamma(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float smp[4][4] = {{0}};

	for (i=0; i<width_in; i++) {
		for (j=0; j<3; j++) {
			push_f(smp[j], i2f_map[in[j]]);
		}
		push_f(smp[3], 1.0f);
		for (j=0; j<border_buf[i]; j++) {
			xscale_up_reduce_n(smp, out, coeff_buf, 4);
			out += 4;
			coeff_buf += 4;
		}
		in += 4;
	}
}

static void oil_xscale_up(unsigned char *in, int width_in, float *out,
	enum oil_colorspace cs_in, float *coeff_buf, int *border_buf)
{
	switch(cs_in) {
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
	case OIL_CS_ARGB:
		oil_xscale_up_argb(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBX:
		xscale_up_rgbx(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGB_NOGAMMA:
		xscale_up_rgb_nogamma(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		xscale_up_rgba_nogamma(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		xscale_up_rgbx_nogamma(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

/* Global functions */
void oil_global_init(void)
{
	build_s2l();
	build_l2s();
	build_i2f();
}

#define ALIGN16(x) (((x) + 15) & ~15)

static int calc_coeffs_len(int in_dim, int out_dim)
{
	return TAPS * max(in_dim, out_dim) * sizeof(float);
}

static int upscale_alloc_size(int fed_height, int out_height, int fed_width,
	int out_width, enum oil_colorspace cs)
{
	/* For upscale the coeff buffer is indexed by output sample (4 per
	 * output) and the border buffer is indexed by fed input sample. With
	 * crop padding, fed_dim may exceed out_dim, so size borders by fed. */
	return ALIGN16(calc_coeffs_len(fed_width, out_width))
		+ ALIGN16(fed_width * sizeof(int))
		+ ALIGN16(calc_coeffs_len(fed_height, out_height))
		+ ALIGN16(fed_height * sizeof(int))
		+ ALIGN16(out_width * OIL_CMP(cs) * TAPS * sizeof(float));
}

static void upscale_init(struct oil_scale *os)
{
	int coeffs_x_len, coeffs_y_len, borders_x_len, borders_y_len;
	char *p;

	coeffs_x_len = ALIGN16(calc_coeffs_len(os->in_width, os->out_width));
	borders_x_len = ALIGN16(os->in_width * sizeof(int));
	coeffs_y_len = ALIGN16(calc_coeffs_len(os->in_height, os->out_height));
	borders_y_len = ALIGN16(os->in_height * sizeof(int));

	p = os->buf;
	os->coeffs_x = (float *)p;		p += coeffs_x_len;
	os->borders_x = (int *)p;		p += borders_x_len;
	os->coeffs_y = (float *)p;		p += coeffs_y_len;
	os->borders_y = (int *)p;		p += borders_y_len;
	os->rb = (float *)p;

	scale_up_coeffs(os->in_width, os->src_width, os->src_x_off,
		os->out_width, os->coeffs_x, os->borders_x);
	scale_up_coeffs(os->in_height, os->src_height, os->src_y_off,
		os->out_height, os->coeffs_y, os->borders_y);
	os->slots_y = 0;
}

static int downscale_alloc_size(int fed_height, int out_height, int fed_width,
	int out_width, double src_height, double src_width,
	enum oil_colorspace cs)
{
	int taps_x, taps_y, ceil_src_w, ceil_src_h;

	/* max_taps is bounded by the kernel span in source pixels. Use the
	 * ceiling of src_dim so we don't under-allocate; passing fed_dim
	 * would also be safe but over-allocates with halo. */
	ceil_src_w = (int)ceil(src_width);
	ceil_src_h = (int)ceil(src_height);
	taps_x = max_taps(ceil_src_w, out_width);
	taps_y = max_taps(ceil_src_h, out_height);

	return ALIGN16(calc_coeffs_len(fed_width, out_width))
		+ ALIGN16(out_width * sizeof(int))
		+ ALIGN16(calc_coeffs_len(fed_height, out_height))
		+ ALIGN16(out_height * sizeof(int))
		+ ALIGN16(max(taps_x, taps_y) * sizeof(float))
		+ ALIGN16(out_width * OIL_CMP(cs) * TAPS * sizeof(float));
}

static void downscale_init(struct oil_scale *os)
{
	int coeffs_x_len, coeffs_y_len, borders_x_len, borders_y_len, sums_len;
	char *p;

	coeffs_x_len = ALIGN16(calc_coeffs_len(os->in_width, os->out_width));
	borders_x_len = ALIGN16(os->out_width * sizeof(int));
	coeffs_y_len = ALIGN16(calc_coeffs_len(os->in_height, os->out_height));
	borders_y_len = ALIGN16(os->out_height * sizeof(int));
	sums_len = ALIGN16(os->out_width * OIL_CMP(os->cs) * TAPS * sizeof(float));

	p = os->buf;
	os->coeffs_x = (float *)p;		p += coeffs_x_len;
	os->borders_x = (int *)p;		p += borders_x_len;
	os->coeffs_y = (float *)p;		p += coeffs_y_len;
	os->borders_y = (int *)p;		p += borders_y_len;
	os->sums_y = (float *)p;		p += sums_len;
	os->tmp_coeffs = (float *)p;

	scale_down_coeffs(os->in_width, os->src_width, os->src_x_off,
		os->out_width, os->coeffs_x, os->borders_x, os->tmp_coeffs);
	scale_down_coeffs(os->in_height, os->src_height, os->src_y_off,
		os->out_height, os->coeffs_y, os->borders_y, os->tmp_coeffs);
	os->slots_y = os->borders_y[0];
}

int oil_scale_alloc_size_ex(int in_height, int out_height, int in_width,
	int out_width, double src_height, double src_width,
	enum oil_colorspace cs)
{
	if (out_width > src_width) {
		return upscale_alloc_size(in_height, out_height, in_width,
			out_width, cs);
	} else {
		return downscale_alloc_size(in_height, out_height, in_width,
			out_width, src_height, src_width, cs);
	}
}

int oil_scale_alloc_size(int in_height, int out_height, int in_width,
	int out_width, enum oil_colorspace cs)
{
	return oil_scale_alloc_size_ex(in_height, out_height, in_width,
		out_width, in_height, in_width, cs);
}

int oil_scale_init_allocated_ex(struct oil_scale *os, int in_height,
	int out_height, int in_width, int out_width, double src_y,
	double src_height, double src_x, double src_width,
	enum oil_colorspace cs, void *buf)
{
	/* sanity check on arguments */
	if (!os || !buf || in_height > MAX_DIMENSION || out_height > MAX_DIMENSION ||
		in_height < 1 || out_height < 1 ||
		in_width > MAX_DIMENSION || out_width > MAX_DIMENSION ||
		in_width < 1 || out_width < 1) {
		return -1;
	}

	/* logical source rect must fit inside the fed buffer */
	if (src_x < 0.0 || src_y < 0.0 || src_width <= 0.0 ||
		src_height <= 0.0 || src_x + src_width > (double)in_width ||
		src_y + src_height > (double)in_height) {
		return -1;
	}

	/* only allow upscaling if both dimensions are being upscaled */
	if ((out_height > src_height) != (out_width > src_width)) {
		return -1;
	}

	// Lazy perform global init, in case oil_global_ini() hasn't been
	// called yet.
	if (!s2l_map[128]) {
		oil_global_init();
	}

	memset(os, 0, sizeof(struct oil_scale));
	os->in_height = in_height;
	os->out_height = out_height;
	os->in_width = in_width;
	os->out_width = out_width;
	os->src_y_off = src_y;
	os->src_x_off = src_x;
	os->src_height = src_height;
	os->src_width = src_width;
	os->cs = cs;
	os->buf = buf;

	if (out_width > src_width) {
		upscale_init(os);
	} else {
		downscale_init(os);
	}

	return 0;
}

int oil_scale_init_allocated(struct oil_scale *os, int in_height,
	int out_height, int in_width, int out_width, enum oil_colorspace cs,
	void *buf)
{
	return oil_scale_init_allocated_ex(os, in_height, out_height, in_width,
		out_width, 0.0, (double)in_height, 0.0, (double)in_width, cs,
		buf);
}

int oil_scale_init_ex(struct oil_scale *os, int in_height, int out_height,
	int in_width, int out_width, double src_y, double src_height,
	double src_x, double src_width, enum oil_colorspace cs)
{
	int alloc_size, ret;
	void *buf;

	alloc_size = oil_scale_alloc_size_ex(in_height, out_height, in_width,
		out_width, src_height, src_width, cs);
	buf = calloc(1, alloc_size);
	if (!buf) {
		return -2;
	}

	ret = oil_scale_init_allocated_ex(os, in_height, out_height, in_width,
		out_width, src_y, src_height, src_x, src_width, cs, buf);
	if (ret) {
		free(buf);
		return ret;
	}

	return 0;
}

int oil_scale_init(struct oil_scale *os, int in_height, int out_height,
	int in_width, int out_width, enum oil_colorspace cs)
{
	return oil_scale_init_ex(os, in_height, out_height, in_width, out_width,
		0.0, (double)in_height, 0.0, (double)in_width, cs);
}

void oil_scale_restart(struct oil_scale *os)
{
	os->in_pos = os->out_pos = 0;
	os->sums_y_tap = 0;
	if ((double)os->out_height <= os->src_height) {
		/* downscale: sums_y accumulates partial output across input rows;
		 * stale state from a prior pass would corrupt the next output. */
		memset(os->sums_y, 0,
			os->out_width * OIL_CMP(os->cs) * TAPS * sizeof(float));
		os->slots_y = os->borders_y[0];
	} else {
		os->slots_y = 0;
	}
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
	os->rb = NULL;
	os->sums_y = NULL;
	os->tmp_coeffs = NULL;
}

int oil_scale_slots(struct oil_scale *ys)
{
	if ((double)ys->out_height <= ys->src_height) {
		return ys->slots_y;
	}
	if (ys->in_pos) {
		return ys->slots_y == 0;
	}
	return ys->borders_y[0] == 0 ? 2 : 1;
}

static float *get_rb_line(struct oil_scale *os, int line)
{
	int sl_len;
	sl_len = OIL_CMP(os->cs) * os->out_width;
	return os->rb + line * sl_len;
}

static void down_scale_in(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	switch(os->cs) {
	case OIL_CS_RGB:
		scale_down_rgb(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_G:
		scale_down_g(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_CMYK:
		scale_down_cmyk(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBA:
		scale_down_rgba(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_GA:
		scale_down_ga(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_ARGB:
		oil_scale_down_argb(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX:
		scale_down_rgbx(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGB_NOGAMMA:
		scale_down_rgb_nogamma(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		scale_down_rgba_nogamma(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		scale_down_rgbx_nogamma(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}

	os->slots_y -= 1;
	os->in_pos++;
}

static void up_scale_in(struct oil_scale *os, unsigned char *in)
{
	float *tmp;

	tmp = get_rb_line(os, os->in_pos % 4);
	oil_xscale_up(in, os->in_width, tmp, os->cs, os->coeffs_x, os->borders_x);

	os->in_pos++;
	os->slots_y = os->borders_y[os->in_pos - 1];
}

int oil_scale_in(struct oil_scale *os, unsigned char *in)
{
	if (oil_scale_slots(os) == 0) {
		return -1;
	}
	if ((double)os->out_width > os->src_width) {
		up_scale_in(os, in);
	} else {
		down_scale_in(os, in);
	}
	return 0;
}

int oil_scale_out(struct oil_scale *os, unsigned char *out)
{
	int i, sl_len, is_down;
	float *in[4];

	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	is_down = (double)os->out_height <= os->src_height;
	if (is_down) {
		yscale_out(os->sums_y, os->out_width, out, os->cs, os->sums_y_tap);
		os->sums_y_tap = (os->sums_y_tap + 1) & 3;
	} else {
		sl_len = OIL_CMP(os->cs) * os->out_width;
		for (i=0; i<4; i++) {
			in[i] = get_rb_line(os, (os->in_pos + i) % 4);
		}
		yscale_up(in, sl_len, os->coeffs_y + os->out_pos * 4, out,
			os->cs);
		os->slots_y -= 1;
	}

	os->out_pos++;
	if (is_down && os->out_pos < os->out_height) {
		os->slots_y = os->borders_y[os->out_pos];
	}
	return 0;
}

int oil_scale_out_discard(struct oil_scale *os)
{
	int is_down;

	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	is_down = (double)os->out_height <= os->src_height;
	if (is_down) {
		/* Use yscale_out to shift the sums_y accumulators, discarding
		 * the output pixels. This avoids needing layout-specific shift
		 * logic for each colorspace's sums_y memory layout. */
		int sl_len = os->out_width * OIL_CMP(os->cs);
		unsigned char tmp[sl_len];
		yscale_out(os->sums_y, os->out_width, tmp, os->cs, os->sums_y_tap);
		os->sums_y_tap = (os->sums_y_tap + 1) & 3;
	} else {
		os->slots_y -= 1;
	}

	os->out_pos++;
	if (is_down && os->out_pos < os->out_height) {
		os->slots_y = os->borders_y[os->out_pos];
	}
	return 0;
}

int oil_required_input_rect(int img_height, int img_width, double src_y,
	double src_height, double src_x, double src_width, int out_height,
	int out_width, int *fed_y, int *fed_height, int *fed_x, int *fed_width)
{
	double tap_mult_x, tap_mult_y, halo_x, halo_y;
	int fy0, fx0, fy1, fx1;

	if (!fed_y || !fed_height || !fed_x || !fed_width) {
		return -1;
	}
	if (img_height < 1 || img_width < 1 || out_height < 1 || out_width < 1 ||
		src_x < 0.0 || src_y < 0.0 || src_width <= 0.0 ||
		src_height <= 0.0 ||
		src_x + src_width > (double)img_width ||
		src_y + src_height > (double)img_height) {
		return -1;
	}

	/* Catmull-Rom half-support is 2 * tap_mult in source pixels;
	 * tap_mult = max(1, src_dim/out_dim). */
	tap_mult_x = src_width > (double)out_width ? src_width / out_width : 1.0;
	tap_mult_y = src_height > (double)out_height ? src_height / out_height : 1.0;
	halo_x = 2.0 * tap_mult_x;
	halo_y = 2.0 * tap_mult_y;

	fx0 = (int)floor(src_x - halo_x);
	fy0 = (int)floor(src_y - halo_y);
	fx1 = (int)ceil(src_x + src_width + halo_x);
	fy1 = (int)ceil(src_y + src_height + halo_y);

	if (fx0 < 0) fx0 = 0;
	if (fy0 < 0) fy0 = 0;
	if (fx1 > img_width) fx1 = img_width;
	if (fy1 > img_height) fy1 = img_height;

	*fed_x = fx0;
	*fed_y = fy0;
	*fed_width = fx1 - fx0;
	*fed_height = fy1 - fy0;
	return 0;
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

int oil_compute_cover_rect(int img_width, int img_height,
	int out_width, int out_height, enum oil_gravity gravity,
	double *src_x, double *src_y, double *src_width, double *src_height)
{
	double sw, sh, sx, sy;

	if (img_width < 1 || img_height < 1 || out_width < 1 || out_height < 1) {
		return -1;
	}

	/* int64 cross-product keeps the branch deterministic up to
	 * MAX_DIMENSION; doubles would lose precision near aspect equality
	 * if MAX_DIMENSION ever grew past 2^26 or so. */
	if ((int64_t)img_width * out_height > (int64_t)img_height * out_width) {
		/* Image is wider than the output aspect: crop horizontally. */
		sh = img_height;
		sw = (double)img_height * out_width / out_height;
		if (sw > (double)img_width) {
			sw = img_width;
		}
		sy = 0.0;
		switch (gravity) {
		case OIL_GRAVITY_LEFT:
		case OIL_GRAVITY_TOP_LEFT:
		case OIL_GRAVITY_BOTTOM_LEFT:
			sx = 0.0;
			break;
		case OIL_GRAVITY_RIGHT:
		case OIL_GRAVITY_TOP_RIGHT:
		case OIL_GRAVITY_BOTTOM_RIGHT:
			sx = img_width - sw;
			break;
		default:
			sx = (img_width - sw) / 2.0;
			break;
		}
		/* Floating-point rounding in the sw computation above could
		 * still leave sx as a tiny negative; oil_scale_init_ex rejects
		 * any src_x < 0 so clamp defensively. */
		if (sx < 0.0) {
			sx = 0.0;
		}
	} else {
		/* Image is taller (or equal): crop vertically (or not at all). */
		sw = img_width;
		sh = (double)img_width * out_height / out_width;
		if (sh > (double)img_height) {
			sh = img_height;
		}
		sx = 0.0;
		switch (gravity) {
		case OIL_GRAVITY_TOP:
		case OIL_GRAVITY_TOP_LEFT:
		case OIL_GRAVITY_TOP_RIGHT:
			sy = 0.0;
			break;
		case OIL_GRAVITY_BOTTOM:
		case OIL_GRAVITY_BOTTOM_LEFT:
		case OIL_GRAVITY_BOTTOM_RIGHT:
			sy = img_height - sh;
			break;
		default:
			sy = (img_height - sh) / 2.0;
			break;
		}
		if (sy < 0.0) {
			sy = 0.0;
		}
	}

	*src_x = sx;
	*src_y = sy;
	*src_width = sw;
	*src_height = sh;
	return 0;
}
