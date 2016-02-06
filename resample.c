/**
 * Copyright (c) 2014-2016 Timothy Elliott
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

#include "resample.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * Bicubic interpolation. 2 base taps on either side.
 */
#define TAPS 4

/**
 * 64-bit type that uses 1 bit for signedness, 33 bits for the integer, and 30
 * bits for the fraction.
 *
 * 0-29: fraction, 30-62: integer, 63: sign.
 *
 * Useful for storing the product of a fix1_30 type and an unsigned char.
 */
typedef int64_t fix33_30;

/**
 * We add this to a fix33_30 value in order to bump up rounding errors.
 *
 * The best possible value was determined by comparing to a reference
 * implementation and comparing values for the minimal number of errors.
 */
#define TOPOFF 8192

/**
 * Signed type that uses 1 bit for signedness, 1 bit for the integer, and 30
 * bits for the fraction.
 *
 * 0-29: fraction, 30: integer, 31: sign.
 *
 * Useful for storing coefficients.
 */
typedef int32_t fix1_30;
#define ONE_FIX1_30 (1<<30)

/**
 * Calculate the greatest common denominator between a and b.
 */
static uint32_t gcd(uint32_t a, uint32_t b)
{
	uint32_t c;
	while (a != 0) {
		c = a;
		a = b%a;
		b = c;
	}
	return b;
}

/**
 * Round and clamp a fix33_30 value between 0 and 255. Returns an unsigned char.
 */
static uint8_t clamp(fix33_30 x)
{
	if (x < 0) {
		return 0;
	}

	/* add 0.5 and bump up rounding errors before truncating */
	x += (1<<29) + TOPOFF;

	/* This is safe because we have the < 0 check above and a sample can't
	 * end up with a value over 512 */
	if (x & (1l<<38)) {
		return 255;
	}

	return x >> 30;
}

/**
 * Map from a discreet dest coordinate to a continuous source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
static double map(uint32_t pos, double scale)
{
	return (pos + 0.5) / scale - 0.5;
}

/**
 * Given input and output dimensions and an output position, return the
 * corresponding input position and put the sub-pixel remainder in rest.
 */
int32_t split_map(uint32_t dim_in, uint32_t dim_out, uint32_t pos, float *rest)
{
	double scale, smp;
	int32_t smp_i;

	scale = dim_out / (double)dim_in;
	smp = map(pos, scale);
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
uint64_t calc_taps(uint32_t dim_in, uint32_t dim_out)
{
	uint64_t tmp;
	if (dim_out >= dim_in) {
		return TAPS;
	}
	tmp = ((uint64_t)dim_in * TAPS) / dim_out;
	/* Round up to the nearest even integer */
	tmp += tmp&1;
	return tmp;
}

/**
 * Helper macros to extract byte components from an int32_t holding rgba.
 */
#define rgba_r(x) ((x) & 0x000000FF)
#define rgba_g(x) (((x) & 0x0000FF00) >> 8)
#define rgba_b(x) (((x) & 0x00FF0000) >> 16)
#define rgba_a(x) (((x) & 0xFF000000) >> 24)

/**
 * Convert rgba values in fix33_30 types to a uint32_t.
 */
static uint32_t fix33_30_to_rgba(fix33_30 r, fix33_30 g, fix33_30 b, fix33_30 a)
{
	return clamp(r) + ((uint32_t)clamp(g) << 8) +
		((uint32_t)clamp(b) << 16) + ((uint32_t)clamp(a) << 24);
}

/**
 * Catmull-Rom interpolator.
 */
static float catrom(float x)
{
	if (x<1) {
		return (3*x*x*x - 5*x*x + 2) / 2;
	}
	return (-1*x*x*x + 5*x*x - 8*x + 4) / 2;
}

/**
 * Convert a single-precision float to a fix1_30 fixed point int. x must be
 * between 0 and 1.
 */
static fix1_30 f_to_fix1_30(float x)
{
	return x * ONE_FIX1_30;
}

/**
 * Given an offset tx, calculate TAPS * tap_mult coefficients.
 *
 * The coefficients are stored as fix1_30 fixed point ints in coeffs.
 */
static void calc_coeffs(fix1_30 *coeffs, float tx, uint32_t taps)
{
	uint32_t i;
	float tmp, total, tap_mult;
	fix1_30 tmp_fixed;

	total = 0;

	tap_mult = (float)taps / TAPS;
	tx = 1 - tx - taps / 2;

	for (i=0; i<taps; i++) {
		tmp = catrom(fabsf(tx) / tap_mult);
		tmp_fixed = f_to_fix1_30(tmp);
		coeffs[i] = tmp_fixed;
		total += tmp;
		tx += 1;
	}

	for (i=0; i<taps; i++) {
		coeffs[i] /= total;
	}
}

int strip_scale(uint8_t **in, uint32_t strip_height, size_t len, uint8_t *out,
	float ty)
{
	fix1_30 *coeffs;
	uint32_t i, j;
	fix33_30 coeff, total;

	coeffs = malloc(strip_height * sizeof(fix1_30));
	if (!coeffs) {
		return -2; // unable to allocate
	}
	calc_coeffs(coeffs, ty, strip_height);

	for (i=0; i<len; i++) {
		total = 0;
		for (j=0; j<strip_height; j++) {
			coeff = coeffs[j];
			total += coeff * in[j][i];
		}
		out[i] = clamp(total);
	}

	free(coeffs);
	return 0;
}

/* Bicubic x scaler */

static void sample_generic(uint8_t cmp, uint32_t taps, fix1_30 *coeffs,
	uint8_t *in, uint8_t *out)
{
	uint8_t i;
	uint32_t j;
	fix33_30 total, coeff;

	for (i=0; i<cmp; i++) {
		total = 0;
		for (j=0; j<taps; j++){
			coeff = coeffs[j];
			total += coeff * in[j * cmp + i];
		}
		out[i] = clamp(total);
	}
}

static uint32_t sample_rgba(uint32_t taps, fix1_30 *coeffs, uint32_t *in)
{
	uint32_t i, sample;
	fix33_30 r, g, b, a, coeff;

	r = g = b = a = 0;
	for (i=0; i<taps; i++) {
		coeff = coeffs[i];
		sample = in[i];
		r += coeff * rgba_r(sample);
		g += coeff * rgba_g(sample);
		b += coeff * rgba_b(sample);
		a += coeff * rgba_a(sample);
	}
	return fix33_30_to_rgba(r, g, b, a);
}

static void xscale_set_sample(uint32_t taps, fix1_30 *coeffs, uint8_t *in,
	uint8_t *out, uint8_t cmp)
{
	if (cmp == 4) {
		*(uint32_t *)out = sample_rgba(taps, coeffs, (uint32_t *)in);
	} else {
		sample_generic(cmp, taps, coeffs, in, out);
	}
}

/* padded scanline */

void padded_sl_extend_edges(uint8_t *buf, uint32_t width, size_t pad_len,
	uint8_t cmp)
{
	uint8_t *pad_right;
	size_t i;
	pad_right = buf + pad_len + (size_t)width * cmp;
	for (i=0; i<pad_len; i++) {
		buf[i] = (buf + pad_len)[i % cmp];
		pad_right[i] = (pad_right - cmp)[i % cmp];
	}
}

size_t padded_sl_len_offset(uint32_t in_width, uint32_t out_width,
	uint8_t cmp, size_t *offset)
{
	uint64_t taps;
	taps = calc_taps(in_width, out_width);
	*offset = (taps / 2 + 1) * cmp;
	return (size_t)in_width * cmp + *offset * 2;
}

int xscale_padded(uint8_t *in, uint32_t in_width, uint8_t *out,
	uint32_t out_width, uint8_t cmp)
{
	float tx;
	fix1_30 *coeffs;
	uint32_t i, j, in_chunk, out_chunk, scale_gcd;
	uint64_t taps;
	int32_t xsmp_i;
	uint8_t *out_pos, *tmp;

	if (!in_width || !out_width || !cmp) {
		return -1; // bad input parameter
	}

	taps = calc_taps(in_width, out_width);
	coeffs = malloc(taps * sizeof(fix1_30));
	if (!coeffs) {
		return -2; // unable to allocate space for coefficients
	}

	scale_gcd = gcd(in_width, out_width);
	in_chunk = in_width / scale_gcd;
	out_chunk = out_width / scale_gcd;

	for (i=0; i<out_chunk; i++) {
		xsmp_i = split_map(in_width, out_width, i, &tx);
		calc_coeffs(coeffs, tx, taps);

		xsmp_i += 1 - taps / 2;
		out_pos = out + i * cmp;
		for (j=0; j<scale_gcd; j++) {
			tmp = in + xsmp_i * cmp;
			xscale_set_sample(taps, coeffs, tmp, out_pos, cmp);
			out_pos += out_chunk * cmp;
			xsmp_i += in_chunk;
		}
	}

	free(coeffs);
	return 0;
}

int xscale(uint8_t *in, uint32_t in_width, uint8_t *out, uint32_t out_width,
	uint8_t cmp)
{
	int ret;
	uint8_t *psl_buf, *psl_pos0;
	size_t psl_len, psl_offset;

	if (!in_width || !out_width || !cmp) {
		return -1; // bad input parameter
	}

	psl_len = padded_sl_len_offset(in_width, out_width, cmp, &psl_offset);
	psl_buf = malloc(psl_len);
	if (!psl_buf) {
		return -2; // unable to allocate scanline
	}
	psl_pos0 = psl_buf + psl_offset;
	memcpy(psl_pos0, in, (size_t)in_width * cmp);
	padded_sl_extend_edges(psl_buf, in_width, psl_offset, cmp);
	ret = xscale_padded(psl_pos0, in_width, out, out_width, cmp);
	free(psl_buf);
	return ret;
}

int sl_rbuf_init(struct sl_rbuf *rb, uint32_t height, size_t sl_len)
{
	rb->height = height;
	rb->count = 0;
	rb->length = sl_len;
	rb->buf = malloc(sl_len * height);
	if (!rb->buf) {
		return -2;
	}
	rb->virt = malloc(sizeof(uint8_t *) * height);
	if (!rb->virt) {
		free(rb->buf);
		return -2;
	}
	return 0;
}

void sl_rbuf_free(struct sl_rbuf *rb)
{
	free(rb->buf);
	free(rb->virt);
}

uint8_t *sl_rbuf_next(struct sl_rbuf *rb)
{
	return rb->buf + (rb->count++ % rb->height) * rb->length;
}

uint8_t **sl_rbuf_virt(struct sl_rbuf *rb, uint32_t last_target)
{
	uint32_t i, safe, height, last_idx;
	height = rb->height;
	last_idx = rb->count - 1;

	// Make sure we have the 1st scanline if extending upwards
	if (last_target < last_idx && last_idx > height - 1) {
		return 0;
	}

	for (i=0; i<height; i++) {
		safe = last_target < i ? 0 : last_target - i;
		safe = safe > last_idx ? last_idx : safe;
		rb->virt[height - i - 1] = rb->buf + (safe % height) * rb->length;
	}
	return rb->virt;
}

static void yscaler_map_pos(struct yscaler *ys, uint32_t pos)
{
	long target;
	target = split_map(ys->in_height, ys->out_height, pos, &ys->ty);
	ys->target = target + ys->rb.height / 2;
}

int yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	size_t scanline_len)
{
	int ret;
	uint32_t taps;
	taps = calc_taps(in_height, out_height);
	ys->in_height = in_height;
	ys->out_height = out_height;
	ret = sl_rbuf_init(&ys->rb, taps, scanline_len);
	yscaler_map_pos(ys, 0);
	return ret;
}

void yscaler_free(struct yscaler *ys)
{
	sl_rbuf_free(&ys->rb);
}

unsigned char *yscaler_next(struct yscaler *ys)
{
	if (ys->rb.count == ys->in_height || ys->rb.count > ys->target) {
		return 0;
	}
	return sl_rbuf_next(&ys->rb);
}

int yscaler_scale(struct yscaler *ys, uint8_t *out, uint32_t pos)
{
	int ret;
	uint8_t **virt;
	virt = sl_rbuf_virt(&ys->rb, ys->target);
	ret = strip_scale(virt, ys->rb.height, ys->rb.length, out, ys->ty);
	yscaler_map_pos(ys, pos + 1);
	return ret;
}

int yscaler_prealloc_scale(uint32_t in_height, uint32_t out_height,
	uint8_t **in, uint8_t *out, uint32_t pos, uint32_t width, uint8_t cmp)
{
	uint32_t i, taps;
	int32_t smp_i, strip_pos;
	uint8_t **virt;
	float ty;
	int ret;

	taps = calc_taps(in_height, out_height);
	virt = malloc(taps * sizeof(uint8_t *));
	if (!virt) {
		return -2;
	}
	smp_i = split_map(in_height, out_height, pos, &ty);
	strip_pos = smp_i + 1 - taps / 2;

	for (i=0; i<taps; i++) {
		if (strip_pos < 0) {
			virt[i] = in[0];
		} else if (strip_pos > in_height - 1) {
			virt[i] = in[in_height - 1];
		} else {
			virt[i] = in[strip_pos];
		}
		strip_pos++;
	}

	ret = strip_scale(virt, taps, (size_t)width * cmp, out, ty);
	free(virt);
	return ret;
}
