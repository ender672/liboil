#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "resample.h"

/**
 * shared test helpers
 */
static long double cubic(long double b, long double c, long double x)
{
	if (x<1.0l) {
		return (
			(12.0l - 9.0l * b - 6.0l * c) * x*x*x +
			(-18.0l + 12.0l * b + 6.0l * c) * x*x +
			(6.0l -  2.0l * b)
		) / 6.0l;
	}
	if (x<2.0l) {
		return (
			(-b - 6.0l * c) * x*x*x +
			(6.0l * b + 30.0l * c) * x*x +
			(-12.0l * b - 48.0l * c) * x +
			(8.0l * b + 24.0l * c)
		) / 6.0l;
	}
	return 0.0;
}

static long double catrom(long double x)
{
	return cubic(0, 0.5l, x);
}

static void calc_coeffs(long double *coeffs, long double offset, long taps)
{
	long i;
	long double tap_offset, tap_mult, sum;

	tap_mult = (long double)taps / 4;
	sum = 0;
	for (i=0; i<taps; i++) {
		tap_offset = 1 - offset - taps / 2 + i;
		coeffs[i] = catrom(fabsl(tap_offset) / tap_mult) / tap_mult;
		sum += coeffs[i];
	}
	for (i=0; i<taps; i++) {
		coeffs[i] /= sum;
	}
}

unsigned char clamp(long double x)
{
	if (x < 0) {
		return 0;
	}

	/* This rounds to the nearest integer */
	x += 0.5l;
	if (x > 255) {
		return 255;
	}
	return x;
}

long double get_delta(long double expected, unsigned char actual)
{
	if (clamp(expected) != actual) {
		return fabsl(expected - actual);
	}
	return 0;
}

long double validate_sample(long taps, unsigned char *samples, long double *coeffs, unsigned char expected)
{
	long i;
	long double sum;

	sum = 0;

	for (i=0; i<taps; i++) {
		sum += coeffs[i] * samples[i];
	}

	return get_delta(sum, expected);
}

static void fill_rand(unsigned char *buf, long len)
{
	long i;
	for (i=0; i<len; i++) {
		buf[i] = rand();
	}
}

/**
 * calc_taps
 */
static uint64_t calc_taps_check(uint32_t dim_in, uint32_t dim_out)
{
	uint64_t tmp_i;
	tmp_i = (uint64_t)dim_in * 4 / dim_out;
	return tmp_i + (tmp_i%2);
}

static void test_calc_taps2(long dim_in, long dim_out)
{
	long check, res;
	res = calc_taps(dim_in, dim_out);
	check = calc_taps_check(dim_in, dim_out);
	assert(res == check);
}

static void test_calc_taps()
{
	/* make sure our math matches the reference */
	test_calc_taps2(10000, 1);
	test_calc_taps2(400, 200);
	test_calc_taps2(600, 200);
	test_calc_taps2(10000, 10);
	test_calc_taps2(10003, 17);
	test_calc_taps2(10000, 9999);

	/* Test uint32_t overflow to uint64_t */
	test_calc_taps2(0xFFFFFFFF, 1);

	/* zoom -- always 4 */
	assert(calc_taps(100, 100) == 4);
	assert(calc_taps(1, 1) == 4);
	assert(calc_taps(1, 10000) == 4);
	assert(calc_taps(1, 0xFFFFFFFF) == 4);
}

/**
 * padded_sl_len_offset()
 */

static void test_padded_sl_len_offset()
{
	size_t len, offset;
	len = padded_sl_len_offset(100, 100, 4, &offset);
	// (taps / 2 + 1) * cmp (taps should be 4)
	assert(offset == 12);
	// width * components + offset * 2
	assert(len == 100 * 4 + 24);
}

/**
 * split_map
 */
static long double map(long dim_in, long dim_out, long pos)
{
	return (pos + 0.5l) * (long double)dim_in / dim_out - 0.5l;
}

static long double split_map_check(long dim_in, long dim_out, long pos,
	long double *ty)
{
	long double smp;
	long smp_i;

	smp = map(dim_in, dim_out, pos);
	smp_i = floorl(smp);
	*ty = smp - smp_i;
	return smp_i;
}

static void test_split_map(long dim_in, long dim_out)
{
	long i, pos, pos_check;
	float ty;
	long double ty_check, delta;

	for (i=1; i<=dim_out; i++) {
		pos = split_map(dim_in, dim_out, i, &ty);
		pos_check = split_map_check(dim_in, dim_out, i, &ty_check);

		assert(pos == pos_check);

		delta = fabsl(ty - ty_check);

		/**
		 * The acceptable delta was chosen arbitrarily to make test pass
		 * with the current logic.
		 *
		 * Should it be tightened? What effect does it have when ty is
		 * off by this much?
		 */
		assert(delta < 0.0000001l);
	}
}

static void test_split_map_all()
{
	test_split_map(100, 100);
	test_split_map(1, 1);
	test_split_map(10000, 1);
	test_split_map(1, 10000);
	test_split_map(10000, 9);
	test_split_map(9, 10000);
	test_split_map(10000, 9999);
	test_split_map(9999, 10000);
	test_split_map(10000, 19);
	test_split_map(19, 10000);
}

/**
 * xscale
 */
static void validate_scanline(unsigned char *in, long width_in,
	unsigned char *out, long width_out, int cmp)
{
	long i, k, smp_i, smp_safe, taps;
	float ty;
	long double *coeffs, delta;
	unsigned char *samples;
	int j;

	taps = calc_taps(width_in, width_out);
	coeffs = malloc(taps * sizeof(long double));
	samples = malloc(taps * sizeof(unsigned char));

	for (i=0; i<1; i++) {
		for (j=0; j<cmp; j++) {
			smp_i = split_map(width_in, width_out, i, &ty);
			calc_coeffs(coeffs, ty, taps);
			for (k=0; k<taps; k++) {
				smp_safe = smp_i - taps / 2 + 1 + k;
				if (smp_safe < 0) {
					smp_safe = 0;
				} else if (smp_safe >= width_in) {
					smp_safe = width_in - 1;
				}
				samples[k] = in[smp_safe * cmp + j];
			}

			delta = validate_sample(taps, samples, coeffs, out[i * cmp + j]);
			assert(!delta);
		}
	}

	free(samples);
	free(coeffs);
}

static void test_xscale(long width_in, long width_out, int cmp)
{
	unsigned char *inbuf, *outbuf;

	inbuf = malloc(width_in * cmp);
	fill_rand(inbuf, width_in * cmp);
	outbuf = malloc(width_out * cmp);

	xscale(inbuf, width_in, outbuf, width_out, cmp);
	validate_scanline(inbuf, width_in, outbuf, width_out, cmp);

	free(outbuf);
	free(inbuf);
}

static void test_xscale_fmt(long width_in, long width_out)
{
	test_xscale(width_in, width_out, 1);
	test_xscale(width_in, width_out, 2);
	test_xscale(width_in, width_out, 3);
	test_xscale(width_in, width_out, 4);
}

static void test_xscale_all()
{
	test_xscale_fmt(10000, 19);
	test_xscale_fmt(10000, 10);
	test_xscale_fmt(19, 10000);
	test_xscale_fmt(10, 10);
	test_xscale_fmt(1, 1);
	test_xscale_fmt(10000, 1);
	test_xscale_fmt(1, 10000);
	test_xscale_fmt(10000, 9999);
	test_xscale_fmt(9999, 10000);
	test_xscale_fmt(13000, 1000);
}

/**
 * strip_scale
 */
void strip_scale_check(unsigned char **in, long taps, size_t len,
	unsigned char *out, long double ty)
{
	size_t i, j;
	long double *coeffs, delta;
	unsigned char *samples;

	coeffs = malloc(taps * sizeof(long double));
	samples = malloc(taps * sizeof(unsigned char));
	calc_coeffs(coeffs, ty, taps);

	for (i=0; i<len; i++) {
		for (j=0; j<taps; j++) {
			samples[j] = in[j][i];
		}

		delta = validate_sample(taps, samples, coeffs, out[i]);
		assert(!delta);
	}

	free(coeffs);
	free(samples);
}

static void test_strip_scale(long taps, long width, int cmp, float ty)
{
	unsigned char **scanlines, *out;
	long i;

	scanlines = malloc(taps * sizeof(unsigned char *));
	out = malloc(width * cmp);

	for (i=0; i<taps; i++) {
		scanlines[i] = malloc(width * cmp);
		fill_rand(scanlines[i], width * cmp);
	}

	strip_scale(scanlines, taps, (size_t)width * cmp, out, ty);
	strip_scale_check(scanlines, taps, (size_t)width * cmp, out, ty);

	for (i=0; i<taps; i++) {
		free(scanlines[i]);
	}
	free(out);
	free(scanlines);
}

static void test_strip_scale_all()
{
	test_strip_scale(4, 1000, 4, 0.2345);
	test_strip_scale(8, 1000, 4, 0.5);
	test_strip_scale(12, 1000, 1, 0.0);
	test_strip_scale(12, 1000, 3, 0.00005);
	test_strip_scale(12, 1000, 2, 0.99999);
}

/**
 * sl_rbuf
 */
static void test_sl_rbuf()
{
	uint8_t *p, *buf1, *buf2, *buf3, *buf4, **virt;
	struct sl_rbuf rb;
	sl_rbuf_init(&rb, 4, 1024);
	assert(rb.count == 0);

	buf1 = rb.buf;
	buf2 = rb.buf + 1024;
	buf3 = rb.buf + 1024 * 2;
	buf4 = rb.buf + 1024 * 3;

	p = sl_rbuf_next(&rb);
	assert(p == buf1);
	assert(rb.count == 1);

	virt = sl_rbuf_virt(&rb, 0);
	assert(virt[0] == buf1);
	assert(virt[1] == buf1);
	assert(virt[2] == buf1);
	assert(virt[3] == buf1);

	p = sl_rbuf_next(&rb);
	assert(p == buf2);
	assert(rb.count == 2);

	virt = sl_rbuf_virt(&rb, 1); // 0, 0, 0, 1
	assert(virt[0] == buf1);
	assert(virt[1] == buf1);
	assert(virt[2] == buf1);
	assert(virt[3] == buf2);

	p = sl_rbuf_next(&rb);
	assert(p == buf3);
	assert(rb.count == 3);

	p = sl_rbuf_next(&rb);
	assert(p == buf4);
	assert(rb.count == 4);

	p = sl_rbuf_next(&rb);
	assert(p == buf1);
	assert(rb.count == 5);

	virt = sl_rbuf_virt(&rb, 5); // 2, 3, 4, 5
	assert(virt[0] == buf3);
	assert(virt[1] == buf4);
	assert(virt[2] == buf1);
	assert(virt[3] == buf1);

	virt = sl_rbuf_virt(&rb, 3); // 0, 1, 2, 3 <= we don't have 0 anymore!
	assert(virt == 0);

	sl_rbuf_free(&rb);
}

/**
 * yscaler
 */
static void test_yscaler()
{
	struct yscaler ys;
	uint32_t i, j;
	uint8_t *tmp, *buf;

	buf = malloc(1024);
	yscaler_init(&ys, 1000, 30, 1024);
	j = 0;
	for (i=0; i<30; i++) {
		while((tmp = yscaler_next(&ys))) {
			j++;
			fill_rand(tmp, 1024);
		}
		yscaler_scale(&ys, buf, i);
	}
	assert(j == 1000);
	yscaler_free(&ys);
	free(buf);
}

/**
 * yscaler_prealloc_scale
 */
static void test_yscaler_prealloc_scale()
{
	uint32_t i;
	uint8_t *buf_in, *buf_out, **virt;

	buf_in = malloc(300 * 1024);
	fill_rand(buf_in, 300 * 1024);
	buf_out = malloc(12 * 1024);

	virt = malloc(sizeof(uint8_t *) * 300);
	for (i=0; i<300; i++) {
		virt[i] = buf_in + i * 1024;
	}

	for (i=0; i<12; i++) {
		yscaler_prealloc_scale(300, 12, virt, buf_out, i, 256, 4);
	}

	free(virt);
	free(buf_out);
	free(buf_in);
}

int main()
{
	test_calc_taps();
	test_split_map_all();
	test_xscale_all();
	test_padded_sl_len_offset();
	test_strip_scale_all();
	test_sl_rbuf();
	test_yscaler();
	test_yscaler_prealloc_scale();
	printf("All tests pass.\n");
	return 0;
}
