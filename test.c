#include "assert.h"
#include "resample.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"

/* Precise calculation of reverse map. */
static long double map(long dim_in, long dim_out, long pos)
{
	return (pos + 0.5l) * (long double)dim_in / dim_out - 0.5l;
}

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

static void calc_coeffs(long double *coeffs, long double offset, long tap_mult)
{
	long i, taps;
	long double tap_offset;

	taps = tap_mult * 4;
	for (i=0; i<taps; i++) {
		tap_offset = 1 - offset - taps / 2 + i;
		coeffs[i] = catrom(fabsl(tap_offset) / tap_mult) / tap_mult;
	}
}

unsigned char clamp(long double x)
{
	if (x < 0) {
		return 0;
	}
	x += 0.000000000001l;
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

void strip_scale_check(unsigned char **in, long taps, long width,
	unsigned char *out, int cmp, int opts, long double ty)
{
	long i, j, len, tap_mult, fails;
	long double *coeffs, exp_val, delta, max_delta;

	fails = 0;
	max_delta = 0;

	len = width * cmp;
	tap_mult = taps / 4;
	coeffs = malloc(taps * sizeof(long double));
	calc_coeffs(coeffs, ty, tap_mult);

	for (i=0; i<len; i++) {
		if (cmp == 4 && (opts & OIL_FILLER) && i%4 == 3) {
			continue;
		}
		exp_val = 0;
		for (j=0; j<taps; j++) {
			exp_val += coeffs[j] * in[j][i];
		}

		delta = get_delta(exp_val, out[i]);
		if (delta) {
			fails++;
			if (delta > max_delta) {
				max_delta = delta;
			}
		}
	}

	free(coeffs);
	if (fails > 0) {
		printf("(y) len: %ld, cmp: %d, ty: %.20Lf, fails: %ld, max_delta: %.20Lf\n", len, cmp, ty, fails, max_delta);
	}
}

static void validate_scanline(unsigned char *in, long width_in,
	unsigned char *out, long width_out, int cmp, int opts)
{
	long i, k, smp_i, smp_safe, tap_mult, fails, taps;
	long double *coeffs, smp, exp_val, delta, max_delta;
	int j;

	fails = 0;
	max_delta = 0;

	taps = calc_taps(width_in, width_out);
	tap_mult = taps / 4;
	coeffs = malloc(taps * sizeof(long double));

	for (i=0; i<width_out; i++) {
		for (j=0; j<cmp; j++) {
			if (cmp == 4 && (opts & OIL_FILLER) && j == 3) {
				continue;
			}
			smp = map(width_in, width_out, i);
			smp_i = floorl(smp);
			calc_coeffs(coeffs, smp - smp_i, tap_mult);
			exp_val = 0;
			for (k=0; k<taps; k++) {
				smp_safe = smp_i - taps / 2 + 1 + k;
				if (smp_safe < 0) {
					smp_safe = 0;
				} else if (smp_safe >= width_in) {
					smp_safe = width_in - 1;
				}
				exp_val += coeffs[k] * in[smp_safe * cmp + j];
			}

			delta = get_delta(exp_val, out[i * cmp + j]);
			if (delta) {
				fails++;
				if (delta > max_delta) {
					max_delta = delta;
				}
			}
		}
	}

	free(coeffs);
	if (fails > 0) {
		printf("(x) in: %ld, out: %ld, cmp: %d, fails: %ld, max_delta: %.20Lf\n", width_in, width_out, cmp, fails, max_delta);
	}
}

static void fill_rand(unsigned char *buf, long len)
{
	long i;
	for (i=0; i<len; i++) {
		buf[i] = rand();
	}	
}

static void test_xscale(long width_in, long width_out, int cmp, int opts)
{
	unsigned char *inbuf, *outbuf;

	inbuf = malloc(width_in * cmp);
	fill_rand(inbuf, width_in * cmp);
	outbuf = malloc(width_out * cmp);

	xscale(inbuf, width_in, outbuf, width_out, cmp, opts);
	validate_scanline(inbuf, width_in, outbuf, width_out, cmp, opts);

	free(outbuf);
	free(inbuf);
}

static void test_xscale_fmt(long width_in, long width_out)
{
	test_xscale(width_in, width_out, 1, 0);
	test_xscale(width_in, width_out, 2, 0);
	test_xscale(width_in, width_out, 3, 0);
	test_xscale(width_in, width_out, 4, 0);
	test_xscale(width_in, width_out, 4, OIL_FILLER);
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
}

static void test_calc_taps()
{
	int res;

	res = calc_taps(100, 100);
	assert(res == 4);

	res = calc_taps(1, 1);
	assert(res == 4);

	res = calc_taps(1, 10000);
	assert(res == 4);

	res = calc_taps(400, 200);
	assert(res == 8);

	res = calc_taps(600, 200);
	assert(res == 12);

	res = calc_taps(10000, 10);
	assert(res == 4000);
}

/* Precise calculation of bottom of scanline */
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
		if (delta > 0.0000001l) {
			printf("delta: %.30Lf\n", delta);
		}
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

static void test_yscale(long taps, long width, int cmp, int opts, float ty)
{
	unsigned char **scanlines, *out;
	long i;

	scanlines = malloc(taps * sizeof(unsigned char *));
	out = malloc(width * cmp);

	for (i=0; i<taps; i++) {
		scanlines[i] = malloc(width * cmp);
		fill_rand(scanlines[i], width * cmp);
	}

	strip_scale((void **)scanlines, taps, width, (void *)out, ty, cmp, opts);
	strip_scale_check(scanlines, taps, width, out, cmp, opts, ty);

	for (i=0; i<taps; i++) {
		free(scanlines[i]);
	}
	free(out);
	free(scanlines);
}

static void test_yscale_all()
{
	test_yscale(4, 1000, 4, 0, 0.2345);
	test_yscale(8, 1000, 4, OIL_FILLER, 0.5);
	test_yscale(12, 1000, 1, 0, 0.0);
	test_yscale(12, 1000, 3, 0, 0.00005);
	test_yscale(12, 1000, 2, 0, 0.99999);
}

int main()
{
	test_xscale_all();
	test_calc_taps();
	test_split_map_all();
	test_yscale_all();
	return 0;
}