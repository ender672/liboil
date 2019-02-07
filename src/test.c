#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "oil_resample.c"

static long double linear_sample_to_srgb_reference(long double in_f)
{
	long double tmp;
	if (in_f <= 0.0L) {
		return 0.0L;
	}
	if (in_f >= 1.0L) {
		return 1.0L;
	}
	if (in_f <= 0.00313066844250063L) {
		return in_f * 12.92L;
	} else {
		tmp = powl(in_f, 1/2.4L);
		return 1.055L * tmp - 0.055L;
	}
}

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

static long double ref_catrom(long double x)
{
	return cubic(0, 0.5l, x);
}

static void ref_calc_coeffs(long double *coeffs, long double offset, long taps)
{
	long i;
	long double tap_offset, tap_mult, fudge;

	tap_mult = (long double)taps / 4;
	fudge = 1.0;
	for (i=0; i<taps; i++) {
		tap_offset = 1 - offset - taps / 2 + i;
		coeffs[i] = ref_catrom(fabsl(tap_offset) / tap_mult) / tap_mult;
		fudge -= coeffs[i];
	}
	coeffs[taps / 2] += fudge;
}

static void fill_rand(float *buf, long len)
{
	long i;
	for (i=0; i<len; i++) {
		buf[i] = (float)rand()/RAND_MAX;
	}
}

static void fill_rand8(uint8_t *buf, long len)
{
	long i;
	for (i=0; i<len; i++) {
		buf[i] = rand() % 256;
	}
}

/**
 * calc_taps
 */
static uint64_t calc_taps_check(uint32_t dim_in, uint32_t dim_out)
{
	uint64_t tmp_i;
	tmp_i = (uint64_t)dim_in * 4 / dim_out;
	return tmp_i - (tmp_i%2);
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

	/* zoom -- always 4 */
	assert(calc_taps(100, 100) == 4);
	assert(calc_taps(1, 1) == 4);
	assert(calc_taps(1, 10000) == 4);
}

/**
 * split_map
 */
static long double ref_map(long dim_in, long dim_out, long pos)
{
	return (pos + 0.5l) * (long double)dim_in / dim_out - 0.5l;
}

static long double split_map_check(long dim_in, long dim_out, long pos,
	long double *ty)
{
	long double smp;
	long smp_i;

	smp = ref_map(dim_in, dim_out, pos);
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

static void ref_sample_generic(long taps, long double *coeffs, long double *in, long double *out, uint8_t cmp)
{
	uint8_t i;
	long j;
	long double sum;

	for (i=0; i<cmp; i++) {
		sum = 0;
		for (j=0; j<taps; j++) {
			sum += coeffs[j] * (in[j * cmp + i]);
		}
		out[i] = sum;
	}
}

static void ref_sample_rgbx(long taps, long double *coeffs, long double *in, long double *out)
{
	long j;
	long double sums[3];

	sums[0] = sums[1] = sums[2] = 0;
	for (j=0; j<taps; j++) {
		sums[0] += coeffs[j] * (in[j * 4]);
		sums[1] += coeffs[j] * (in[j * 4 + 1]);
		sums[2] += coeffs[j] * (in[j * 4 + 2]);
	}
	out[0] = sums[0];
	out[1] = sums[1];
	out[2] = sums[2];
	out[3] = 0;
}

static void ref_set_sample(long taps, long double *coeffs, long double *in,
	long double *out, enum oil_colorspace cs)
{
	switch(cs) {
	case OIL_CS_RGBX:
		ref_sample_rgbx(taps, coeffs, in, out);
		break;
	case OIL_CS_G:
	case OIL_CS_GA:
	case OIL_CS_RGB:
	case OIL_CS_RGBA:
	case OIL_CS_CMYK:
		ref_sample_generic(taps, coeffs, in, out, OIL_CMP(cs));
		break;
	}
}

static void validate_scanline8(uint8_t *oil, long double *ref, size_t width, int cmp)
{
	int i, j, ref_i, pos;
	long double error, ref_f;
	for (i=0; i<width; i++) {
		for (j=0; j<cmp; j++) {
			pos = i * cmp + j;
			ref_f = ref[pos] * 255.0L;
			ref_i = lroundl(ref_f);
			error = fabsl(oil[pos] - ref_f);
			if (error > 0.5001L) {
				fprintf(stderr, "[%d:%d] expected: %d, got %d (%.9Lf)\n", i, j, ref_i, oil[pos], ref_f);
			}
		}
	}
}

/**
 * strip_scale
 */

static long double clamp_f(long double in)
{
	if (in <= 0.0L) {
		return 0.0L;
	}
	if (in >= 1.0L) {
		return 1.0L;
	}
	return in;
}

static void postprocess(long double *in, long double *out, enum oil_colorspace cs)
{
	long double alpha;
	switch (cs) {
	case OIL_CS_G:
		out[0] = clamp_f(in[0]);
		break;
	case OIL_CS_GA:
		alpha = clamp_f(in[1]);
		if (alpha != 0.0L) {
			in[0] /= alpha;
		}
		out[0] = clamp_f(in[0]);
		out[1] = alpha;
		break;
	case OIL_CS_RGB:
		out[0] = linear_sample_to_srgb_reference(in[0]);
		out[1] = linear_sample_to_srgb_reference(in[1]);
		out[2] = linear_sample_to_srgb_reference(in[2]);
		break;
	case OIL_CS_RGBX:
		out[0] = linear_sample_to_srgb_reference(in[0]);
		out[1] = linear_sample_to_srgb_reference(in[1]);
		out[2] = linear_sample_to_srgb_reference(in[2]);
		out[3] = 0;
		break;
	case OIL_CS_RGBA:
		alpha = clamp_f(in[3]);
		if (alpha != 0.0L) {
			in[0] /= alpha;
			in[1] /= alpha;
			in[2] /= alpha;
		}
		out[0] = linear_sample_to_srgb_reference(in[0]);
		out[1] = linear_sample_to_srgb_reference(in[1]);
		out[2] = linear_sample_to_srgb_reference(in[2]);
		out[3] = alpha;
		break;
	case OIL_CS_CMYK:
		out[0] = clamp_f(in[0]);
		out[1] = clamp_f(in[1]);
		out[2] = clamp_f(in[2]);
		out[3] = clamp_f(in[3]);
		break;
	}
}

static void ref_strip_scale(float **in, long taps, long width, long double *out,
	long double ty, enum oil_colorspace cs)
{
	long double *coeffs, *samples, *tmp;
	int i, j, k, cmp;

	cmp = OIL_CMP(cs);
	coeffs = malloc(taps * sizeof(long double));
	samples = malloc(taps * cmp * sizeof(long double));
	tmp = malloc(cmp * sizeof(long double));
	ref_calc_coeffs(coeffs, ty, taps);

	for (i=0; i<width; i++) {
		for (k=0; k<cmp; k++) {
			for (j=0; j<taps; j++) {
				samples[j * cmp + k] = in[j][i * cmp + k];
			}
		}
		ref_set_sample(taps, coeffs, samples, tmp, cs);
		postprocess(tmp, out, cs);
		out += OIL_CMP(cs);
	}

	free(tmp);
	free(coeffs);
	free(samples);
}

static void test_strip_scale(long taps, long width, float ty,
	enum oil_colorspace cs)
{
	long double *out_ref;
	unsigned char *out_oil;
	long i, cmp;
	float *coeffs, **scanlines;

	cmp = OIL_CMP(cs);
	scanlines = malloc(taps * sizeof(float *));
	out_oil = malloc(width * cmp * sizeof(unsigned char));
	out_ref = malloc(width * cmp * sizeof(long double));

	for (i=0; i<taps; i++) {
		scanlines[i] = malloc(width * cmp * sizeof(float));
		fill_rand(scanlines[i], width * cmp);
	}

	coeffs = malloc(taps * sizeof(float));
	strip_scale(scanlines, taps, width * cmp, out_oil, coeffs, ty, cs);
	ref_strip_scale(scanlines, taps, width, out_ref, ty, cs);
	validate_scanline8(out_oil, out_ref, width, cmp);

	for (i=0; i<taps; i++) {
		free(scanlines[i]);
	}
	free(coeffs);
	free(out_ref);
	free(out_oil);
	free(scanlines);
}

static void test_strip_scale_each_cs(long taps, long width, float ty)
{
	test_strip_scale(taps, width, ty, OIL_CS_G);
	test_strip_scale(taps, width, ty, OIL_CS_GA);
	test_strip_scale(taps, width, ty, OIL_CS_RGB);
	test_strip_scale(taps, width, ty, OIL_CS_RGBX);
	test_strip_scale(taps, width, ty, OIL_CS_RGBA);
	test_strip_scale(taps, width, ty, OIL_CS_CMYK);
}

static void test_strip_scale_all()
{
	test_strip_scale_each_cs(49, 1000,  0.2345f);
	test_strip_scale_each_cs(8, 1000,  0.5);
	test_strip_scale_each_cs(12, 1000, 0.0);
	test_strip_scale_each_cs(12, 1000, 0.00005);
	test_strip_scale_each_cs(12, 1000, 0.99999);
}

static void test_yscale_transparent()
{
	unsigned char *out_oil;
	long i, j, taps, width, cmp;
	float ty, *coeffs, **scanlines;
	long double *out_ref;
	enum oil_colorspace cs;

	cs = OIL_CS_RGBA;
	taps = 8;
	width = 100;
	ty = 0.5;

	cmp = OIL_CMP(cs);
	scanlines = malloc(taps * sizeof(float *));
	out_oil = malloc(width * cmp * sizeof(uint8_t));
	out_ref = malloc(width * cmp * sizeof(long double));

	for (i=0; i<taps; i++) {
		scanlines[i] = malloc(width * cmp * sizeof(float));
		fill_rand(scanlines[i], width * cmp);
		for (j=0; j<width; j++) {
			scanlines[i][j*3+3] = 0.0f;
		}
	}

	coeffs = malloc(taps * sizeof(float));
	strip_scale(scanlines, taps, width * cmp, out_oil, coeffs, ty, cs);
	ref_strip_scale(scanlines, taps, width, out_ref, ty, cs);
	validate_scanline8(out_oil, out_ref, width, cmp);

	for (i=0; i<taps; i++) {
		free(scanlines[i]);
	}
	free(coeffs);
	free(out_oil);
	free(out_ref);
	free(scanlines);
}

/**
 * yscaler
 */

static long double srgb_sample_to_linear_reference(int in)
{
	long double in_f, tmp;
	in_f = in / 255.0L;
	if (in_f <= 0.0404482362771082L) {
		return in_f / 12.92L;
	} else {
		tmp = ((in_f + 0.055L)/1.055L);
		return powl(tmp, 2.4L);
	}
}

static void test_srgb_to_linear()
{
	long double reference, actual, diff;
	int i;

	for (i=0; i<=255; i++) {
		reference = srgb_sample_to_linear_reference(i);
		actual = s2l_map_f[i];
		diff = fabsl(actual - reference);
		assert(diff < 0.0000001L);
	}
}

static void test_linear_to_srgb()
{
	int actual, ref_i, input;
	long double reference, ref_f, i_f;

	for (input=0; input<65536; input++) {
		i_f = input / 65535.0L;
		reference = linear_sample_to_srgb_reference(i_f);
		actual = linear_sample_to_srgb(i_f);
		ref_f = reference * 255.0L;
		ref_i = lroundl(ref_f);
		assert(ref_i == actual);
	}
}

static void test_roundtrip()
{
	int i;
	float linear, back_to_srgb;
	for (i=0; i<=255; i++) {
		linear = s2l_map_f[i];
		back_to_srgb = linear_sample_to_srgb(linear);
		assert(back_to_srgb == i);
	}
}

static void test_colorspace_helpers()
{
	test_srgb_to_linear();
	test_linear_to_srgb();
	test_roundtrip();
}

static void test_xscale_limit(long width_in, long width_out,
	enum oil_colorspace cs)
{
	unsigned char *buf_in;
	unsigned short *buf_out;
	size_t len_in, len_out;
	struct oil_scale os;

	len_in = (size_t)width_in * OIL_CMP(cs) * sizeof(unsigned char);
	buf_in = malloc(len_in);
	assert(buf_in);

	len_out = (size_t)width_out * OIL_CMP(cs) * sizeof(unsigned short);
	buf_out = malloc(len_out);
	assert(buf_out);

	fill_rand8(buf_in, len_in);
	oil_scale_init(&os, 1, 1, width_in, width_out, cs);
	oil_scale_in(&os, buf_in);
	oil_scale_free(&os);

	free(buf_in);
	free(buf_out);
}

static void test_xscale_limits()
{
	test_xscale_limit(1000000, 1, OIL_CS_RGBX);
	test_xscale_limit(1000000, 1, OIL_CS_RGBA);
	test_xscale_limit(1000000, 1, OIL_CS_CMYK);
}

static void test_yscale_limit(long height_in, long width,
	long height_out, enum oil_colorspace cs)
{
	long i, k;
	float *tmp;
	uint8_t *buf_out;
	size_t len;
	struct oil_scale ys;
	int res;

	len = (size_t)width * OIL_CMP(cs) * sizeof(uint8_t);
	buf_out = malloc(len);
	assert(buf_out);

	res = oil_scale_init(&ys, height_in, height_out, width, width, cs);
	assert(res==0);

	for (i=0; i<height_out; i++) {
		for (k=oil_scale_slots(&ys); k>0; k--) {
			tmp = ys.rb + (ys.in_pos % ys.taps) * ys.sl_len;
			ys.in_pos++;
			fill_rand(tmp, len);
		}
		oil_scale_out(&ys, buf_out);
	}

	free(buf_out);
	oil_scale_free(&ys);
}

static void test_yscale_limits()
{
	test_yscale_limit(1000000, 10, 1, OIL_CS_RGBX);
}

static void test_xscale_down_coeffs(int in_width, int out_width)
{
	float *coeff_buf, *cur_coeff, *sample_coeffs[4], *tmp;
	long double tot;
	int i, j, k, *border_buf, taps, sample_pos[4];

	taps = calc_taps(in_width, out_width);

	coeff_buf = cur_coeff = malloc(4 * in_width * sizeof(float));
	border_buf = malloc(out_width * sizeof(int));
	for (i=0; i<4; i++) {
		sample_pos[i] = 0;
		sample_coeffs[i] = malloc(2 * taps * sizeof(float));
	}

	xscale_calc_coeffs(in_width, out_width, coeff_buf, border_buf);

	for (i=0; i<out_width; i++) {
		for (j=border_buf[i]; j>0; j--) {
			for (k=0; k<4; k++) {
				sample_coeffs[k][sample_pos[k]] = cur_coeff[k];
				sample_pos[k] += 1;
			}
			cur_coeff += 4;
		}

		tot = 0.0;
		for (k=0; k<sample_pos[0]; k++) {
			tot += sample_coeffs[0][k];
		}

		if (fabsl(1.0F - tot) > 0.00001F) {
			fprintf(stderr, "[%9Lf] ", tot);
			fprintf(stderr, "%4d: ", i);
			for (k=0; k<sample_pos[0]; k++) {
				fprintf(stderr, "%6.3f, ", sample_coeffs[0][k]);
			}
			fprintf(stderr, "\n");
		}

		tmp = sample_coeffs[0];
		sample_coeffs[0] = sample_coeffs[1];
		sample_coeffs[1] = sample_coeffs[2];
		sample_coeffs[2] = sample_coeffs[3];
		sample_coeffs[3] = tmp;

		sample_pos[0] = sample_pos[1];
		sample_pos[1] = sample_pos[2];
		sample_pos[2] = sample_pos[3];
		sample_pos[3] = 0;
	}
	free(coeff_buf);
	free(border_buf);
	for (i=0; i<4; i++) {
		free(sample_coeffs[i]);
	}
}

static void test_xscale_down_borders(int in_width, int out_width)
{
	float *coeff_buf;
	int i, *border_buf, tot;

	tot = 0;

	coeff_buf = malloc(4 * in_width * sizeof(float));
	border_buf = malloc(out_width * sizeof(int));
	xscale_calc_coeffs(in_width, out_width, coeff_buf, border_buf);

	for (i=0; i<out_width; i++) {
		tot += border_buf[i];
	}
	assert(tot == in_width);

	free(coeff_buf);
	free(border_buf);
}

static void test_xscale_coeffs_dimensions(int in_width, int out_width)
{
	test_xscale_down_coeffs(in_width, out_width);
	test_xscale_down_borders(in_width, out_width);
}

static void test_xscale_coeffs()
{
	test_xscale_coeffs_dimensions(100, 1);
	test_xscale_coeffs_dimensions(100, 99);
	test_xscale_coeffs_dimensions(2398423, 23423);
	test_xscale_coeffs_dimensions(1000000, 303);
	test_xscale_coeffs_dimensions(100, 70);
	test_xscale_coeffs_dimensions(1000000, 999999);
}

int main()
{
	int t = 1531289551;
	//int t = time(NULL);
	printf("seed: %d\n", t);
	srand(t);
	oil_global_init();
	test_calc_taps();
	test_split_map_all();
	test_strip_scale_all();
	test_colorspace_helpers();
	test_xscale_limits();
	test_yscale_limits();
	test_yscale_transparent();
	test_xscale_coeffs();
	printf("All tests pass.\n");
	return 0;
}
