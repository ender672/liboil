#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "resample.c"

static uint8_t linear_sample_to_srgb_reference(uint16_t in)
{
	long double in_f, result, tmp;
	in_f = in / 65535.0L;
	if (in_f <= 0.00313066844250063L) {
		result = in_f * 12.92L;
	} else {
		tmp = pow(in_f, 1/2.4L);
		result = 1.055L * tmp - 0.055L;
	}
	return round(result * 255);
}

static uint8_t clamp_16_to_8(uint16_t in)
{
	return round((in / 65535.0) * 255);
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

static void fill_rand(uint16_t *buf, long len)
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
	assert(len == (100 * 4 + 24) * 2);
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

static uint16_t fto16(long double x)
{
	x = fmin(1, x);
	x = fmax(0, x);
	return round(x * 65535);
}

static void ref_sample_generic(long taps, long double *coeffs, uint16_t *in, uint16_t *out, uint8_t cmp)
{
	uint8_t i;
	long j;
	long double sum;

	for (i=0; i<cmp; i++) {
		sum = 0;
		for (j=0; j<taps; j++) {
			sum += coeffs[j] * (in[j * cmp + i] / 65535.0L);
		}
		out[i] = fto16(sum);
	}
}

static void ref_sample_rgbx(long taps, long double *coeffs, uint16_t *in, uint16_t *out)
{
	long j;
	long double sums[3];

	sums[0] = sums[1] = sums[2] = 0;
	for (j=0; j<taps; j++) {
		sums[0] += coeffs[j] * (in[j * 4] / 65535.0L);
		sums[1] += coeffs[j] * (in[j * 4 + 1] / 65535.0L);
		sums[2] += coeffs[j] * (in[j * 4 + 2] / 65535.0L);
	}
	out[0] = fto16(sums[0]);
	out[1] = fto16(sums[1]);
	out[2] = fto16(sums[2]);
	out[3] = 0;
}

static void ref_set_sample(long taps, long double *coeffs, uint16_t *in,
	uint16_t *out, enum oil_colorspace cs)
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
		ref_sample_generic(taps, coeffs, in, out, CS_TO_CMP(cs));
		break;
	}
}

/**
 * xscale
 */
static void ref_xscale(uint16_t *in, long width_in, uint16_t *out,
	long width_out, enum oil_colorspace cs)
{
	long i, k, smp_i, smp_safe, taps;
	float ty;
	long double *coeffs;
	uint8_t cmp;
	uint16_t *samples;
	int j;

	taps = calc_taps(width_in, width_out);
	coeffs = malloc(taps * sizeof(long double));
	cmp = CS_TO_CMP(cs);
	samples = malloc(taps * sizeof(uint16_t) * cmp);

	for (i=0; i<width_out; i++) {
		smp_i = split_map(width_in, width_out, i, &ty);
		ref_calc_coeffs(coeffs, ty, taps);
		for (k=0; k<taps; k++) {
			smp_safe = smp_i - taps / 2 + 1 + k;
			if (smp_safe < 0) {
				smp_safe = 0;
			} else if (smp_safe >= width_in) {
				smp_safe = width_in - 1;
			}
			for (j=0; j<cmp; j++) {
				samples[k * cmp + j] = in[smp_safe * cmp + j];
			}
		}
		ref_set_sample(taps, coeffs, samples, out + i * cmp, cs);
	}

	free(samples);
	free(coeffs);
}

static void validate_scanline16(uint16_t *oil, uint16_t *ref, size_t len)
{
	size_t i;
	uint16_t diff;
	for (i=0; i<len; i++) {
		diff = labs((long)oil[i] - ref[i]);
		if (diff > 1) {
			fprintf(stderr, "oil got: %d, reference was %d\n", oil[i], ref[i]);
		}
	}
}

static void validate_scanline8(uint8_t *oil, uint8_t *ref, size_t len)
{
	size_t i;
	uint16_t diff;
	for (i=0; i<len; i++) {
		diff = labs((int)oil[i] - ref[i]);
		if (diff > 1) {
			fprintf(stderr, "%zu: oil got: %d, reference was %d\n", i, oil[i], ref[i]);
		}
	}
}

static void test_xscale(long width_in, long width_out, enum oil_colorspace cs)
{
	uint8_t cmp;
	uint16_t *psl_buf, *inbuf, *outbuf, *ref_outbuf;
	size_t psl_len, psl_offset;

	cmp = CS_TO_CMP(cs);
	psl_len = padded_sl_len_offset(width_in, width_out, cmp, &psl_offset);
	psl_buf = malloc(psl_len);
	inbuf = psl_buf + psl_offset;
	fill_rand(inbuf, width_in * cmp);
	outbuf = malloc(width_out * cmp * sizeof(uint16_t));
	ref_outbuf = malloc(width_out * cmp * sizeof(uint16_t));

	padded_sl_extend_edges(psl_buf, width_in, psl_offset, cmp);
	xscale_padded(inbuf, width_in, outbuf, width_out, cs);
	ref_xscale(inbuf, width_in, ref_outbuf, width_out, cs);
	validate_scanline16(outbuf, ref_outbuf, width_out * cmp);

	free(ref_outbuf);
	free(outbuf);
	free(psl_buf);
}

static void test_xscale_fmt(long width_in, long width_out)
{
	test_xscale(width_in, width_out, OIL_CS_G);
	test_xscale(width_in, width_out, OIL_CS_GA);
	test_xscale(width_in, width_out, OIL_CS_RGB);
	test_xscale(width_in, width_out, OIL_CS_RGBA);
	test_xscale(width_in, width_out, OIL_CS_RGBX);
	test_xscale(width_in, width_out, OIL_CS_CMYK);
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

static void postprocess(uint16_t *in, uint8_t *out, enum oil_colorspace cs)
{
	switch (cs) {
	case OIL_CS_G:
		out[0] = clamp_16_to_8(in[0]);
		break;
	case OIL_CS_GA:
		out[0] = clamp_16_to_8(in[0]);
		out[1] = clamp_16_to_8(in[1]);
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
		out[0] = linear_sample_to_srgb_reference(in[0]);
		out[1] = linear_sample_to_srgb_reference(in[1]);
		out[2] = linear_sample_to_srgb_reference(in[2]);
		out[3] = clamp_16_to_8(in[3]);
		break;
	case OIL_CS_CMYK:
		out[0] = clamp_16_to_8(in[0]);
		out[1] = clamp_16_to_8(in[1]);
		out[2] = clamp_16_to_8(in[2]);
		out[3] = clamp_16_to_8(in[3]);
		break;
	}
}

static void ref_strip_scale(uint16_t **in, long taps, long width, uint8_t *out,
	long double ty, enum oil_colorspace cs_out)
{
	size_t i, j;
	long double *coeffs;
	uint16_t *tmp, *samples;
	uint8_t k, cmp_in;
	enum oil_colorspace cs_in;

	cs_in = cs_out;
	if (cs_in == OIL_CS_RGB) {
		cs_in = OIL_CS_RGBX;
	}
	cmp_in = CS_TO_CMP(cs_in);
	coeffs = malloc(taps * sizeof(long double));
	samples = malloc(taps * cmp_in * sizeof(uint16_t));
	tmp = malloc(cmp_in * sizeof(uint16_t));
	ref_calc_coeffs(coeffs, ty, taps);

	for (i=0; i<width; i++) {
		for (k=0; k<cmp_in; k++) {
			for (j=0; j<taps; j++) {
				samples[j * cmp_in + k] = in[j][i * cmp_in + k];
			}
		}
		ref_set_sample(taps, coeffs, samples, tmp, cs_in);
		postprocess(tmp, out, cs_out);
		out += CS_TO_CMP(cs_out);
	}

	free(tmp);
	free(coeffs);
	free(samples);
}

static void test_strip_scale(long taps, long width, float ty,
	enum oil_colorspace cs)
{
	uint8_t cmp_in, cmp_out, *out_oil, *out_ref;
	uint16_t **scanlines;
	long i;

	cmp_out = CS_TO_CMP(cs);
	cmp_in = cmp_out;
	if (cs == OIL_CS_RGB) {
		cmp_in = 4;
	}
	scanlines = malloc(taps * sizeof(uint16_t *));
	out_oil = malloc(width * cmp_out * sizeof(uint8_t));
	out_ref = malloc(width * cmp_out * sizeof(uint8_t));

	for (i=0; i<taps; i++) {
		scanlines[i] = malloc(width * cmp_in * sizeof(uint16_t));
		fill_rand(scanlines[i], width * cmp_in);
	}

	strip_scale(scanlines, taps, width * cmp_in, out_oil, ty, cs);
	ref_strip_scale(scanlines, taps, width, out_ref, ty, cs);
	validate_scanline8(out_oil, out_ref, width * cmp_out);

	for (i=0; i<taps; i++) {
		free(scanlines[i]);
	}
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
	test_strip_scale_each_cs(4, 1000,  0.2345);
	test_strip_scale_each_cs(8, 1000,  0.5);
	test_strip_scale_each_cs(12, 1000, 0.0);
	test_strip_scale_each_cs(12, 1000, 0.00005);
	test_strip_scale_each_cs(12, 1000, 0.99999);
}

/**
 * sl_rbuf
 */
static void test_sl_rbuf()
{
	uint16_t *p, *buf1, *buf2, *buf3, *buf4, **virt;
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
	uint16_t *tmp;
	uint8_t *buf;

	buf = malloc(1024 * 4);
	yscaler_init(&ys, 1000, 30, 1024, OIL_CS_RGBA);
	j = 0;
	for (i=0; i<30; i++) {
		while((tmp = yscaler_next(&ys))) {
			j++;
			fill_rand(tmp, 1024 * 4);
		}
		yscaler_scale(&ys, buf, i);
	}
	assert(j == 1000);
	yscaler_free(&ys);
	free(buf);
}

static uint16_t srgb_sample_to_linear_reference(uint8_t in)
{
	long double in_f, result, tmp;
	in_f = in / 255.0L;
	if (in_f <= 0.0404482362771082L) {
		result = in_f / 12.92L;
	} else {
		tmp = ((in_f + 0.055L)/1.055L);
		result = powl(tmp, 2.4L);
	}
	return round(result * 65535);
}

static void test_srgb_to_linear()
{
	uint16_t input, reference, actual;

	for (input=0; input<=255; input++) {
		reference = srgb_sample_to_linear_reference(input);
		actual = srgb_sample_to_linear(input);
		assert(actual == reference);
	}
}

static void test_linear_to_srgb()
{
	uint8_t reference, actual;
	uint32_t input;

	for (input=0; input<65536; input++) {
		reference = linear_sample_to_srgb_reference(input);
		actual = linear_sample_to_srgb(input);
		assert(actual == reference);
	}
}

static void test_roundtrip()
{
	uint16_t input, linear, back_to_srgb;

	back_to_srgb = 0;

	for (input=0; input<=255; input++) {
		linear = srgb_sample_to_linear(input);
		back_to_srgb = linear_sample_to_srgb(linear);
		assert(back_to_srgb == input);
	}
}

static void test_colorspace_helpers()
{
	test_srgb_to_linear();
	test_linear_to_srgb();
	test_roundtrip();
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
	test_colorspace_helpers();
	printf("All tests pass.\n");
	return 0;
}
