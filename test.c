#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "oil_resample.h"

static long double srgb_sample_to_linear_reference(long double in_f)
{
	long double tmp;
	if (in_f <= 0.0404482362771082L) {
		return in_f / 12.92L;
	} else {
		tmp = ((in_f + 0.055L)/1.055L);
		return powl(tmp, 2.4L);
	}
}

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

static void ref_calc_coeffs(long double *coeffs, long double offset, long taps,
	long ltrim, long rtrim)
{
	long i;
	long double tap_offset, tap_mult, fudge, total_check;

	assert(taps - ltrim - rtrim > 0);
	tap_mult = (long double)taps / 4;
	fudge = 0.0;
	for (i=0; i<taps; i++) {
		if (i<ltrim || i>=taps-rtrim) {
			coeffs[i] = 0;
			continue;
		}
		tap_offset = 1 - offset - taps / 2 + i;
		coeffs[i] = ref_catrom(fabsl(tap_offset) / tap_mult) / tap_mult;
		fudge += coeffs[i];
	}
	total_check = 0.0;
	for (i=0; i<taps; i++) {
		coeffs[i] /= fudge;
		total_check += coeffs[i];
	}
	assert(abs(total_check - 1.0) < 0.0000000001L);
}

static void fill_rand8(unsigned char *buf, long len)
{
	long i;
	for (i=0; i<len; i++) {
		buf[i] = rand() % 256;
	}
}

static long calc_taps_check(long dim_in, long dim_out)
{
	long tmp_i;
	if (dim_in < dim_out) {
		return 4;
	}
	tmp_i = dim_in * 4 / dim_out;
	return tmp_i - (tmp_i%2);
}

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

static long double worst;

static void validate_scanline8(unsigned char *oil, long double *ref,
	size_t width, int cmp)
{
	int i, j, ref_i, pos;
	long double error, ref_f;
	for (i=0; i<width; i++) {
		for (j=0; j<cmp; j++) {
			pos = i * cmp + j;
			ref_f = ref[pos] * 255.0L;
			ref_i = lroundl(ref_f);
			error = fabsl(oil[pos] - ref_f) - 0.5L;
			if (error > worst) {
				worst = error;
			}
			if (error > 0.06L) {
				fprintf(stderr, "[%d:%d] expected: %d, got %d (%.9Lf)\n", i, j, ref_i, oil[pos], ref_f);
			}
		}
	}
}

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

static void preprocess(long double *in, enum oil_colorspace cs)
{
	switch (cs) {
	case OIL_CS_G:
	case OIL_CS_CMYK:
	case OIL_CS_UNKNOWN:
		break;
	case OIL_CS_GA:
		in[0] *= in[1];
		break;
	case OIL_CS_RGB:
		in[0] = srgb_sample_to_linear_reference(in[0]);
		in[1] = srgb_sample_to_linear_reference(in[1]);
		in[2] = srgb_sample_to_linear_reference(in[2]);
		break;
	case OIL_CS_RGBX:
		in[0] = srgb_sample_to_linear_reference(in[0]);
		in[1] = srgb_sample_to_linear_reference(in[1]);
		in[2] = srgb_sample_to_linear_reference(in[2]);
		in[3] = 0;
		break;
	case OIL_CS_RGBA:
		in[0] = in[3] * srgb_sample_to_linear_reference(in[0]);
		in[1] = in[3] * srgb_sample_to_linear_reference(in[1]);
		in[2] = in[3] * srgb_sample_to_linear_reference(in[2]);
		break;
	}
}

static void postprocess(long double *in, enum oil_colorspace cs)
{
	long double alpha;
	switch (cs) {
	case OIL_CS_G:
		in[0] = clamp_f(in[0]);
		break;
	case OIL_CS_GA:
		alpha = clamp_f(in[1]);
		if (alpha != 0.0L) {
			in[0] /= alpha;
		}
		in[0] = clamp_f(in[0]);
		in[1] = alpha;
		break;
	case OIL_CS_RGB:
		in[0] = linear_sample_to_srgb_reference(in[0]);
		in[1] = linear_sample_to_srgb_reference(in[1]);
		in[2] = linear_sample_to_srgb_reference(in[2]);
		break;
	case OIL_CS_RGBX:
		in[0] = linear_sample_to_srgb_reference(in[0]);
		in[1] = linear_sample_to_srgb_reference(in[1]);
		in[2] = linear_sample_to_srgb_reference(in[2]);
		in[3] = 0;
		break;
	case OIL_CS_RGBA:
		alpha = clamp_f(in[3]);
		if (alpha != 0.0L) {
			in[0] /= alpha;
			in[1] /= alpha;
			in[2] /= alpha;
		}
		in[0] = linear_sample_to_srgb_reference(in[0]);
		in[1] = linear_sample_to_srgb_reference(in[1]);
		in[2] = linear_sample_to_srgb_reference(in[2]);
		in[3] = alpha;
		break;
	case OIL_CS_CMYK:
		in[0] = clamp_f(in[0]);
		in[1] = clamp_f(in[1]);
		in[2] = clamp_f(in[2]);
		in[3] = clamp_f(in[3]);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static unsigned char **alloc_2d_uchar(long width, long height)
{
	long i;
	unsigned char **rows;

	rows = malloc(height * sizeof(unsigned char*));
	for (i=0; i<height; i++) {
		rows[i] = malloc(width * sizeof(unsigned char));
	}
	return rows;
}

static void free_2d_uchar(unsigned char **ptr, long height)
{
	int i;

	for (i=0; i<height; i++) {
		free(ptr[i]);
	}
	free(ptr);
}

static long double **alloc_2d_ld(long width, long height)
{
	long i;
	long double **rows;

	rows = malloc(height * sizeof(long double*));
	for (i=0; i<height; i++) {
		rows[i] = malloc(width * sizeof(long double));
	}
	return rows;
}

static void free_2d_ld(long double **ptr, long height)
{
	int i;

	for (i=0; i<height; i++) {
		free(ptr[i]);
	}
	free(ptr);
}

static void ref_xscale(long double *in, long in_width, long double *out,
	long out_width, long cmp)
{
	long i, j, k, stride, taps, smp_i, start, ltrim, rtrim, start_safe,
		taps_safe, max_pos, in_pos;
	long double *tmp, *coeffs, in_val, tx;

	stride = cmp * in_width;
	tmp = malloc(stride * sizeof(long double));

	taps = calc_taps_check(in_width, out_width);
	coeffs = malloc(taps * sizeof(long double));
	max_pos = in_width - 1;
	for (i=0; i<out_width; i++) {
		smp_i = split_map_check(in_width, out_width, i, &tx);
		start = smp_i - (taps/2 - 1);

		start_safe = start;
		if (start_safe < 0) {
			start_safe = 0;
		}
		ltrim = start_safe - start;

		taps_safe = taps - ltrim;
		if (start_safe + taps_safe > max_pos) {
			taps_safe = max_pos - start_safe + 1;
		}
		rtrim = (start + taps) - (start_safe + taps_safe);

		ref_calc_coeffs(coeffs, tx, taps, ltrim, rtrim);

		for (j=0; j<cmp; j++) {
			out[i * cmp + j] = 0;
			for(k=0; k<taps_safe; k++) {
				in_pos = start_safe + k;
				in_val = in[in_pos * cmp + j];
				out[i * cmp + j] += coeffs[ltrim + k] * in_val;
			}
		}
	}

	free(tmp);
	free(coeffs);
}

static void ref_transpose_line(long double *in, long width,
	long double **out, long out_offset, long cmp)
{
	long i, j;
	for (i=0; i<width; i++) {
		for (j=0; j<cmp; j++) {
			out[i][out_offset + j] = in[i * cmp + j];
		}
	}
}

static void ref_transpose_column(long double **in, long height,
	long double *out, long in_offset, long cmp)
{
	long i, j;
	for (i=0; i<height; i++) {
		for (j=0; j<cmp; j++) {
			out[i * cmp + j] = in[i][in_offset + j];
		}
	}
}

static void ref_yscale(long double **in, long width, long in_height,
	long double **out, long out_height, long cmp)
{
	long i;
	long double *transposed, *trans_scaled;

	transposed = malloc(in_height * cmp * sizeof(long double));
	trans_scaled = malloc(out_height * cmp * sizeof(long double));
	for (i=0; i<width; i++) {
		ref_transpose_column(in, in_height, transposed, i * cmp, cmp);
		ref_xscale(transposed, in_height, trans_scaled, out_height, cmp);
		ref_transpose_line(trans_scaled, out_height, out, i * cmp, cmp);
	}
	free(transposed);
	free(trans_scaled);
}

static void ref_scale(unsigned char **in, long in_width, long in_height,
	long double **out, long out_width, long out_height,
	enum oil_colorspace cs)
{
	long i, j, cmp, stride;
	long double *pre_line, **intermediate;

	cmp = OIL_CMP(cs);
	stride = cmp * in_width;

	// horizontal scaling
	pre_line = malloc(stride * sizeof(long double));
	intermediate = alloc_2d_ld(out_width * cmp, in_height);
	for (i=0; i<in_height; i++) {
		// Convert chars to floats
		for (j=0; j<stride; j++) {
			pre_line[j] = in[i][j] / 255.0F;
		}

		// Preprocess
		for (j=0; j<in_width; j++) {
			preprocess(pre_line + j * cmp, cs);
		}

		// xscale
		ref_xscale(pre_line, in_width, intermediate[i], out_width, cmp);
	}

	// vertical scaling
	ref_yscale(intermediate, out_width, in_height, out, out_height, cmp);
	for (i=0; i<out_height; i++) {
		for (j=0; j<out_width; j++) {
			postprocess(out[i] + j * cmp, cs);
		}
	}

	free(pre_line);
	free_2d_ld(intermediate, in_height);
}

static void do_oil_scale(unsigned char **input_image, long in_width,
	long in_height, unsigned char **output_image, long out_width,
	long out_height, enum oil_colorspace cs)
{
	struct oil_scale os;
	long i, j, in_line;

	oil_scale_init(&os, in_height, out_height, in_width, out_width, cs);
	in_line = 0;
	for (i=0; i<out_height; i++) {
		for (j=oil_scale_slots(&os); j>0; j--) {
			oil_scale_in(&os, input_image[in_line++]);
		}
		oil_scale_out(&os, output_image[i]);
	}
	oil_scale_free(&os);
}

static void test_scale(long in_width, long in_height,
	unsigned char **input_image, long out_width, long out_height,
	enum oil_colorspace cs)
{
	long i, out_row_stride;
	unsigned char **oil_output_image;
	long double **ref_output_image;

	out_row_stride = OIL_CMP(cs) * out_width;

	/* oil scaling */
	oil_output_image = alloc_2d_uchar(out_row_stride, out_height);
	do_oil_scale(input_image, in_width, in_height, oil_output_image,
		out_width, out_height, cs);

	/* reference scaling */
	ref_output_image = alloc_2d_ld(out_row_stride, out_height);
	ref_scale(input_image, in_width, in_height, ref_output_image, out_width,
		out_height, cs);

	/* compare the two */
	for (i=0; i<out_height; i++) {
		validate_scanline8(oil_output_image[i], ref_output_image[i],
			out_width, OIL_CMP(cs));
	}

	free_2d_uchar(oil_output_image, out_height);
	free_2d_ld(ref_output_image, out_height);
}

static void test_scale_square_rand(long in_dim, long out_dim,
	enum oil_colorspace cs)
{
	long i, in_row_stride;
	unsigned char **input_image;

	in_row_stride = OIL_CMP(cs) * in_dim;

	/* Allocate & populate input image */
	input_image = alloc_2d_uchar(in_row_stride, in_dim);
	for (i=0; i<in_dim; i++) {
		fill_rand8(input_image[i], in_row_stride);
	}
	test_scale(in_dim, in_dim, input_image, out_dim, out_dim, cs);
	free_2d_uchar(input_image, in_dim);
}

static void test_scale_catrom_extremes()
{
	unsigned char **input_image;

	/* Allocate & populate input image */
	input_image = alloc_2d_uchar(4, 4);

	input_image[0][0] = 0;
	input_image[0][1] = 0;
	input_image[0][2] = 0;
	input_image[0][3] = 0;

	input_image[1][0] = 0;
	input_image[1][1] = 255;
	input_image[1][2] = 255;
	input_image[1][3] = 0;

	input_image[2][0] = 0;
	input_image[2][1] = 255;
	input_image[2][2] = 255;
	input_image[2][3] = 0;

	input_image[3][0] = 0;
	input_image[3][1] = 0;
	input_image[3][2] = 0;
	input_image[3][3] = 0;

	test_scale(4, 4, input_image, 7, 7, OIL_CS_G);
	free_2d_uchar(input_image, 4);
}

static void test_scale_each_cs(long dim_a, long dim_b)
{
	test_scale_square_rand(dim_a, dim_b, OIL_CS_G);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_GA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGB);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_CMYK);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBX);
}

static void test_scale_all_permutations(long dim_a, long dim_b)
{
	test_scale_each_cs(dim_a, dim_b);
	test_scale_each_cs(dim_b, dim_a);
}

static void test_scale_all()
{
	test_scale_all_permutations(5, 1);
	test_scale_all_permutations(8, 1);
	test_scale_all_permutations(8, 3);
	test_scale_all_permutations(100, 1);
	test_scale_all_permutations(100, 99);
	test_scale_all_permutations(2, 1);
}

int main()
{
	int t = 1531289551;
	//int t = time(NULL);
	printf("seed: %d\n", t);
	srand(t);
	oil_global_init();
	test_scale_all();
	test_scale_catrom_extremes();
	printf("worst error: %Lf\n", worst);
	printf("All tests pass.\n");
	return 0;
}
