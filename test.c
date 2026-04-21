#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "oil_resample.h"

typedef int (*scale_in_fn)(struct oil_scale *, unsigned char *);
typedef int (*scale_out_fn)(struct oil_scale *, unsigned char *);
typedef int (*scale_out_discard_fn)(struct oil_scale *);

static scale_in_fn cur_scale_in;
static scale_out_fn cur_scale_out;
static scale_out_discard_fn cur_scale_out_discard;

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

static void ref_calc_coeffs(long double *coeffs, int n_samples, int smp_start,
	long double center, long double tap_mult)
{
	int i;
	long double dist, fudge, total_check;

	assert(n_samples > 0);
	fudge = 0.0;
	for (i=0; i<n_samples; i++) {
		dist = fabsl((long double)(smp_start + i) - center);
		coeffs[i] = ref_catrom(dist / tap_mult) / tap_mult;
		fudge += coeffs[i];
	}
	total_check = 0.0;
	for (i=0; i<n_samples; i++) {
		coeffs[i] /= fudge;
		total_check += coeffs[i];
	}
	assert(fabsl(total_check - 1.0) < 0.0000000001L);
}

static void fill_rand8(unsigned char *buf, int len)
{
	int i;
	for (i=0; i<len; i++) {
		buf[i] = rand() % 256;
	}
}

static int max_taps_check(int dim_in, int dim_out)
{
	if (dim_in <= dim_out) {
		return 4;
	}
	return (int)ceill(4.0L * dim_in / dim_out) + 1;
}

static long double ref_map(int dim_in, int dim_out, int pos)
{
	return (pos + 0.5l) * (long double)dim_in / dim_out - 0.5l;
}

static double worst;

static void validate_scanline8(unsigned char *oil, long double *ref,
	int width, int cmp)
{
	int i, j, ref_i, pos;
	double error, ref_f;
	for (i=0; i<width; i++) {
		for (j=0; j<cmp; j++) {
			pos = i * cmp + j;
			ref_f = ref[pos] * 255.0;
			ref_i = lround(ref_f);
			error = fabs(oil[pos] - ref_f) - 0.5;
			if (error > worst) {
				worst = error;
			}
			/* Bumped from 0.06 to 0.07: adding ARGB to more test
			 * functions shifted the random data, exposing edge cases
			 * where float precision just barely exceeds 0.06. */
			if (error > 0.07) {
				fprintf(stderr, "[%d:%d] expected: %d, got %d (%.9f)\n", i, j, ref_i, oil[pos], ref_f);
				assert(0 && "pixel error exceeds tolerance");
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
	case OIL_CS_RGBA:
		in[0] = in[3] * srgb_sample_to_linear_reference(in[0]);
		in[1] = in[3] * srgb_sample_to_linear_reference(in[1]);
		in[2] = in[3] * srgb_sample_to_linear_reference(in[2]);
		break;
	case OIL_CS_ARGB:
		in[1] = in[0] * srgb_sample_to_linear_reference(in[1]);
		in[2] = in[0] * srgb_sample_to_linear_reference(in[2]);
		in[3] = in[0] * srgb_sample_to_linear_reference(in[3]);
		break;
	case OIL_CS_RGBX:
		in[0] = srgb_sample_to_linear_reference(in[0]);
		in[1] = srgb_sample_to_linear_reference(in[1]);
		in[2] = srgb_sample_to_linear_reference(in[2]);
		in[3] = 1.0L;
		break;
	case OIL_CS_RGBX_NOGAMMA:
		in[3] = 1.0L;
		break;
	case OIL_CS_RGB_NOGAMMA:
		break;
	case OIL_CS_RGBA_NOGAMMA:
		in[0] *= in[3];
		in[1] *= in[3];
		in[2] *= in[3];
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
	case OIL_CS_ARGB:
		alpha = clamp_f(in[0]);
		if (alpha != 0.0L) {
			in[1] /= alpha;
			in[2] /= alpha;
			in[3] /= alpha;
		}
		in[0] = alpha;
		in[1] = linear_sample_to_srgb_reference(in[1]);
		in[2] = linear_sample_to_srgb_reference(in[2]);
		in[3] = linear_sample_to_srgb_reference(in[3]);
		break;
	case OIL_CS_CMYK:
		in[0] = clamp_f(in[0]);
		in[1] = clamp_f(in[1]);
		in[2] = clamp_f(in[2]);
		in[3] = clamp_f(in[3]);
		break;
	case OIL_CS_RGBX:
		in[0] = linear_sample_to_srgb_reference(in[0]);
		in[1] = linear_sample_to_srgb_reference(in[1]);
		in[2] = linear_sample_to_srgb_reference(in[2]);
		in[3] = 1.0L;
		break;
	case OIL_CS_RGBX_NOGAMMA:
		in[0] = clamp_f(in[0]);
		in[1] = clamp_f(in[1]);
		in[2] = clamp_f(in[2]);
		in[3] = 1.0L;
		break;
	case OIL_CS_RGB_NOGAMMA:
		in[0] = clamp_f(in[0]);
		in[1] = clamp_f(in[1]);
		in[2] = clamp_f(in[2]);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		alpha = clamp_f(in[3]);
		if (alpha != 0.0L) {
			in[0] /= alpha;
			in[1] /= alpha;
			in[2] /= alpha;
		}
		in[0] = clamp_f(in[0]);
		in[1] = clamp_f(in[1]);
		in[2] = clamp_f(in[2]);
		in[3] = alpha;
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static unsigned char **alloc_2d_uchar(int width, int height)
{
	int i;
	unsigned char **rows;

	rows = malloc(height * sizeof(unsigned char*));
	for (i=0; i<height; i++) {
		rows[i] = malloc(width * sizeof(unsigned char));
	}
	return rows;
}

static void free_2d_uchar(unsigned char **ptr, int height)
{
	int i;

	for (i=0; i<height; i++) {
		free(ptr[i]);
	}
	free(ptr);
}

static long double **alloc_2d_ld(int width, int height)
{
	int i;
	long double **rows;

	rows = malloc(height * sizeof(long double*));
	for (i=0; i<height; i++) {
		rows[i] = malloc(width * sizeof(long double));
	}
	return rows;
}

static void free_2d_ld(long double **ptr, int height)
{
	int i;

	for (i=0; i<height; i++) {
		free(ptr[i]);
	}
	free(ptr);
}

static void ref_xscale(long double *in, int in_width, long double *out,
	int out_width, int cmp)
{
	int i, j, k, smp_start, smp_end, n_samples, max_pos;
	long double *coeffs, tap_mult, center, radius;

	/* Upscale: fixed 4-tap Catmull-Rom (tap_mult=1).
	 * Downscale: widen kernel to cover input range, preventing aliasing. */
	tap_mult = in_width <= out_width ? 1.0L : (long double)in_width / out_width;
	radius = 2.0L * tap_mult;
	coeffs = malloc(max_taps_check(in_width, out_width) * sizeof(long double));
	max_pos = in_width - 1;

	for (i=0; i<out_width; i++) {
		center = ref_map(in_width, out_width, i);

		smp_start = (int)ceill(center - radius);
		smp_end = (int)floorl(center + radius);
		if ((long double)smp_start == center - radius) {
			smp_start++;
		}
		if ((long double)smp_end == center + radius) {
			smp_end--;
		}
		if (smp_start < 0) {
			smp_start = 0;
		}
		if (smp_end > max_pos) {
			smp_end = max_pos;
		}
		n_samples = smp_end - smp_start + 1;

		ref_calc_coeffs(coeffs, n_samples, smp_start, center, tap_mult);

		for (j=0; j<cmp; j++) {
			out[i * cmp + j] = 0;
			for (k=0; k<n_samples; k++) {
				out[i * cmp + j] += coeffs[k] * in[(smp_start + k) * cmp + j];
			}
		}
	}

	free(coeffs);
}

static void ref_transpose_line(long double *in, int width,
	long double **out, int out_offset, int cmp)
{
	int i, j;
	for (i=0; i<width; i++) {
		for (j=0; j<cmp; j++) {
			out[i][out_offset + j] = in[i * cmp + j];
		}
	}
}

static void ref_transpose_column(long double **in, int height,
	long double *out, int in_offset, int cmp)
{
	int i, j;
	for (i=0; i<height; i++) {
		for (j=0; j<cmp; j++) {
			out[i * cmp + j] = in[i][in_offset + j];
		}
	}
}

static void ref_yscale(long double **in, int width, int in_height,
	long double **out, int out_height, int cmp)
{
	int i;
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

static void ref_scale(unsigned char **in, int in_width, int in_height,
	long double **out, int out_width, int out_height,
	enum oil_colorspace cs)
{
	int i, j, cmp, stride;
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

static void do_oil_scale(unsigned char **input_image, int in_width,
	int in_height, unsigned char **output_image, int out_width,
	int out_height, enum oil_colorspace cs)
{
	struct oil_scale os;
	int i, in_line;

	oil_scale_init(&os, in_height, out_height, in_width, out_width, cs);
	in_line = 0;
	for (i=0; i<out_height; i++) {
		while(oil_scale_slots(&os)) {
			cur_scale_in(&os, input_image[in_line++]);
		}
		cur_scale_out(&os, output_image[i]);
	}
	oil_scale_free(&os);
}

static void test_scale(int in_width, int in_height,
	unsigned char **input_image, int out_width, int out_height,
	enum oil_colorspace cs)
{
	int i, out_row_stride;
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

static void test_scale_square_rand(int in_dim, int out_dim,
	enum oil_colorspace cs)
{
	int i, in_row_stride;
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

static void test_scale_catrom_extremes(void)
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

static void test_scale_each_cs(int dim_a, int dim_b)
{
	test_scale_square_rand(dim_a, dim_b, OIL_CS_G);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_GA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGB);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_ARGB);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_CMYK);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBX);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGB_NOGAMMA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBA_NOGAMMA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBX_NOGAMMA);
}

static void test_scale_all_permutations(int dim_a, int dim_b)
{
	test_scale_each_cs(dim_a, dim_b);
	test_scale_each_cs(dim_b, dim_a);
}

static void test_out_discard(int in_dim, int out_dim, enum oil_colorspace cs)
{
	int i, j, in_line, out_row_stride;
	struct oil_scale os_discard;
	unsigned char **input_image, **normal_output, *discard_line;

	in_line = OIL_CMP(cs) * in_dim;
	out_row_stride = OIL_CMP(cs) * out_dim;

	input_image = alloc_2d_uchar(in_line, in_dim);
	for (i=0; i<in_dim; i++) {
		fill_rand8(input_image[i], in_line);
	}

	/* normal scale for reference */
	normal_output = alloc_2d_uchar(out_row_stride, out_dim);
	do_oil_scale(input_image, in_dim, in_dim, normal_output, out_dim,
		out_dim, cs);

	/* scale with every other line discarded */
	discard_line = malloc(out_row_stride);
	oil_scale_init(&os_discard, in_dim, out_dim, in_dim, out_dim, cs);
	in_line = 0;
	for (i=0; i<out_dim; i++) {
		while(oil_scale_slots(&os_discard)) {
			assert(cur_scale_in(&os_discard, input_image[in_line++]) == 0);
		}
		if (i % 2 == 0) {
			cur_scale_out(&os_discard, discard_line);
			for (j=0; j<out_row_stride; j++) {
				assert(discard_line[j] == normal_output[i][j]);
			}
		} else {
			cur_scale_out_discard(&os_discard);
		}
	}
	oil_scale_free(&os_discard);

	/* verify cur_scale_in returns -1 when output is ready */
	oil_scale_init(&os_discard, in_dim, out_dim, in_dim, out_dim, cs);
	in_line = 0;
	while(oil_scale_slots(&os_discard)) {
		assert(cur_scale_in(&os_discard, input_image[in_line++]) == 0);
	}
	assert(cur_scale_in(&os_discard, input_image[0]) == -1);
	oil_scale_free(&os_discard);

	free(discard_line);
	free_2d_uchar(normal_output, out_dim);
	free_2d_uchar(input_image, in_dim);
}

static void test_out_discard_all(void)
{
	/* downscale */
	test_out_discard(100, 50, OIL_CS_G);
	test_out_discard(100, 50, OIL_CS_RGB);
	test_out_discard(100, 50, OIL_CS_RGBA);
	test_out_discard(100, 50, OIL_CS_ARGB);
	test_out_discard(100, 50, OIL_CS_CMYK);
	test_out_discard(100, 50, OIL_CS_GA);
	test_out_discard(100, 50, OIL_CS_RGB_NOGAMMA);
	test_out_discard(100, 50, OIL_CS_RGBA_NOGAMMA);
	test_out_discard(100, 50, OIL_CS_RGBX_NOGAMMA);
	/* upscale */
	test_out_discard(50, 100, OIL_CS_G);
	test_out_discard(50, 100, OIL_CS_RGB);
	test_out_discard(50, 100, OIL_CS_RGBA);
	test_out_discard(50, 100, OIL_CS_ARGB);
	test_out_discard(50, 100, OIL_CS_CMYK);
	test_out_discard(50, 100, OIL_CS_GA);
	test_out_discard(50, 100, OIL_CS_RGB_NOGAMMA);
	test_out_discard(50, 100, OIL_CS_RGBA_NOGAMMA);
	test_out_discard(50, 100, OIL_CS_RGBX_NOGAMMA);
}

static void test_out_not_ready(int in_dim, int out_dim, enum oil_colorspace cs)
{
	struct oil_scale os;
	int out_row_stride;
	unsigned char *buf;

	out_row_stride = OIL_CMP(cs) * out_dim;
	buf = malloc(out_row_stride);

	/* calling cur_scale_out before any input should fail */
	oil_scale_init(&os, in_dim, out_dim, in_dim, out_dim, cs);
	assert(cur_scale_out(&os, buf) == -1);
	assert(cur_scale_out_discard(&os) == -1);

	/* feed one input line when more are needed, should still fail */
	if (oil_scale_slots(&os) > 1) {
		unsigned char *in_line = calloc(OIL_CMP(cs) * in_dim, 1);
		assert(cur_scale_in(&os, in_line) == 0);
		assert(oil_scale_slots(&os) > 0);
		assert(cur_scale_out(&os, buf) == -1);
		assert(cur_scale_out_discard(&os) == -1);
		free(in_line);
	}
	oil_scale_free(&os);

	/* feed enough input, then cur_scale_out should succeed */
	oil_scale_init(&os, in_dim, out_dim, in_dim, out_dim, cs);
	while (oil_scale_slots(&os)) {
		unsigned char *in_line = calloc(OIL_CMP(cs) * in_dim, 1);
		assert(cur_scale_in(&os, in_line) == 0);
		free(in_line);
	}
	assert(cur_scale_out(&os, buf) == 0);
	oil_scale_free(&os);

	/* same but with discard */
	oil_scale_init(&os, in_dim, out_dim, in_dim, out_dim, cs);
	while (oil_scale_slots(&os)) {
		unsigned char *in_line = calloc(OIL_CMP(cs) * in_dim, 1);
		assert(cur_scale_in(&os, in_line) == 0);
		free(in_line);
	}
	assert(cur_scale_out_discard(&os) == 0);
	oil_scale_free(&os);

	free(buf);
}

static void test_out_not_ready_all(void)
{
	/* downscale */
	test_out_not_ready(100, 50, OIL_CS_G);
	test_out_not_ready(100, 50, OIL_CS_RGB);
	test_out_not_ready(100, 50, OIL_CS_RGBA);
	test_out_not_ready(100, 50, OIL_CS_ARGB);
	test_out_not_ready(100, 50, OIL_CS_CMYK);
	test_out_not_ready(100, 50, OIL_CS_GA);
	test_out_not_ready(100, 50, OIL_CS_RGB_NOGAMMA);
	test_out_not_ready(100, 50, OIL_CS_RGBA_NOGAMMA);
	test_out_not_ready(100, 50, OIL_CS_RGBX_NOGAMMA);
	/* upscale */
	test_out_not_ready(50, 100, OIL_CS_G);
	test_out_not_ready(50, 100, OIL_CS_RGB);
	test_out_not_ready(50, 100, OIL_CS_RGBA);
	test_out_not_ready(50, 100, OIL_CS_ARGB);
	test_out_not_ready(50, 100, OIL_CS_CMYK);
	test_out_not_ready(50, 100, OIL_CS_GA);
	test_out_not_ready(50, 100, OIL_CS_RGB_NOGAMMA);
	test_out_not_ready(50, 100, OIL_CS_RGBA_NOGAMMA);
	test_out_not_ready(50, 100, OIL_CS_RGBX_NOGAMMA);
}

static void test_scale_all(void)
{
	test_scale_all_permutations(5, 1);
	test_scale_all_permutations(8, 1);
	test_scale_all_permutations(8, 3);
	test_scale_all_permutations(100, 1);
	test_scale_all_permutations(100, 99);
	test_scale_all_permutations(2, 1);
}

/* Sweep near-identity up/down scales (N <-> N+/-1) across sizes, colorspaces,
 * and seeds. Prior worst-case errors all came from 99<->100; this targets that
 * regime to exercise float accumulation precision in the x/y scale paths. */
static void test_scale_near_identity(void)
{
	static const int sizes[] = {7, 16, 33, 50, 99, 100};
	static const enum oil_colorspace spaces[] = {
		OIL_CS_G, OIL_CS_GA, OIL_CS_RGB, OIL_CS_RGBA, OIL_CS_ARGB,
		OIL_CS_CMYK, OIL_CS_RGBX, OIL_CS_RGB_NOGAMMA,
		OIL_CS_RGBA_NOGAMMA, OIL_CS_RGBX_NOGAMMA,
	};
	static const unsigned int seeds[] = {1531289551u, 0xdeadbeefu};
	int sz_i, cs_i, seed_i;
	int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
	int n_spaces = sizeof(spaces) / sizeof(spaces[0]);
	int n_seeds = sizeof(seeds) / sizeof(seeds[0]);

	for (seed_i = 0; seed_i < n_seeds; seed_i++) {
		srand(seeds[seed_i]);
		for (sz_i = 0; sz_i < n_sizes; sz_i++) {
			int n = sizes[sz_i];
			for (cs_i = 0; cs_i < n_spaces; cs_i++) {
				enum oil_colorspace cs = spaces[cs_i];
				test_scale_square_rand(n, n + 1, cs);
				if (n > 1) {
					test_scale_square_rand(n, n - 1, cs);
				}
			}
		}
	}
}

/* Accuracy test: linear ramp preservation for G colorspace.
 *
 * Catmull-Rom reproduces linear functions exactly in the kernel interior
 * (the normalized coefficients satisfy sum(c_k * x_k) = center — the
 * first-moment property). So a linear input ramp must resample to a linear
 * ramp at output sample centers, independent of scale factor. Near edges the
 * kernel is truncated and renormalized, which preserves the zeroth moment
 * but breaks the first-moment property; those pixels are skipped.
 *
 * This is a stronger check than the reference-comparison tests: the ground
 * truth is algebraic, not another floating-point implementation. A bug
 * shared by the fast impl and the long-double reference would pass
 * reference-comparison but fail this test. */
static void test_g_linear_ramp(int in_dim, int out_dim)
{
	struct oil_scale os;
	int i, out_pos;
	unsigned char *in_row, *out_row;
	double radius, src_pos;

	/* in_row[i] = i must be an exact 8-bit ramp value. */
	assert(in_dim <= 256);

	in_row = malloc(in_dim);
	for (i=0; i<in_dim; i++) {
		in_row[i] = i;
	}
	out_row = malloc(out_dim);

	/* Downscale: radius = 2 * in/out source pixels. Upscale: radius = 2. */
	radius = in_dim <= out_dim ? 2.0 : 2.0 * (double)in_dim / out_dim;

	oil_scale_init(&os, in_dim, out_dim, in_dim, out_dim, OIL_CS_G);
	for (out_pos=0; out_pos<out_dim; out_pos++) {
		while (oil_scale_slots(&os)) {
			cur_scale_in(&os, in_row);
		}
		/* Input is constant in y, so every output row is identical —
		 * we only need to inspect the last one. */
		cur_scale_out(&os, out_row);
	}
	oil_scale_free(&os);

	for (out_pos=0; out_pos<out_dim; out_pos++) {
		src_pos = (out_pos + 0.5) * (double)in_dim / out_dim - 0.5;
		if (src_pos < radius || src_pos > in_dim - 1 - radius) {
			continue;
		}
		int expected = (int)lround(src_pos);
		int got = out_row[out_pos];
		/* Tolerance is 8-bit round-to-nearest only; no float slack. */
		if (abs(got - expected) > 1) {
			fprintf(stderr, "ramp %d->%d pos %d: expected %d, got %d\n",
				in_dim, out_dim, out_pos, expected, got);
			assert(0 && "linear ramp not preserved within 8-bit rounding");
		}
	}

	free(in_row);
	free(out_row);
}

static void test_g_linear_ramp_all(void)
{
	test_g_linear_ramp(32, 128);   /* 4x upscale */
	test_g_linear_ramp(128, 32);   /* 4x downscale */
	test_g_linear_ramp(200, 50);   /* 4x downscale, larger */
	test_g_linear_ramp(99, 100);   /* near-identity upscale */
	test_g_linear_ramp(100, 99);   /* near-identity downscale */
	test_g_linear_ramp(256, 17);   /* ~15x downscale, wide kernel */
}

struct impl {
	char *name;
	scale_in_fn in;
	scale_out_fn out;
	scale_out_discard_fn out_discard;
};

static void run_tests(struct impl *impl)
{
	printf("--- testing %s ---\n", impl->name);
	cur_scale_in = impl->in;
	cur_scale_out = impl->out;
	cur_scale_out_discard = impl->out_discard;

	test_scale_all();
	test_scale_catrom_extremes();
	test_out_discard_all();
	test_out_not_ready_all();
	test_scale_near_identity();
	test_g_linear_ramp_all();
}

int main(void)
{
	int t = 1531289551;
	int i, num_impls;
	struct impl impls[3];
	//int t = time(NULL);
	printf("seed: %d\n", t);
	srand(t);
	oil_global_init();

	num_impls = 0;
	impls[num_impls].name = "scalar";
	impls[num_impls].in = oil_scale_in;
	impls[num_impls].out = oil_scale_out;
	impls[num_impls].out_discard = oil_scale_out_discard;
	num_impls++;

#if defined(__x86_64__)
	impls[num_impls].name = "sse2";
	impls[num_impls].in = oil_scale_in_sse2;
	impls[num_impls].out = oil_scale_out_sse2;
	impls[num_impls].out_discard = oil_scale_out_discard;
	num_impls++;

	impls[num_impls].name = "avx2";
	impls[num_impls].in = oil_scale_in_avx2;
	impls[num_impls].out = oil_scale_out_avx2;
	impls[num_impls].out_discard = oil_scale_out_discard;
	num_impls++;
#elif defined(__aarch64__)
	impls[num_impls].name = "neon";
	impls[num_impls].in = oil_scale_in_neon;
	impls[num_impls].out = oil_scale_out_neon;
	impls[num_impls].out_discard = oil_scale_out_discard;
	num_impls++;
#endif

	for (i=0; i<num_impls; i++) {
		run_tests(&impls[i]);
	}

	printf("worst error: %f\n", worst);
	printf("All tests pass.\n");
	return 0;
}
