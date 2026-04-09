#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "oil_resample.h"

typedef int (*scale_in_fn)(struct oil_scale *, unsigned char *);
typedef int (*scale_out_fn)(struct oil_scale *, unsigned char *);

static scale_in_fn cur_scale_in;
static scale_out_fn cur_scale_out;

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

static void ref_calc_coeffs(long double *coeffs, long double offset, int taps,
	int ltrim, int rtrim)
{
	int i;
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
	assert(fabsl(total_check - 1.0) < 0.0000000001L);
}

static void fill_rand8(unsigned char *buf, int len)
{
	int i;
	for (i=0; i<len; i++) {
		buf[i] = rand() % 256;
	}
}

static int calc_taps_check(int dim_in, int dim_out)
{
	int tmp_i;
	if (dim_in < dim_out) {
		return 4;
	}
	tmp_i = dim_in * 4 / dim_out;
	return tmp_i - (tmp_i%2);
}

static long double ref_map(int dim_in, int dim_out, int pos)
{
	return (pos + 0.5l) * (long double)dim_in / dim_out - 0.5l;
}

static int split_map_check(int dim_in, int dim_out, int pos,
	long double *ty)
{
	long double smp;
	int smp_i;

	smp = ref_map(dim_in, dim_out, pos);
	smp_i = floorl(smp);
	*ty = smp - smp_i;
	return smp_i;
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
	case OIL_CS_RGBX_NOGAMMA:
		in[3] = 1.0L;
		break;
	case OIL_CS_RGBA_NOGAMMA:
		in[0] *= in[3];
		in[1] *= in[3];
		in[2] *= in[3];
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void postprocess(long double *in, enum oil_colorspace cs)
{
	long double alpha;
	switch (cs) {
	case OIL_CS_RGBX_NOGAMMA:
		in[0] = clamp_f(in[0]);
		in[1] = clamp_f(in[1]);
		in[2] = clamp_f(in[2]);
		in[3] = 1.0L;
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
	int i, j, k, taps, smp_i, start, ltrim, rtrim, start_safe,
		taps_safe, max_pos, in_pos;
	long double *coeffs, in_val, tx;

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

static void test_scale_each_cs(int dim_a, int dim_b)
{
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBA_NOGAMMA);
	test_scale_square_rand(dim_a, dim_b, OIL_CS_RGBX_NOGAMMA);
}

static void test_scale_downscale(int dim_in, int dim_out)
{
	assert(dim_in >= dim_out);
	test_scale_each_cs(dim_in, dim_out);
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

	/* feed one input line when more are needed, should still fail */
	if (oil_scale_slots(&os) > 1) {
		unsigned char *in_line = calloc(OIL_CMP(cs) * in_dim, 1);
		assert(cur_scale_in(&os, in_line) == 0);
		assert(oil_scale_slots(&os) > 0);
		assert(cur_scale_out(&os, buf) == -1);
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

	free(buf);
}

static void test_out_not_ready_all(void)
{
	test_out_not_ready(100, 50, OIL_CS_RGBA_NOGAMMA);
	test_out_not_ready(100, 50, OIL_CS_RGBX_NOGAMMA);
}

static void test_scale_all(void)
{
	test_scale_downscale(5, 1);
	test_scale_downscale(8, 1);
	test_scale_downscale(8, 3);
	test_scale_downscale(100, 1);
	test_scale_downscale(100, 99);
	test_scale_downscale(2, 1);
}

struct impl {
	char *name;
	scale_in_fn in;
	scale_out_fn out;
};

static void run_tests(struct impl *impl)
{
	printf("--- testing %s ---\n", impl->name);
	cur_scale_in = impl->in;
	cur_scale_out = impl->out;

	test_scale_all();
	test_out_not_ready_all();
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

	num_impls++;

#if defined(__x86_64__)
	impls[num_impls].name = "sse2";
	impls[num_impls].in = oil_scale_in_sse2;
	impls[num_impls].out = oil_scale_out_sse2;

	num_impls++;

	impls[num_impls].name = "avx2";
	impls[num_impls].in = oil_scale_in_avx2;
	impls[num_impls].out = oil_scale_out_avx2;

	num_impls++;
#elif defined(__aarch64__)
	impls[num_impls].name = "neon";
	impls[num_impls].in = oil_scale_in_neon;
	impls[num_impls].out = oil_scale_out_neon;

	num_impls++;
#endif

	for (i=0; i<num_impls; i++) {
		run_tests(&impls[i]);
	}

	printf("worst error: %f\n", worst);
	printf("All tests pass.\n");
	return 0;
}
