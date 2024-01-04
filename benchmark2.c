#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <png.h>
#include "oil_resample.h"
#include "oil_libpng.h"
#include <immintrin.h>

struct bench_image {
	unsigned char *buffer;
	int width;
	int height;
	enum oil_colorspace cs;
};

static float s2l_map[256];

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

static struct bench_image png(char *path, enum oil_colorspace cs)
{
	struct bench_image bench_image;
	int i;
	png_structp rpng;
	png_infop rinfo;
	FILE *input;
	size_t row_stride, buf_len;
	unsigned char **buf_ptrs;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (setjmp(png_jmpbuf(rpng))) {
		fprintf(stderr, "PNG Decoding Error.\n");
		exit(1);
	}

	input = fopen(path, "r");
	if (!input) {
		fprintf(stderr, "Unable to open file.\n");
		exit(1);
	}

	rinfo = png_create_info_struct(rpng);
	png_init_io(rpng, input);
	png_read_info(rpng, rinfo);

	if (png_get_color_type(rpng, rinfo) != PNG_COLOR_TYPE_RGBA) {
		fprintf(stderr, "Input image must be RGBA.\n");
		exit(1);
	};

	switch(cs) {
	case OIL_CS_G:
		png_set_rgb_to_gray(rpng, 1, -1, -1);
		png_set_strip_alpha(rpng);
		break;
	case OIL_CS_GA:
		png_set_rgb_to_gray(rpng, 1, -1, -1);
		break;
	case OIL_CS_RGB:
		png_set_strip_alpha(rpng);
		break;
	case OIL_CS_CMYK: /* Kind of cheating on CMYK by giving it RGBA */
	case OIL_CS_RGBA:
	case OIL_CS_UNKNOWN:
		break;
	}

	bench_image.width = png_get_image_width(rpng, rinfo);
	bench_image.height = png_get_image_height(rpng, rinfo);
	bench_image.cs = cs;

	row_stride = bench_image.width * OIL_CMP(cs);
	buf_len = (size_t)bench_image.height * row_stride;
	bench_image.buffer = malloc(buf_len);
	buf_ptrs = malloc(bench_image.height * sizeof(unsigned char *));
	if (!bench_image.buffer || !buf_ptrs) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	for (i=0; i<bench_image.height; i++) {
		buf_ptrs[i] = bench_image.buffer + i * row_stride;
	}

	png_read_image(rpng, buf_ptrs);
	png_destroy_read_struct(&rpng, &rinfo, NULL);

	free(buf_ptrs);
	fclose(input);
	return bench_image;
}

double time_to_ms(clock_t t)
{
	return (double)t / (CLOCKS_PER_SEC / 1000);
}

static void yscale_down_sse(float *in, int width, float *coeffs, float *sums)
{
	int i, sl_len;
	__m128 coeffs2, sums2, sample;

	coeffs2 = _mm_loadu_ps(coeffs);

	sl_len = width * 3;
	for (i=0; i<sl_len; i++) {
		sums2 = _mm_loadu_ps(sums);
		sample = _mm_set1_ps(in[i]);
		sums2 = _mm_add_ps(_mm_mul_ps(coeffs2, sample), sums2);
		_mm_store_ps(sums, sums2);
		sums += 4;
	}
}

static void xscale_down_rgb_sse(unsigned char *in, float *out, int out_width, float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 coeffs, sample, sum_r, sum_g, sum_b;

	sum_r = _mm_setzero_ps();
	sum_g = _mm_setzero_ps();
	sum_b = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			coeffs = _mm_loadu_ps(coeff_buf);

			sample = _mm_set1_ps(s2l_map[in[0]]);
			sum_r = _mm_add_ps(_mm_mul_ps(coeffs, sample), sum_r);

			sample = _mm_set1_ps(s2l_map[in[1]]);
			sum_g = _mm_add_ps(_mm_mul_ps(coeffs, sample), sum_g);

			sample = _mm_set1_ps(s2l_map[in[2]]);
			sum_b = _mm_add_ps(_mm_mul_ps(coeffs, sample), sum_b);

			in += 3;
			coeff_buf += 4;
		}

		_mm_store_ss(out, sum_r);
		_mm_store_ss(out+1, sum_g);
		_mm_store_ss(out+2, sum_b);

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);

		out += 3;
	}
}

static void xscale_down_rgb_sse_both(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, sample_x, sum_r, sum_g, sum_b, coeffs_y, sums_y,
		sample_y;

	coeffs_y = _mm_loadu_ps(coeffs_y_f);

	sum_r = _mm_setzero_ps();
	sum_g = _mm_setzero_ps();
	sum_b = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			coeffs_x = _mm_load_ps(coeffs_x_f);

			sample_x = _mm_set1_ps(s2l_map[in[0]]);
			sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

			sample_x = _mm_set1_ps(s2l_map[in[1]]);
			sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

			sample_x = _mm_set1_ps(s2l_map[in[2]]);
			sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

			in += 3;
			coeffs_x_f += 4;
		}

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_r, sum_r, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_g, sum_g, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_b, sum_b, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
	}
}

static void oil_scale_in2(struct oil_scale *os, unsigned char *in)
{
	float *coeffs;

	xscale_down_rgb_sse(in, os->rb, os->out_width, os->coeffs_x, os->borders_x);
	os->borders_y[os->out_pos] -= 1;

	coeffs = os->coeffs_y + os->in_pos * 4;
	yscale_down_sse(os->rb, os->out_width, coeffs, os->sums_y);

	os->in_pos++;
}

static void oil_scale_in3(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	xscale_down_rgb_sse_both(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
	os->borders_y[os->out_pos] -= 1;

	os->in_pos++;
}

clock_t resize(struct bench_image image, int out_width, int out_height)
{
	int i;
	enum oil_colorspace cs;
	struct oil_scale os;
	unsigned char *inbuf, *outbuf;
	size_t in_row_stride;
	clock_t t;

	cs = image.cs;
	in_row_stride = image.width * OIL_CMP(cs);

	inbuf = image.buffer;
	outbuf = malloc(out_width * OIL_CMP(cs));

	t = clock();
	oil_scale_init(&os, image.height, out_height, image.width, out_width, cs);

	for(i=0; i<out_height; i++) {
		while (oil_scale_slots(&os)) {

			oil_scale_in3(&os, inbuf);
			inbuf += in_row_stride;
		}
		oil_scale_out(&os, outbuf);
	}
	t = clock() - t;
	free(outbuf);
	oil_scale_free(&os);
	return t;
}

void do_bench(struct bench_image image, double ratio, int iterations)
{
	int i, out_width, out_height;
	clock_t t_min, t_tmp;

	out_width = round(image.width * ratio);
	out_height = 500000; /* reasonable maximum height */

	oil_fix_ratio(image.width, image.height, &out_width, &out_height);

	t_min = 0;
	for (i=0; i<iterations; i++) {
		t_tmp = resize(image, out_width, out_height);
		if (!t_min || t_tmp < t_min) {
			t_min = t_tmp;
		}
	}

	printf("    to %4dx%4d %6.2fms\n", out_width, out_height, time_to_ms(t_min));
}

void do_bench_sizes(char *name, char *path, enum oil_colorspace cs,
	int iterations)
{
	struct bench_image image;

	image = png(path, cs);

	printf("%dx%d %s\n", image.width, image.height, name);

	do_bench(image, 0.01, iterations);
	do_bench(image, 0.125, iterations);
	do_bench(image, 0.8, iterations);

	free(image.buffer);
}

int main(int argc, char *argv[])
{
	clock_t t;
	size_t i, num_spaces;
	int iterations;
	char *end;

	enum oil_colorspace spaces[] = {
		OIL_CS_G,
		OIL_CS_GA,
		OIL_CS_RGB,
		OIL_CS_RGBA,
		OIL_CS_CMYK,
	};

	char *space_names[] = {
		"G",
		"GA",
		"RGB",
		"RGBA",
		"CMYK",
	};

	if (argc < 2 || argc > 3) {
		fprintf(stderr, "Usage: %s <path> [colorspace]\n", argv[0]);
		return 1;
	}

	iterations = 100;
	if (getenv("OILITERATIONS")) {
		iterations = strtoul(getenv("OILITERATIONS"), &end, 10);
		if (!end) {
			fprintf(stderr, "Invalid environment variable OILITERATIONS.");
			return 1;
		}
	}
	fprintf(stderr, "Iterations: %d\n", iterations);

	t = clock();
	oil_global_init();
	t = clock() - t;
	build_s2l();
	printf("global init: %6.2fms\n", time_to_ms(t));

	num_spaces = sizeof(spaces)/sizeof(spaces[0]);
	if (argc == 2) {
		for (i=0; i<num_spaces; i++) {
			do_bench_sizes(space_names[i], argv[1], spaces[i], iterations);
		}
		return 0;
	}

	for (i=0; i<num_spaces; i++) {
		if (strcmp(space_names[i], argv[2]) == 0) {
			break;
		}
	}

	if (i >= num_spaces) {
		fprintf(stderr, "Colorspace not recognized.\n");
		return 1;
	}

	do_bench_sizes(space_names[i], argv[1], spaces[i], iterations);
	return 0;
}
