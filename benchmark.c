#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <png.h>
#include "oil_resample.h"
#include "oil_libpng.h"

struct bench_image {
	unsigned char *buffer;
	int width;
	int height;
	enum oil_colorspace cs;
};

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
	case OIL_CS_CMYK: /* Kind of cheating on CMYK by giving it RGBA */
		png_set_strip_alpha(rpng);
		break;
	case OIL_CS_RGBX:
		png_set_strip_alpha(rpng);
		png_set_filler(rpng, 0xffff, PNG_FILLER_AFTER);
		break;
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
	buf_ptrs = malloc(bench_image.height * sizeof(unsigned char *));{}
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

int time_to_us(clock_t t)
{
	double time_taken;
	time_taken = (double)t / (CLOCKS_PER_SEC / 1000 / 1000);
	return round(time_taken);
}

clock_t resize(struct bench_image image, int out_width, int out_height)
{
	int i, j;
	enum oil_colorspace cs;
	struct oil_scale os;
	unsigned char *outbuf;
	size_t in_row_stride;
	clock_t t;

	cs = image.cs;
	in_row_stride = image.width * OIL_CMP(cs);

	outbuf = malloc(out_width * OIL_CMP(cs));

	t = clock();
	oil_scale_init(&os, image.height, out_width, image.width, out_height, cs);
	for(i=0; i<out_height; i++) {
		for (j=oil_scale_slots(&os); j>0; j--) {
			image.buffer += in_row_stride;
			oil_scale_in(&os, image.buffer);
		}
		oil_scale_out(&os, outbuf);
	}
	t = clock() - t;
	free(outbuf);
	oil_scale_free(&os);
	return t;
}

void do_bench(struct bench_image image, double ratio)
{
	int i, out_width, out_height;
	clock_t t_min, t_tmp;

	out_width = round(image.width * ratio);
	out_height = 500000; /* reasonable maximum height */

	oil_fix_ratio(image.width, image.height, &out_width, &out_height);

	t_min = 0;
	for (i=0; i<260; i++) {
		t_tmp = resize(image, out_width, out_height);
		if (!t_min || t_tmp < t_min) {
			t_min = t_tmp;
		}
	}

	printf("    to %4dx%4d %6dus\n", out_width, out_height, time_to_us(t_min));
}

void do_bench_sizes(char *name, char *path, enum oil_colorspace cs)
{
	struct bench_image image;

	image = png(path, cs);

	printf("Scale %dx%d %s image\n", image.width, image.height, name);

	do_bench(image, 0.01);
	do_bench(image, 0.125);
	do_bench(image, 0.8);
	do_bench(image, 2.14);

	free(image.buffer);
}

int main(int argc, char *argv[])
{
	clock_t t;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <path>\n", argv[0]);
		return 1;
	}

	t = clock();
	oil_global_init();
	t = clock() - t;
	printf("global init: %6dus\n", time_to_us(t));

	do_bench_sizes("G", argv[1], OIL_CS_G);
	do_bench_sizes("GA", argv[1], OIL_CS_GA);
	do_bench_sizes("RGB", argv[1], OIL_CS_RGB);
	do_bench_sizes("RGBX", argv[1], OIL_CS_RGBX);
	do_bench_sizes("RGBA", argv[1], OIL_CS_RGBA);
	do_bench_sizes("CMYK", argv[1], OIL_CS_CMYK);

	return 0;
}
