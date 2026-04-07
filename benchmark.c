#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <png.h>
#include "oil_resample.h"
#include "oil_libpng.h"

typedef int (*scale_in_fn)(struct oil_scale *, unsigned char *);
typedef int (*scale_out_fn)(struct oil_scale *, unsigned char *);

struct bench_image {
	unsigned char *buffer;
	int width;
	int height;
	enum oil_colorspace cs;
};

static struct bench_image load_png(char *path, enum oil_colorspace cs)
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

	input = fopen(path, "rb");
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
	case OIL_CS_RGBX:
	case OIL_CS_RGBX_NOGAMMA:
		png_set_strip_alpha(rpng);
		png_set_filler(rpng, 0xffff, PNG_FILLER_AFTER);
		break;
	case OIL_CS_RGB_NOGAMMA:
		png_set_strip_alpha(rpng);
		break;
	case OIL_CS_CMYK: /* Kind of cheating on CMYK by giving it RGBA */
	case OIL_CS_RGBA:
	case OIL_CS_RGBA_NOGAMMA:
	case OIL_CS_ARGB:
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
	return (double)t * 1000.0 / CLOCKS_PER_SEC;
}

clock_t resize(struct bench_image image, int out_width, int out_height,
	scale_in_fn do_in, scale_out_fn do_out)
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
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate output buffer.\n");
		exit(1);
	}

	t = clock();
	oil_scale_init(&os, image.height, out_height, image.width, out_width, cs);
	for(i=0; i<out_height; i++) {
		while (oil_scale_slots(&os)) {
			do_in(&os, inbuf);
			inbuf += in_row_stride;
		}
		do_out(&os, outbuf);
	}
	t = clock() - t;
	free(outbuf);
	oil_scale_free(&os);
	return t;
}

void do_bench(struct bench_image image, double ratio, int iterations,
	scale_in_fn do_in, scale_out_fn do_out)
{
	int i, out_width, out_height;
	clock_t t_min, t_tmp;

	out_width = round(image.width * ratio);
	out_height = 500000; /* reasonable maximum height */

	oil_fix_ratio(image.width, image.height, &out_width, &out_height);

	t_min = 0;
	for (i=0; i<iterations; i++) {
		t_tmp = resize(image, out_width, out_height, do_in, do_out);
		if (!t_min || t_tmp < t_min) {
			t_min = t_tmp;
		}
	}

	printf("    to %4dx%4d %6.2fms\n", out_width, out_height, time_to_ms(t_min));
}

/* filter: 0=all, 1=downscale only (ratio<1), 2=upscale only (ratio>=1) */
void do_bench_sizes(char *name, char *path, enum oil_colorspace cs,
	int iterations, int filter, char *impl_name,
	scale_in_fn do_in, scale_out_fn do_out)
{
	struct bench_image image;
	double ratios[] = { 0.01, 0.125, 0.8, 2.14 };
	size_t i, num_ratios;

	image = load_png(path, cs);

	printf("%dx%d %s [%s]\n", image.width, image.height, name, impl_name);

	num_ratios = sizeof(ratios)/sizeof(ratios[0]);
	for (i=0; i<num_ratios; i++) {
		if (filter == 1 && ratios[i] >= 1.0) continue;
		if (filter == 2 && ratios[i] < 1.0) continue;
		do_bench(image, ratios[i], iterations, do_in, do_out);
	}

	free(image.buffer);
}

struct impl {
	char *name;
	scale_in_fn in;
	scale_out_fn out;
};

void run_bench(char *path, char *cs_arg, int iterations, int filter,
	struct impl *impls, int num_impls)
{
	size_t i, j, num_spaces;
	clock_t t;

	enum oil_colorspace spaces[] = {
		OIL_CS_G,
		OIL_CS_GA,
		OIL_CS_RGB,
		OIL_CS_RGBX,
		OIL_CS_RGBA,
		OIL_CS_CMYK,
		OIL_CS_RGB_NOGAMMA,
		OIL_CS_RGBA_NOGAMMA,
		OIL_CS_RGBX_NOGAMMA,
	};

	char *space_names[] = {
		"G",
		"GA",
		"RGB",
		"RGBX",
		"RGBA",
		"CMYK",
		"RGB_NOGAMMA",
		"RGBA_NOGAMMA",
		"RGBX_NOGAMMA",
	};

	t = clock();
	oil_global_init();
	t = clock() - t;
	printf("global init: %6.2fms\n", time_to_ms(t));

	num_spaces = sizeof(spaces)/sizeof(spaces[0]);

	if (cs_arg) {
		for (i=0; i<num_spaces; i++) {
			if (strcmp(space_names[i], cs_arg) == 0)
				break;
		}
		if (i >= num_spaces) {
			fprintf(stderr, "Colorspace not recognized.\n");
			exit(1);
		}
		/* single colorspace */
		for (j=0; j<(size_t)num_impls; j++) {
			do_bench_sizes(space_names[i], path, spaces[i],
				iterations, filter, impls[j].name,
				impls[j].in, impls[j].out);
		}
		return;
	}

	for (i=0; i<num_spaces; i++) {
		for (j=0; j<(size_t)num_impls; j++) {
			do_bench_sizes(space_names[i], path, spaces[i],
				iterations, filter, impls[j].name,
				impls[j].in, impls[j].out);
		}
	}
}

int main(int argc, char *argv[])
{
	int iterations, filter, arg_pos, impl_mode;
	char *end, *path, *cs_arg;
	unsigned long ul;
	struct impl impls[2];
	int num_impls;

	/* Parse flags */
	filter = 0;
	impl_mode = 0; /* 0=both, 1=scalar, 2=simd */
	arg_pos = 1;
	while (arg_pos < argc && argv[arg_pos][0] == '-') {
		if (strcmp(argv[arg_pos], "--down") == 0) {
			filter = 1;
		} else if (strcmp(argv[arg_pos], "--up") == 0) {
			filter = 2;
		} else if (strcmp(argv[arg_pos], "--scalar") == 0) {
			impl_mode = 1;
		} else if (strcmp(argv[arg_pos], "--simd") == 0) {
			impl_mode = 2;
		} else {
			fprintf(stderr, "Unknown option: %s\n", argv[arg_pos]);
			return 1;
		}
		arg_pos++;
	}

	if (argc - arg_pos < 1 || argc - arg_pos > 2) {
		fprintf(stderr, "Usage: %s [--up|--down] [--scalar|--simd] <path> [colorspace]\n",
			argv[0]);
		return 1;
	}

	path = argv[arg_pos];
	cs_arg = (argc - arg_pos == 2) ? argv[arg_pos + 1] : NULL;

	iterations = 100;
	if (getenv("OILITERATIONS")) {
		errno = 0;
		ul = strtoul(getenv("OILITERATIONS"), &end, 10);
		if (*end != '\0' || errno != 0 || ul == 0 || ul > INT_MAX) {
			fprintf(stderr, "Invalid environment variable OILITERATIONS.");
			return 1;
		}
		iterations = (int)ul;
	}
	fprintf(stderr, "Iterations: %d\n", iterations);

	num_impls = 0;

	if (impl_mode != 2) {
		impls[num_impls].name = "scalar";
		impls[num_impls].in = oil_scale_in;
		impls[num_impls].out = oil_scale_out;
		num_impls++;
	}

#if defined(__x86_64__)
	if (impl_mode != 1) {
		impls[num_impls].name = "sse2";
		impls[num_impls].in = oil_scale_in_sse2;
		impls[num_impls].out = oil_scale_out_sse2;
		num_impls++;
	}
#elif defined(__aarch64__)
	if (impl_mode != 1) {
		impls[num_impls].name = "neon";
		impls[num_impls].in = oil_scale_in_neon;
		impls[num_impls].out = oil_scale_out_neon;
		num_impls++;
	}
#else
	if (impl_mode == 2) {
		fprintf(stderr, "No SIMD support compiled in.\n");
		return 1;
	}
#endif

	run_bench(path, cs_arg, iterations, filter, impls, num_impls);
	return 0;
}
