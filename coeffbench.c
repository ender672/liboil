#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include "oil_resample.h"

static double time_to_us(clock_t t)
{
	return (double)t * 1000000.0 / CLOCKS_PER_SEC;
}

static void bench_one(int in_w, int in_h, int out_w, int out_h,
	enum oil_colorspace cs, int iterations)
{
	struct oil_scale os;
	void *buf;
	int alloc_size, i, ret;
	clock_t t, t_min;

	alloc_size = oil_scale_alloc_size(in_h, out_h, in_w, out_w, cs);
	buf = calloc(1, alloc_size);
	if (!buf) {
		fprintf(stderr, "Unable to allocate buffer.\n");
		exit(1);
	}

	/* Warm up once so any first-touch faults don't pollute timing. */
	ret = oil_scale_init_allocated(&os, in_h, out_h, in_w, out_w, cs, buf);
	if (ret) {
		fprintf(stderr, "oil_scale_init_allocated failed: %d\n", ret);
		exit(1);
	}

	t_min = 0;
	for (i=0; i<iterations; i++) {
		/* Reset buffer outside the timed region. Upscale borders use
		 * += and would accumulate across runs; downscale coeffs are
		 * scatter-written and need zeros at unwritten positions. */
		memset(buf, 0, alloc_size);
		t = clock();
		oil_scale_init_allocated(&os, in_h, out_h, in_w, out_w, cs, buf);
		t = clock() - t;
		if (!t_min || t < t_min) {
			t_min = t;
		}
	}

	printf("  %5dx%-5d -> %5dx%-5d  %9.2f us\n",
		in_w, in_h, out_w, out_h, time_to_us(t_min));

	free(buf);
}

struct case_t {
	int in_w, in_h, out_w, out_h;
};

static void run_section(const char *name, struct case_t *cases, int n,
	enum oil_colorspace cs, int iterations)
{
	int i;
	printf("%s\n", name);
	for (i=0; i<n; i++) {
		bench_one(cases[i].in_w, cases[i].in_h,
			cases[i].out_w, cases[i].out_h, cs, iterations);
	}
}

static void print_help(char *prog)
{
	printf("Usage: %s [options]\n", prog);
	printf("\n");
	printf("Benchmark oil coefficient calculation (scale_up_coeffs / scale_down_coeffs)\n");
	printf("by timing oil_scale_init_allocated() with a pre-allocated buffer.\n");
	printf("\n");
	printf("Options:\n");
	printf("  -h, --help    Show this help message and exit\n");
	printf("\n");
	printf("Environment:\n");
	printf("  OILITERATIONS    Number of iterations per case (default 2000)\n");
}

int main(int argc, char *argv[])
{
	int iterations, i;
	clock_t t;
	char *end;
	unsigned long ul;

	for (i=1; i<argc; i++) {
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			print_help(argv[0]);
			return 0;
		}
		fprintf(stderr, "Unknown option: %s\n", argv[i]);
		return 1;
	}

	iterations = 2000;
	if (getenv("OILITERATIONS")) {
		errno = 0;
		ul = strtoul(getenv("OILITERATIONS"), &end, 10);
		if (*end != '\0' || errno != 0 || ul == 0 || ul > INT_MAX) {
			fprintf(stderr, "Invalid environment variable OILITERATIONS.\n");
			return 1;
		}
		iterations = (int)ul;
	}
	fprintf(stderr, "Iterations: %d\n", iterations);

	/* Force global init outside the timed region. */
	t = clock();
	oil_global_init();
	t = clock() - t;
	printf("global init: %.2f us\n\n", time_to_us(t));

	struct case_t downscale[] = {
		{  640,  480,  320,  240 },
		{ 1920, 1080,  960,  540 },
		{ 1920, 1080,  240,  135 },
		{ 4096, 4096, 2048, 2048 },
		{ 4096, 4096,  512,  512 },
		{ 4096, 4096,   64,   64 },
		{ 8192, 8192,  100,  100 },
	};

	struct case_t upscale[] = {
		{  320,  240,  640,  480 },
		{  640,  480, 1920, 1080 },
		{  100,  100, 4096, 4096 },
		{ 1024, 1024, 2048, 2048 },
		{ 1024, 1024, 8192, 8192 },
	};

	struct case_t identity[] = {
		{  640,  480,  640,  480 },
		{ 1920, 1080, 1920, 1080 },
		{ 4096, 4096, 4096, 4096 },
	};

	run_section("Downscale", downscale,
		sizeof(downscale)/sizeof(downscale[0]),
		OIL_CS_RGBA, iterations);
	printf("\n");
	run_section("Upscale", upscale,
		sizeof(upscale)/sizeof(upscale[0]),
		OIL_CS_RGBA, iterations);
	printf("\n");
	run_section("Identity (downscale path)", identity,
		sizeof(identity)/sizeof(identity[0]),
		OIL_CS_RGBA, iterations);

	return 0;
}
