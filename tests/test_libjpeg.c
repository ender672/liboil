/*
 * Demonstrates the split decode/feed API on oil_libjpeg.
 *
 * Encodes a deterministic RGB gradient to an in-memory JPEG, then decodes
 * the same image two ways:
 *
 *   1. The bundled all-in-one path, oil_libjpeg_read_scanline.
 *   2. The new split path: oil_libjpeg_decode_row into a caller-owned
 *      buffer, then oil_scale_in / oil_scale_out driven by the caller.
 *
 * Asserts the two outputs are byte-identical. This is the property the
 * sdltest threaded pipeline (and any other caller that wants to interpose
 * a slot queue between decode and scale) relies on.
 */

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include "oil_resample.h"
#include "oil_libjpeg.h"

#define IN_W 256
#define IN_H 192

struct jerr {
	struct jpeg_error_mgr mgr;
	jmp_buf jmpbuf;
};

static void on_jpeg_err(j_common_ptr cinfo)
{
	struct jerr *e = (struct jerr *)cinfo->err;
	longjmp(e->jmpbuf, 1);
}

static void encode_gradient_jpeg(unsigned char **out, unsigned long *out_size)
{
	struct jpeg_compress_struct cinfo;
	struct jerr jerr;
	unsigned char *row;
	int x, y;

	cinfo.err = jpeg_std_error(&jerr.mgr);
	jerr.mgr.error_exit = on_jpeg_err;
	if (setjmp(jerr.jmpbuf)) {
		assert(0 && "jpeg encode failed");
	}
	jpeg_create_compress(&cinfo);

	*out = NULL;
	*out_size = 0;
	jpeg_mem_dest(&cinfo, out, out_size);

	cinfo.image_width = IN_W;
	cinfo.image_height = IN_H;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 90, TRUE);
	jpeg_start_compress(&cinfo, TRUE);

	row = malloc((size_t)IN_W * 3);
	while (cinfo.next_scanline < cinfo.image_height) {
		y = cinfo.next_scanline;
		for (x = 0; x < IN_W; x++) {
			row[x*3+0] = (unsigned char)x;
			row[x*3+1] = (unsigned char)y;
			row[x*3+2] = (unsigned char)((x + y) >> 1);
		}
		jpeg_write_scanlines(&cinfo, &row, 1);
	}
	free(row);

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}

/* Open the in-memory JPEG and run jpeg_start_decompress. */
static void open_jpeg(struct jpeg_decompress_struct *dinfo, struct jerr *jerr,
	unsigned char *jpg, unsigned long jpg_size)
{
	dinfo->err = jpeg_std_error(&jerr->mgr);
	jerr->mgr.error_exit = on_jpeg_err;
	if (setjmp(jerr->jmpbuf)) {
		assert(0 && "jpeg decode failed");
	}
	jpeg_create_decompress(dinfo);
	jpeg_mem_src(dinfo, jpg, jpg_size);
	jpeg_read_header(dinfo, TRUE);
	jpeg_calc_output_dimensions(dinfo);
	jpeg_start_decompress(dinfo);
}

static unsigned char *decode_bundled(unsigned char *jpg, unsigned long jpg_size,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	struct jpeg_decompress_struct dinfo;
	struct jerr jerr;
	struct oil_libjpeg ol;
	unsigned char *out;
	int y;

	open_jpeg(&dinfo, &jerr, jpg, jpg_size);
	assert(oil_libjpeg_init_ex(&ol, &dinfo, out_w, out_h,
		src_x, src_y, src_w, src_h, OIL_CS_UNKNOWN) == 0);

	out = malloc((size_t)out_w * out_h * ol.components);
	for (y = 0; y < out_h; y++) {
		oil_libjpeg_read_scanline(&ol,
			out + (size_t)y * out_w * ol.components);
	}

	oil_libjpeg_free(&ol);
	jpeg_destroy_decompress(&dinfo);
	return out;
}

static unsigned char *decode_split(unsigned char *jpg, unsigned long jpg_size,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	struct jpeg_decompress_struct dinfo;
	struct jerr jerr;
	struct oil_libjpeg ol;
	unsigned char *out, *row;
	int y;

	open_jpeg(&dinfo, &jerr, jpg, jpg_size);
	assert(oil_libjpeg_init_ex(&ol, &dinfo, out_w, out_h,
		src_x, src_y, src_w, src_h, OIL_CS_UNKNOWN) == 0);

	row = malloc((size_t)ol.fed_width * ol.components);
	out = malloc((size_t)out_w * out_h * ol.components);
	for (y = 0; y < out_h; y++) {
		while (oil_scale_slots(&ol.os) > 0) {
			oil_libjpeg_decode_row(&ol, row);
			oil_scale_in(&ol.os, row);
		}
		oil_scale_out(&ol.os, out + (size_t)y * out_w * ol.components);
	}
	free(row);

	oil_libjpeg_free(&ol);
	jpeg_destroy_decompress(&dinfo);
	return out;
}

static void check_case(unsigned char *jpg, unsigned long jpg_size,
	const char *label, int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h)
{
	unsigned char *a, *b;
	size_t n = (size_t)out_w * out_h * 3;
	a = decode_bundled(jpg, jpg_size, out_w, out_h, src_x, src_y, src_w, src_h);
	b = decode_split(jpg, jpg_size, out_w, out_h, src_x, src_y, src_w, src_h);
	if (memcmp(a, b, n) != 0) {
		size_t i;
		for (i = 0; i < n; i++) {
			if (a[i] != b[i]) {
				fprintf(stderr, "  %s: mismatch at byte %zu: bundled=%u split=%u\n",
					label, i, a[i], b[i]);
				break;
			}
		}
		assert(0);
	}
	printf("  %s: %dx%d ok\n", label, out_w, out_h);
	free(a);
	free(b);
}

int main(void)
{
	unsigned char *jpg;
	unsigned long jpg_size;

	encode_gradient_jpeg(&jpg, &jpg_size);
	printf("encoded test JPEG: %lu bytes (%dx%d gradient)\n",
		jpg_size, IN_W, IN_H);

	/* Full image, simple downscale. */
	check_case(jpg, jpg_size, "full image",       100,  75,
		0.0, 0.0, (double)IN_W, (double)IN_H);

	/* Integer-aligned crop. */
	check_case(jpg, jpg_size, "integer crop",      80,  60,
		16.0, 32.0, 128.0, 96.0);

	/* Sub-pixel src rect. */
	check_case(jpg, jpg_size, "sub-pixel crop",   100,  75,
		10.25, 5.5, 200.75, 150.25);

	/* Tiny output. */
	check_case(jpg, jpg_size, "tiny output",        4,   3,
		0.0, 0.0, (double)IN_W, (double)IN_H);

	/* Upscale a small region. */
	check_case(jpg, jpg_size, "upscale crop",     200, 150,
		50.0, 40.0, 32.0, 24.0);

	free(jpg);
	printf("All tests pass.\n");
	return 0;
}
