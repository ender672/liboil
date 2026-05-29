/*
 * Demonstrates the split decode/feed API on oil_libpng.
 *
 * Encodes a deterministic gradient to an in-memory PNG, then decodes the
 * same image two ways:
 *
 *   1. The bundled all-in-one path, oil_libpng_read_scanline.
 *   2. The new split path: oil_libpng_decode_row into a caller-owned
 *      buffer, then oil_scale_in / oil_scale_out driven by the caller.
 *
 * Asserts the two outputs are byte-identical across RGB / RGBA and
 * interlaced / non-interlaced inputs. Also exercises the cs_override
 * arg by repeating one case in OIL_CS_RGB_NOGAMMA.
 */

#include <assert.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include "oil_resample.h"
#include "oil_libpng.h"

#define IN_W 192
#define IN_H 144

struct mem_io {
	unsigned char *buf;
	size_t size;
	size_t cap;
	size_t pos;
};

static void mem_write(png_structp png, png_bytep data, size_t len)
{
	struct mem_io *m = png_get_io_ptr(png);
	if (m->size + len > m->cap) {
		size_t new_cap = (m->size + len) * 2;
		unsigned char *grown = realloc(m->buf, new_cap);
		assert(grown);
		m->buf = grown;
		m->cap = new_cap;
	}
	memcpy(m->buf + m->size, data, len);
	m->size += len;
}

static void mem_flush(png_structp png) { (void)png; }

static void mem_read(png_structp png, png_bytep data, size_t len)
{
	struct mem_io *m = png_get_io_ptr(png);
	assert(m->pos + len <= m->size);
	memcpy(data, m->buf + m->pos, len);
	m->pos += len;
}

static void encode_gradient_png(struct mem_io *out, int channels, int interlaced)
{
	png_structp wpng;
	png_infop winfo;
	unsigned char **rows;
	int x, y;
	int color_type = (channels == 4) ?
		PNG_COLOR_TYPE_RGBA : PNG_COLOR_TYPE_RGB;
	int interlace = interlaced ?
		PNG_INTERLACE_ADAM7 : PNG_INTERLACE_NONE;

	out->buf = NULL;
	out->size = out->cap = out->pos = 0;

	wpng = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(wpng);
	winfo = png_create_info_struct(wpng);
	assert(winfo);
	if (setjmp(png_jmpbuf(wpng))) {
		assert(0 && "png encode failed");
	}
	png_set_write_fn(wpng, out, mem_write, mem_flush);
	png_set_IHDR(wpng, winfo, IN_W, IN_H, 8, color_type, interlace,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(wpng, winfo);

	rows = malloc((size_t)IN_H * sizeof(unsigned char *));
	for (y = 0; y < IN_H; y++) {
		rows[y] = malloc((size_t)IN_W * channels);
		for (x = 0; x < IN_W; x++) {
			rows[y][x*channels+0] = (unsigned char)x;
			rows[y][x*channels+1] = (unsigned char)y;
			rows[y][x*channels+2] = (unsigned char)((x + y) >> 1);
			if (channels == 4) {
				rows[y][x*channels+3] =
					(unsigned char)(255 - ((x + y) >> 1));
			}
		}
	}
	png_write_image(wpng, rows);
	png_write_end(wpng, NULL);

	for (y = 0; y < IN_H; y++) free(rows[y]);
	free(rows);
	png_destroy_write_struct(&wpng, &winfo);
}

/* Open the in-memory PNG and run png_read_update_info. */
static void open_png(png_structp *rpng_out, png_infop *rinfo_out,
	struct mem_io *src, struct mem_io *encoded)
{
	png_structp rpng;
	png_infop rinfo;

	src->buf = encoded->buf;
	src->size = encoded->size;
	src->cap = encoded->cap;
	src->pos = 0;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(rpng);
	rinfo = png_create_info_struct(rpng);
	assert(rinfo);
	if (setjmp(png_jmpbuf(rpng))) {
		assert(0 && "png decode failed");
	}
	png_set_read_fn(rpng, src, mem_read);
	png_read_info(rpng, rinfo);
	png_set_packing(rpng);
	png_set_strip_16(rpng);
	png_set_expand(rpng);
	png_set_interlace_handling(rpng);
	png_read_update_info(rpng, rinfo);

	*rpng_out = rpng;
	*rinfo_out = rinfo;
}

static unsigned char *decode_bundled(struct mem_io *encoded,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h,
	enum oil_colorspace cs_override, int *out_cmp)
{
	struct mem_io src;
	png_structp rpng;
	png_infop rinfo;
	struct oil_libpng ol;
	unsigned char *out;
	int y;

	open_png(&rpng, &rinfo, &src, encoded);
	assert(oil_libpng_init_ex(&ol, rpng, rinfo, out_w, out_h,
		src_x, src_y, src_w, src_h, cs_override) == 0);

	out = malloc((size_t)out_w * out_h * ol.components);
	for (y = 0; y < out_h; y++) {
		oil_libpng_read_scanline(&ol,
			out + (size_t)y * out_w * ol.components);
	}
	*out_cmp = ol.components;

	oil_libpng_free(&ol);
	png_destroy_read_struct(&rpng, &rinfo, NULL);
	return out;
}

static unsigned char *decode_split(struct mem_io *encoded,
	int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h,
	enum oil_colorspace cs_override, int *out_cmp)
{
	struct mem_io src;
	png_structp rpng;
	png_infop rinfo;
	struct oil_libpng ol;
	unsigned char *out, *row;
	int y;

	open_png(&rpng, &rinfo, &src, encoded);
	assert(oil_libpng_init_ex(&ol, rpng, rinfo, out_w, out_h,
		src_x, src_y, src_w, src_h, cs_override) == 0);

	row = malloc((size_t)ol.fed_width * ol.components);
	out = malloc((size_t)out_w * out_h * ol.components);
	for (y = 0; y < out_h; y++) {
		while (oil_scale_slots(&ol.os) > 0) {
			oil_libpng_decode_row(&ol, row);
			oil_scale_in(&ol.os, row);
		}
		oil_scale_out(&ol.os, out + (size_t)y * out_w * ol.components);
	}
	free(row);
	*out_cmp = ol.components;

	oil_libpng_free(&ol);
	png_destroy_read_struct(&rpng, &rinfo, NULL);
	return out;
}

static void check_case(struct mem_io *encoded,
	const char *label, int out_w, int out_h,
	double src_x, double src_y, double src_w, double src_h,
	enum oil_colorspace cs_override)
{
	unsigned char *a, *b;
	int cmp_a, cmp_b;
	size_t n;
	a = decode_bundled(encoded, out_w, out_h,
		src_x, src_y, src_w, src_h, cs_override, &cmp_a);
	b = decode_split(encoded, out_w, out_h,
		src_x, src_y, src_w, src_h, cs_override, &cmp_b);
	assert(cmp_a == cmp_b);
	n = (size_t)out_w * out_h * cmp_a;
	if (memcmp(a, b, n) != 0) {
		size_t i;
		for (i = 0; i < n; i++) {
			if (a[i] != b[i]) {
				fprintf(stderr, "  %s: mismatch at byte %zu: "
					"bundled=%u split=%u\n",
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

static void run_suite(struct mem_io *encoded, const char *prefix)
{
	char label[64];
	snprintf(label, sizeof(label), "%s full image",    prefix);
	check_case(encoded, label, 100, 75,
		0.0, 0.0, (double)IN_W, (double)IN_H, OIL_CS_UNKNOWN);
	snprintf(label, sizeof(label), "%s integer crop",  prefix);
	check_case(encoded, label,  64, 48,
		16.0, 24.0, 96.0, 72.0, OIL_CS_UNKNOWN);
	snprintf(label, sizeof(label), "%s sub-pixel crop",prefix);
	check_case(encoded, label, 100, 75,
		8.25, 4.5, 150.75, 110.25, OIL_CS_UNKNOWN);
	snprintf(label, sizeof(label), "%s upscale crop",  prefix);
	check_case(encoded, label, 200, 150,
		40.0, 30.0, 32.0, 24.0, OIL_CS_UNKNOWN);
}

int main(void)
{
	struct mem_io rgb_streaming  = {0};
	struct mem_io rgb_interlaced = {0};
	struct mem_io rgba_streaming = {0};

	encode_gradient_png(&rgb_streaming,  3, 0);
	encode_gradient_png(&rgb_interlaced, 3, 1);
	encode_gradient_png(&rgba_streaming, 4, 0);
	printf("encoded test PNGs: RGB streaming=%zu, RGB interlaced=%zu, "
		"RGBA streaming=%zu bytes (%dx%d gradient)\n",
		rgb_streaming.size, rgb_interlaced.size, rgba_streaming.size,
		IN_W, IN_H);

	run_suite(&rgb_streaming,  "RGB-stream:");
	run_suite(&rgb_interlaced, "RGB-adam7: ");
	run_suite(&rgba_streaming, "RGBA-strm: ");

	/* Exercise the cs_override path (file is RGB, scaler is asked to
	 * treat the bytes as already-linear). */
	check_case(&rgb_streaming, "RGB-stream: nogamma override", 100, 75,
		0.0, 0.0, (double)IN_W, (double)IN_H, OIL_CS_RGB_NOGAMMA);

	free(rgb_streaming.buf);
	free(rgb_interlaced.buf);
	free(rgba_streaming.buf);
	printf("All tests pass.\n");
	return 0;
}
