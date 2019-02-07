#include "oil_resample.h"
#include <stdlib.h>
#include <png.h>

static enum oil_colorspace png_cs_to_oil(png_byte cs)
{
	switch(cs) {
	case PNG_COLOR_TYPE_GRAY:
		return OIL_CS_G;
	case PNG_COLOR_TYPE_GA:
		return OIL_CS_GA;
	case PNG_COLOR_TYPE_RGB:
		return OIL_CS_RGB;
	case PNG_COLOR_TYPE_RGBA:
		return OIL_CS_RGBA;
	}
	exit(99);
}

/** Interlaced PNGs need to be fully decompressed before we can scale the image.
  */
static void png_interlaced(png_structp rpng, png_infop rinfo, png_structp wpng,
	png_infop winfo)
{
	unsigned char **sl, *outbuf;
	int i, j, n, in_width, in_height, out_width, out_height, buf_len, ret;
	struct oil_scale ys;
	enum oil_colorspace cs;

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);
	out_width = png_get_image_width(wpng, winfo);
	out_height = png_get_image_height(wpng, winfo);

	/* Allocate space for the full input image and fill it. */
	sl = malloc(in_height * sizeof(unsigned char *));
	if (!sl) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}
	buf_len = png_get_rowbytes(rpng, rinfo);
	for (i=0; i<in_height; i++) {
		sl[i] = malloc(buf_len);
		if (!sl[i]) {
			fprintf(stderr, "Unable to allocate buffers.\n");
			exit(1);
		}
	}
	png_read_image(rpng, sl);

	cs = png_cs_to_oil(png_get_color_type(rpng, rinfo));
	outbuf = malloc(out_width * OIL_CMP(cs));

	ret = oil_scale_init(&ys, in_height, out_height, in_width, out_width,
		cs);
	if (ret!=0) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	n = 0;
	for(i=0; i<out_height; i++) {
		for (j=oil_scale_slots(&ys); j>0; j--) {
			oil_scale_in(&ys, sl[n++]);
		}
		oil_scale_out(&ys, outbuf);
		png_write_row(wpng, outbuf);
	}

	for (i=0; i<in_height; i++) {
		free(sl[i]);
	}
	free(sl);
	free(outbuf);
	oil_scale_free(&ys);
}

static void png_noninterlaced(png_structp rpng, png_infop rinfo,
	png_structp wpng, png_infop winfo)
{
	int i, j, in_width, in_height, out_width, out_height, ret;
	unsigned char *inbuf, *outbuf;
	struct oil_scale ys;
	enum oil_colorspace cs;

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);
	out_width = png_get_image_width(wpng, winfo);
	out_height = png_get_image_height(wpng, winfo);

	cs = png_cs_to_oil(png_get_color_type(rpng, rinfo));

	inbuf = malloc(in_width * OIL_CMP(cs));
	if (!inbuf) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}
	outbuf = malloc(out_width * OIL_CMP(cs));
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	ret = oil_scale_init(&ys, in_height, out_height, in_width, out_width,
		cs);
	if (ret!=0) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}
	for(i=0; i<out_height; i++) {
		for (j=oil_scale_slots(&ys); j>0; j--) {
			png_read_row(rpng, inbuf, NULL);
			oil_scale_in(&ys, inbuf);
		}
		oil_scale_out(&ys, outbuf);
		png_write_row(wpng, outbuf);
	}

	free(outbuf);
	free(inbuf);
	oil_scale_free(&ys);
}

static void png(FILE *input, FILE *output, int width, int height)
{
	png_structp rpng, wpng;
	png_infop rinfo, winfo;
	png_uint_32 in_width, in_height;
	png_byte ctype;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (setjmp(png_jmpbuf(rpng))) {
		fprintf(stderr, "PNG Decoding Error.\n");
		exit(1);
	}

	rinfo = png_create_info_struct(rpng);
	png_init_io(rpng, input);
	png_read_info(rpng, rinfo);

	png_set_packing(rpng);
	png_set_strip_16(rpng);
	png_set_expand(rpng);
	png_set_interlace_handling(rpng);
	png_read_update_info(rpng, rinfo);

	ctype = png_get_color_type(rpng, rinfo);

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);
	oil_fix_ratio(in_width, in_height, &width, &height);

	wpng = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	winfo = png_create_info_struct(wpng);
	png_init_io(wpng, output);

	png_set_IHDR(wpng, winfo, width, height, 8, ctype, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_write_info(wpng, winfo);

	switch (png_get_interlace_type(rpng, rinfo)) {
	case PNG_INTERLACE_NONE:
		png_noninterlaced(rpng, rinfo, wpng, winfo);
		break;
	case PNG_INTERLACE_ADAM7:
		png_interlaced(rpng, rinfo, wpng, winfo);
		break;
	}

	png_write_end(wpng, winfo);
	png_destroy_write_struct(&wpng, &winfo);
	png_destroy_read_struct(&rpng, &rinfo, NULL);
}

int main(int argc, char *argv[])
{
	int width, height;
	char *end;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s WIDTH HEIGHT < in.png > scale.png\n", argv[0]);
		return 1;
	}

	width = strtoul(argv[1], &end, 10);
	if (*end) {
		fprintf(stderr, "Error: Invalid width.\n");
		return 1;
	}

	height = strtoul(argv[2], &end, 10);
	if (*end) {
		fprintf(stderr, "Error: Invalid height.\n");
		return 1;
	}

	png(stdin, stdout, width, height);

	fclose(stdin);
	return 0;
}
