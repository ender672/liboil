/* SPDX-License-Identifier: MIT */

#include "oil_libpng.h"
#include <stdlib.h>
#include <string.h>

static unsigned char **alloc_full_image_buf(int height, int rowbytes)
{
	int i, j;
	unsigned char **imgbuf;

	imgbuf = malloc(height * sizeof(unsigned char *));
	if (!imgbuf) {
		return NULL;
	}
	for (i=0; i<height; i++) {
		imgbuf[i] = malloc(rowbytes);
		if (!imgbuf[i]) {
			for (j=0; j<i; j++) {
				free(imgbuf[j]);
			}
			free(imgbuf);
			return NULL;
		}
	}
	return imgbuf;
}

static void free_full_image_buf(unsigned char **imgbuf, int height)
{
	int i;
	for (i=0; i<height; i++) {
		free(imgbuf[i]);
	}
	free(imgbuf);
}

int oil_libpng_init(struct oil_libpng *ol, png_structp rpng, png_infop rinfo,
	int out_width, int out_height)
{
	return oil_libpng_init_ex(ol, rpng, rinfo, out_width, out_height,
		0.0, 0.0,
		(double)png_get_image_width(rpng, rinfo),
		(double)png_get_image_height(rpng, rinfo),
		OIL_CS_UNKNOWN);
}

int oil_libpng_init_ex(struct oil_libpng *ol, png_structp rpng, png_infop rinfo,
	int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override)
{
	int ret, in_width, in_height, buf_len, cmp;
	int fed_x, fed_y, fed_w, fed_h;
	int i;
	enum oil_colorspace cs;

	ol->rpng = rpng;
	ol->rinfo = rinfo;
	ol->in_vpos = 0;
	ol->inbuf_offset = 0;
	ol->img_height = 0;
	ol->fed_width = 0;
	ol->components = 0;
	ol->inbuf = NULL;
	ol->inimage = NULL;

	cs = png_cs_to_oil(png_get_color_type(rpng, rinfo));
	if (cs == OIL_CS_UNKNOWN) {
		return -1;
	}
	cmp = OIL_CMP(cs);
	if (cs_override != OIL_CS_UNKNOWN) {
		if (OIL_CMP(cs_override) != cmp) {
			return -1;
		}
		cs = cs_override;
	}

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);
	ol->img_height = in_height;

	if (oil_required_input_rect(in_height, in_width,
		src_y, src_height, src_x, src_width,
		out_height, out_width,
		&fed_y, &fed_h, &fed_x, &fed_w) < 0) {
		return -1;
	}
	ol->fed_width = fed_w;
	ol->components = cmp;

	ret = oil_scale_init_ex(&ol->os, fed_h, out_height, fed_w, out_width,
		src_y - fed_y, src_height,
		src_x - fed_x, src_width,
		cs);
	if (ret != 0) {
		return ret;
	}

	ol->inbuf_offset = fed_x * cmp;
	buf_len = png_get_rowbytes(rpng, rinfo);
	switch (png_get_interlace_type(rpng, rinfo)) {
	case PNG_INTERLACE_NONE:
		ol->inbuf = malloc(buf_len);
		if (!ol->inbuf) {
			oil_scale_free(&ol->os);
			return -2;
		}
		for (i = 0; i < fed_y; i++) {
			png_read_row(rpng, ol->inbuf, NULL);
		}
		break;
	case PNG_INTERLACE_ADAM7:
		ol->inimage = alloc_full_image_buf(in_height, buf_len);
		if (!ol->inimage) {
			oil_scale_free(&ol->os);
			return -2;
		}
		png_read_image(rpng, ol->inimage);
		ol->in_vpos = fed_y;
		break;
	}

	return 0;
}

void oil_libpng_free(struct oil_libpng *ol)
{
	if (ol->inbuf) {
		free(ol->inbuf);
	}
	if (ol->inimage) {
		free_full_image_buf(ol->inimage, ol->img_height);
	}
	oil_scale_free(&ol->os);
}

static void read_scanline_interlaced(struct oil_libpng *ol)
{
	while (oil_scale_slots(&ol->os)) {
		oil_scale_in(&ol->os, ol->inimage[ol->in_vpos++] + ol->inbuf_offset);
	}
}

static void read_scanline(struct oil_libpng *ol)
{
	while (oil_scale_slots(&ol->os)) {
		png_read_row(ol->rpng, ol->inbuf, NULL);
		oil_scale_in(&ol->os, ol->inbuf + ol->inbuf_offset);
	}
}

void oil_libpng_decode_row(struct oil_libpng *ol, unsigned char *dst)
{
	size_t n = (size_t)ol->fed_width * ol->components;
	switch (png_get_interlace_type(ol->rpng, ol->rinfo)) {
	case PNG_INTERLACE_NONE:
		png_read_row(ol->rpng, ol->inbuf, NULL);
		memcpy(dst, ol->inbuf + ol->inbuf_offset, n);
		break;
	case PNG_INTERLACE_ADAM7:
		memcpy(dst, ol->inimage[ol->in_vpos++] + ol->inbuf_offset, n);
		break;
	}
}

void oil_libpng_read_scanline(struct oil_libpng *ol, unsigned char *outbuf)
{
	switch (png_get_interlace_type(ol->rpng, ol->rinfo)) {
	case PNG_INTERLACE_NONE:
		read_scanline(ol);
		break;
	case PNG_INTERLACE_ADAM7:
		read_scanline_interlaced(ol);
		break;
	}
	oil_scale_out(&ol->os, outbuf);
}

enum oil_colorspace png_cs_to_oil(png_byte cs)
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
	default:
		return OIL_CS_UNKNOWN;
	}
}
