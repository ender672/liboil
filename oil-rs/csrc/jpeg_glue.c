#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

/* Opaque handle for JPEG decompression (reading). */
typedef struct {
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
} oil_jpeg_reader;

/* Opaque handle for JPEG compression (writing). */
typedef struct {
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	unsigned char *outbuf;
	unsigned long outsize;
} oil_jpeg_writer;

oil_jpeg_reader *oil_jpeg_reader_create(const unsigned char *data,
	unsigned long size)
{
	oil_jpeg_reader *r = calloc(1, sizeof(oil_jpeg_reader));
	if (!r) {
		return NULL;
	}
	r->dinfo.err = jpeg_std_error(&r->jerr);
	jpeg_create_decompress(&r->dinfo);
	jpeg_mem_src(&r->dinfo, data, size);
	jpeg_read_header(&r->dinfo, TRUE);
	r->dinfo.out_color_space = JCS_RGB;
	jpeg_start_decompress(&r->dinfo);
	return r;
}

unsigned int oil_jpeg_reader_width(const oil_jpeg_reader *r)
{
	return r->dinfo.output_width;
}

unsigned int oil_jpeg_reader_height(const oil_jpeg_reader *r)
{
	return r->dinfo.output_height;
}

int oil_jpeg_reader_components(const oil_jpeg_reader *r)
{
	return r->dinfo.output_components;
}

void oil_jpeg_reader_read_scanline(oil_jpeg_reader *r, unsigned char *buf)
{
	jpeg_read_scanlines(&r->dinfo, &buf, 1);
}

void oil_jpeg_reader_destroy(oil_jpeg_reader *r)
{
	if (!r) {
		return;
	}
	jpeg_finish_decompress(&r->dinfo);
	jpeg_destroy_decompress(&r->dinfo);
	free(r);
}

oil_jpeg_writer *oil_jpeg_writer_create(unsigned int width,
	unsigned int height, int components, int quality)
{
	oil_jpeg_writer *w = calloc(1, sizeof(oil_jpeg_writer));
	if (!w) {
		return NULL;
	}
	w->cinfo.err = jpeg_std_error(&w->jerr);
	jpeg_create_compress(&w->cinfo);
	jpeg_mem_dest(&w->cinfo, &w->outbuf, &w->outsize);
	w->cinfo.image_width = width;
	w->cinfo.image_height = height;
	w->cinfo.input_components = components;
	w->cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&w->cinfo);
	jpeg_set_quality(&w->cinfo, quality, FALSE);
	jpeg_start_compress(&w->cinfo, TRUE);
	return w;
}

void oil_jpeg_writer_write_scanline(oil_jpeg_writer *w, unsigned char *buf)
{
	jpeg_write_scanlines(&w->cinfo, &buf, 1);
}

unsigned char *oil_jpeg_writer_finish(oil_jpeg_writer *w, unsigned long *size)
{
	unsigned char *buf;

	jpeg_finish_compress(&w->cinfo);
	*size = w->outsize;
	buf = w->outbuf;
	w->outbuf = NULL;
	jpeg_destroy_compress(&w->cinfo);
	free(w);
	return buf;
}

void oil_jpeg_free_buf(unsigned char *buf)
{
	free(buf);
}
