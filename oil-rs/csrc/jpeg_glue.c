#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

/* Opaque handle for JPEG decompression (reading). */
typedef struct {
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
	FILE *fp; /* non-NULL when created from a file path */
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
	switch (r->dinfo.jpeg_color_space) {
	case JCS_GRAYSCALE:
		r->dinfo.out_color_space = JCS_GRAYSCALE;
		break;
	case JCS_CMYK:
	case JCS_YCCK:
		r->dinfo.out_color_space = JCS_CMYK;
		break;
	default:
		r->dinfo.out_color_space = JCS_RGB;
		break;
	}
	jpeg_start_decompress(&r->dinfo);
	return r;
}

oil_jpeg_reader *oil_jpeg_reader_create_file(const char *path)
{
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		return NULL;
	}
	oil_jpeg_reader *r = calloc(1, sizeof(oil_jpeg_reader));
	if (!r) {
		fclose(fp);
		return NULL;
	}
	r->fp = fp;
	r->dinfo.err = jpeg_std_error(&r->jerr);
	jpeg_create_decompress(&r->dinfo);
	jpeg_stdio_src(&r->dinfo, fp);
	jpeg_read_header(&r->dinfo, TRUE);
	switch (r->dinfo.jpeg_color_space) {
	case JCS_GRAYSCALE:
		r->dinfo.out_color_space = JCS_GRAYSCALE;
		break;
	case JCS_CMYK:
	case JCS_YCCK:
		r->dinfo.out_color_space = JCS_CMYK;
		break;
	default:
		r->dinfo.out_color_space = JCS_RGB;
		break;
	}
	jpeg_start_decompress(&r->dinfo);
	return r;
}

int oil_jpeg_dimensions_file(const char *path, unsigned int *width,
	unsigned int *height)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		return -1;
	}
	dinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&dinfo);
	jpeg_stdio_src(&dinfo, fp);
	jpeg_read_header(&dinfo, TRUE);
	*width = dinfo.image_width;
	*height = dinfo.image_height;
	jpeg_destroy_decompress(&dinfo);
	fclose(fp);
	return 0;
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

int oil_jpeg_reader_color_space(const oil_jpeg_reader *r)
{
	return (int)r->dinfo.out_color_space;
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
	if (r->fp) {
		fclose(r->fp);
	}
	free(r);
}

oil_jpeg_writer *oil_jpeg_writer_create(unsigned int width,
	unsigned int height, int components, int color_space, int quality)
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
	w->cinfo.in_color_space = (J_COLOR_SPACE)color_space;
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
