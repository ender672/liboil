#include "oil_libjpeg.h"
#include "oil_libpng.h"
#include <stdlib.h>
#include <jpeglib.h>
#include <png.h>

static void png(FILE *input, FILE *output, int width, int height)
{
	int i, in_width, in_height, ret;
	png_structp rpng, wpng;
	png_infop rinfo, winfo;
	png_byte ctype;
	struct oil_libpng ol;
	unsigned char *outbuf;

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

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);
	oil_fix_ratio(in_width, in_height, &width, &height);

	wpng = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	winfo = png_create_info_struct(wpng);
	png_init_io(wpng, output);

	ret = oil_libpng_init(&ol, rpng, rinfo, width, height);
	if (ret!=0) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	ctype = png_get_color_type(rpng, rinfo);
	png_set_IHDR(wpng, winfo, width, height, 8, ctype, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_write_info(wpng, winfo);

	outbuf = malloc(width * OIL_CMP(ol.os.cs));
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	for(i=0; i<height; i++) {
		oil_libpng_read_scanline(&ol, outbuf);
		png_write_row(wpng, outbuf);
	}

	png_write_end(wpng, winfo);
	png_destroy_write_struct(&wpng, &winfo);
	png_destroy_read_struct(&rpng, &rinfo, NULL);

	free(outbuf);
	oil_libpng_free(&ol);
}

static void prepare_jpeg_decompress(FILE *input,
	struct jpeg_decompress_struct *dinfo, struct jpeg_error_mgr *jerr)
{
	long i;

	dinfo->err = jpeg_std_error(jerr);
	jpeg_create_decompress(dinfo);
	jpeg_stdio_src(dinfo, input);

	/* Save custom headers for the compressor, but ignore APP0 & APP14 so
	 * libjpeg can handle them.
	 */
	jpeg_save_markers(dinfo, JPEG_COM, 0xFFFF);
	for (i=1; i<14; i++) {
		jpeg_save_markers(dinfo, JPEG_APP0+i, 0xFFFF);
	}
	jpeg_save_markers(dinfo, JPEG_APP0+15, 0xFFFF);
	jpeg_read_header(dinfo, TRUE);

	/* For testing purposes, you can run with the environment variable
	 * OILRGBX in order to trigger libjpeg turbo's JCS_RGBX color space.
	 */
#ifdef JCS_EXTENSIONS
	if (getenv("OILRGBX") != NULL && dinfo->out_color_space == JCS_RGB) {
		dinfo->out_color_space = JCS_EXT_RGBX;
		jpeg_calc_output_dimensions(dinfo);
	}
#endif

	jpeg_start_decompress(dinfo);
}

static void jpeg(FILE *input, FILE *output, int width_out, int height_out)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	unsigned char *outbuf;
	int i, ret;
	struct oil_libjpeg ol;
	jpeg_saved_marker_ptr marker;

	prepare_jpeg_decompress(input, &dinfo, &jerr);

	/* Use the image dimensions read from the header to calculate our final
	 * output dimensions.
	 */
	oil_fix_ratio(dinfo.output_width, dinfo.output_height, &width_out, &height_out);

	/* set up scaler */
	ret = oil_libjpeg_init(&ol, &dinfo, width_out, height_out);
	if (ret!=0) {
		fprintf(stderr, "Unable to initialize scaler.");
		jpeg_destroy_decompress(&dinfo);
		fclose(input);
		fclose(output);
		exit(1);
	}

	/* Allocate linear converter output buffer */
	outbuf = malloc(width_out * OIL_CMP(ol.os.cs) * sizeof(unsigned char));
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate buffers.");
		exit(1);
	}

	/* Jpeg compressor. */
	cinfo.err = &jerr;
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, output);
	cinfo.image_width = width_out;
	cinfo.image_height = height_out;
	cinfo.input_components = OIL_CMP(ol.os.cs);
	cinfo.in_color_space = oil_cs_to_jpeg(ol.os.cs);
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 94, FALSE);
	jpeg_start_compress(&cinfo, TRUE);

	/* Copy custom headers from source jpeg to dest jpeg. */
	for (marker=dinfo.marker_list; marker; marker=marker->next) {
		jpeg_write_marker(&cinfo, marker->marker, marker->data,
			marker->data_length);
	}

	/* Read scanlines, process image, and write scanlines to the jpeg
	 * encoder.
	 */
	for(i=height_out; i>0; i--) {
		oil_libjpeg_read_scanline(&ol, outbuf);
		jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	jpeg_finish_decompress(&dinfo);
	jpeg_destroy_decompress(&dinfo);
	free(outbuf);
	oil_libjpeg_free(&ol);
}

int looks_like_png(FILE *io)
{
	int peek;
	peek = getc(io);
	ungetc(peek, io);
	return peek == 137;
}

int main(int argc, char *argv[])
{
	int width, height;
	char *end;
	FILE *io_in, *io_out;

	io_in = stdin;
	io_out = stdout;

	if (argc < 3) {
		fprintf(stderr, "Usage: %s WIDTH HEIGHT [file] [file]\n", argv[0]);
		return 1;
	}

	if (argc > 3) {
		io_in = fopen(argv[3], "r");
		if (!io_in) {
			fprintf(stderr, "Unable to open source file.\n");
			return 1;
		}
	}

	if (argc > 4) {
		io_out = fopen(argv[4], "w");
		if (!io_out) {
			fprintf(stderr, "Unable to open destination file.\n");
			return 1;
		}
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

	if (looks_like_png(io_in)) {
		png(io_in, io_out, width, height);
	} else {
		jpeg(io_in, io_out, width, height);
	}

	fclose(io_in);
	fclose(io_out);
	return 0;
}
