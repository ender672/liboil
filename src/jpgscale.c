#include "oil_resample.h"
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

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

	jpeg_start_decompress(dinfo);
}

static enum oil_colorspace jpeg_cs_to_oil(J_COLOR_SPACE cs)
{
	switch(cs) {
	case JCS_GRAYSCALE:
		return OIL_CS_G;
	case JCS_RGB:
		return OIL_CS_RGB;
	case JCS_CMYK:
		return OIL_CS_CMYK;
	default:
		fprintf(stderr, "Unknown jpeg color space: %d\n", cs);
		exit(1);
	}
}

static J_COLOR_SPACE oil_cs_to_jpeg(enum oil_colorspace cs)
{
	switch(cs) {
	case OIL_CS_G:
		return JCS_GRAYSCALE;
	case OIL_CS_RGB:
		return JCS_RGB;
	case OIL_CS_CMYK:
		return JCS_CMYK;
	default:
		fprintf(stderr, "Unknown oil color space: %d\n", cs);
		exit(1);
	}
}

static void jpeg(FILE *input, FILE *output, int width_out, int height_out)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	unsigned char *inbuf, *outbuf;
	int i, j, ret;
	struct oil_scale os;
	jpeg_saved_marker_ptr marker;
	enum oil_colorspace cs;

	prepare_jpeg_decompress(input, &dinfo, &jerr);

	/* Use the image dimensions read from the header to calculate our final
	 * output dimensions.
	 */
	oil_fix_ratio(dinfo.output_width, dinfo.output_height, &width_out, &height_out);

	/* Allocate jpeg decoder output buffer */
	inbuf = malloc(dinfo.output_width * dinfo.output_components);
	if (!inbuf) {
		fprintf(stderr, "Unable to allocate buffers.");
		exit(1);
	}

	/* Map jpeg to oil color space. */
	cs = jpeg_cs_to_oil(dinfo.out_color_space);

	/* set up scaler */
	ret = oil_scale_init(&os, dinfo.output_height, height_out,
		dinfo.output_width, width_out, cs);
	if (ret!=0) {
		fprintf(stderr, "Unable to allocate buffers.");
		exit(1);
	}

	/* Allocate linear converter output buffer */
	outbuf = malloc(width_out * OIL_CMP(os.cs) * sizeof(unsigned char));
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate buffers.");
		exit(1);
	}

	/* Jpeg compressor. */
	cinfo.err = &jerr;
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, output);
	cinfo.image_width = os.out_width;
	cinfo.image_height = os.out_height;
	cinfo.input_components = OIL_CMP(os.cs);
	cinfo.in_color_space = oil_cs_to_jpeg(os.cs);
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
	for(i=os.out_height; i>0; i--) {
		for (j=oil_scale_slots(&os); j>0; j--) {
			jpeg_read_scanlines(&dinfo, &inbuf, 1);
			oil_scale_in(&os, inbuf);
		}
		oil_scale_out(&os, outbuf);
		jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	jpeg_finish_decompress(&dinfo);
	jpeg_destroy_decompress(&dinfo);
	free(outbuf);
	free(inbuf);
	oil_scale_free(&os);
}

int main(int argc, char *argv[])
{
	int width, height;
	char *end;

	if (argc != 3) {
		fprintf(stderr, "Usage: %s WIDTH HEIGHT < in.jpg > scale.jpg\n", argv[0]);
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

	jpeg(stdin, stdout, width, height);

	fclose(stdin);
	return 0;
}
