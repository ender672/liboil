#include "resample.h"
#include "quant.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include <png.h>
#include <gif_lib.h>

/**
 * Mini format for uncompressed image data.
 */

struct header {
	uint8_t sig1;
	uint8_t sig2;
	uint8_t cmp;
	uint8_t opts;
	uint32_t width;
	uint32_t height;
};

static int read_header(FILE *f, struct header *hdr)
{
	size_t read_len;

	read_len = fread(hdr, 1, sizeof(struct header), f);
	if (read_len != sizeof(struct header)) {
		return 1;
	}

	if (hdr->sig1 != 'O' || hdr->sig2 != 'L') {
		return 1;
	}

	return 0;
}

static int write_header(FILE *f, uint32_t width, uint32_t height, uint8_t cmp,
	uint8_t opts)
{
	struct header hdr;
	size_t io_len;

	hdr.sig1 = 'O';
	hdr.sig2 = 'L';
	hdr.width = width;
	hdr.height = height;
	hdr.cmp = cmp;
	hdr.opts = opts;

	io_len = fwrite(&hdr, 1, sizeof(struct header), f);
	if (io_len != sizeof(struct header)) {
		return 1;
	}

	return 0;
}

/**
 * X Scaling.
 */

static int scalex(FILE *input, FILE *output, uint32_t width)
{
	int ret;
	uint32_t i;
	uint8_t *inbuf, *outbuf;
	size_t iolen, inbuf_len, outbuf_len;
	struct header hdr;

	if (read_header(input, &hdr)) {
		return 1;
	}

	if (write_header(output, width, hdr.height, hdr.cmp, hdr.opts)) {
		return 1;
	}

	ret = 0;
	inbuf_len = hdr.width * hdr.cmp;
	outbuf_len = width * hdr.cmp;

	inbuf = malloc(inbuf_len);
	outbuf = malloc(outbuf_len);

	for (i=0; i<hdr.height; i++) {
		iolen = fread(inbuf, 1, inbuf_len, input);
		if (iolen != inbuf_len) {
			ret = 1;
			break;
		}
		xscale(inbuf, hdr.width, outbuf, width, hdr.cmp, hdr.opts);
		iolen = fwrite(outbuf, 1, outbuf_len, output);
		if (iolen != outbuf_len) {
			ret = 1;
			break;
		}
	}

	free(outbuf);
	free(inbuf);
	return ret;
}

int scalex_cmd(int argc, char *argv[])
{
	int i;
	uint32_t width;
	char *end;
	FILE *input;

	for (i=1; i<argc; i++) {
		if (!strcmp(argv[i], "--help")) {
			goto usage_exit;
		}
	}

	if (argc != 2 && argc != 3) {
		goto usage_exit;
	}

	width = strtoul(argv[1], &end, 10);
	if (*end) {
		fprintf(stderr, "Error: Invalid width.\n");
		return 1;
	}

	input = argc == 3 ? fopen(argv[2], "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (scalex(input, stdout, width)) {
		fprintf(stderr, "Error: X Scaling Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;

	usage_exit:
	fprintf(stderr, "Usage: %s WIDTH [FILE]\n", argv[0]);
	return 1;
}

/**
 * Y Scaling.
 */

static int scaley(FILE *input, FILE *output, uint32_t height)
{
	int ret;
	uint32_t i;
	uint8_t *scaled_sl, *tmp;
	size_t buf_len;
	struct header hdr;
	struct yscaler ys;

	if (read_header(input, &hdr)) {
		return 1;
	}

	if (write_header(output, hdr.width, height, hdr.cmp, hdr.opts)) {
		return 1;
	}

	ret = 0;
	buf_len = hdr.width * hdr.cmp;
	scaled_sl = malloc(buf_len);

	yscaler_init(&ys, hdr.height, height, buf_len);
	for (i=0; i<height; i++) {
		while ((tmp = yscaler_next(&ys))) {
			if (!fread(tmp, buf_len, 1, input)) {
				ret = 1;
				break;
			}
		}
		yscaler_scale(&ys, scaled_sl, hdr.width, hdr.cmp, hdr.opts, i);
		if (!fwrite(scaled_sl, buf_len, 1, output)) {
			ret = 1;
			break;
		}
	}

	free(scaled_sl);
	yscaler_free(&ys);
	return ret;
}

int scaley_cmd(int argc, char *argv[])
{
	int i;
	uint32_t height;
	char *end;
	FILE *input;

	for (i=1; i<argc; i++) {
		if (!strcmp(argv[i], "--help")) {
			goto usage_exit;
		}
	}

	if (argc != 2 && argc != 3) {
		goto usage_exit;
	}

	height = strtoul(argv[1], &end, 10);
	if (*end) {
		fprintf(stderr, "Error: Invalid height.\n");
		return 1;
	}

	input = argc == 3 ? fopen(argv[2], "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (scaley(input, stdout, height)) {
		fprintf(stderr, "Error: Y Scaling Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;

	usage_exit:
	fprintf(stderr, "Usage: %s HEIGHT [FILE]\n", argv[0]);
	return 1;
}

/**
 * JPEG info.
 */

static char *jpeg_color_space_to_str(J_COLOR_SPACE jcs)
{
	switch (jcs) {
	case JCS_UNKNOWN:
		return "JCS_UNKNOWN";
	case JCS_GRAYSCALE:
		return "JCS_GRAYSCALE";
	case JCS_RGB:
		return "JCS_RGB";
	case JCS_YCbCr:
		return "JCS_YCbCr";
	case JCS_CMYK:
		return "JCS_CMYK";
	case JCS_YCCK:
		return "JCS_YCCK";
#ifdef JCS_EXTENSIONS
	case JCS_EXT_RGB:
		return "JCS_EXT_RGB";
	case JCS_EXT_RGBX:
		return "JCS_EXT_RGBX";
	case JCS_EXT_BGR:
		return "JCS_EXT_BGR";
	case JCS_EXT_BGRX:
		return "JCS_EXT_BGRX";
	case JCS_EXT_XBGR:
		return "JCS_EXT_XBGR";
	case JCS_EXT_XRGB:
		return "JCS_EXT_XRGB";
	case JCS_EXT_RGBA:
		return "JCS_EXT_RGBA";
	case JCS_EXT_BGRA:
		return "JCS_EXT_BGRA";
	case JCS_EXT_ABGR:
		return "JCS_EXT_ABGR";
	case JCS_EXT_ARGB:
		return "JCS_EXT_ARGB";
#endif
	}
	return "UNKNOWN";
}

static int jpeginfo(FILE *input, FILE *output)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
	char *cs;

	dinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&dinfo);
	jpeg_stdio_src(&dinfo, input);
	jpeg_read_header(&dinfo, TRUE);
	cs = jpeg_color_space_to_str(dinfo.out_color_space);

	printf("width:       %20u\n", dinfo.image_width);
	printf("height:      %20u\n", dinfo.image_height);
	printf("color_space: %20s\n", cs);

	jpeg_destroy_decompress(&dinfo);

	return 0;
}

int jpeginfo_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (jpeginfo(input, stdout)) {
		fprintf(stderr, "Error: JPEG Info Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

/**
 * JPEG Decoder.
 */

static int rjpeg(FILE *input, FILE *output, int rgbx, int downscale)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
	uint32_t i;
	uint8_t *buf;
	size_t buf_len, io_len;
	int ret, opts;

	ret = 0;
	opts = 0;

	dinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&dinfo);
	jpeg_stdio_src(&dinfo, input);
	jpeg_read_header(&dinfo, TRUE);

#ifdef JCS_EXTENSIONS
	if (rgbx && dinfo.out_color_space == JCS_RGB) {
		opts = OIL_FILLER;
		dinfo.out_color_space = JCS_EXT_RGBX;
	}
#endif

	if (downscale) {
		dinfo.scale_denom = downscale;
	}

	jpeg_start_decompress(&dinfo);

	buf_len = dinfo.output_width * dinfo.output_components;
	buf = malloc(buf_len);

	if (write_header(output, dinfo.output_width, dinfo.output_height,
		dinfo.output_components, opts)) {
		ret = 1;
		goto cleanup_exit;
	}

	for(i=0; i<dinfo.output_height; i++) {
		jpeg_read_scanlines(&dinfo, &buf, 1);
		io_len = fwrite(buf, 1, buf_len, output);
		if (io_len != buf_len) {
			ret = 1;
			goto cleanup_exit;
		}
	}

	jpeg_finish_decompress(&dinfo);

	cleanup_exit:
	jpeg_destroy_decompress(&dinfo);
	free(buf);
	return ret;
}

int rjpeg_cmd(int argc, char *argv[])
{
	int i, rgbx, downscale;
	char *arg, *downscale_str, *file_path, *end;
	FILE *input;

	file_path = 0;

	rgbx = 1;
	downscale = 0;
	downscale_str = 0;

	for (i=1; i<argc; i++) {
		arg = argv[i];

		if (!strcmp(arg, "--help")) {
			goto usage_exit;
		} else if (!strcmp(arg, "--norgbx")) {
			rgbx = 0;
		} else if (!strcmp(arg, "--downscale") && i+1 < argc) {
			downscale_str = argv[++i];
		} else {
			file_path = arg;
			i++;
			break;
		}
	}

	if (argc != i) {
		goto usage_exit;
	}

	if (downscale_str) {
		downscale = strtoul(downscale_str, &end, 10);
		if (*end) {
			fprintf(stderr, "Error: Invalid downscale value.\n");
			return 1;
		}
	}

	input = stdin;
	if (file_path) {
		input = fopen(file_path, "r+");
	}
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (rjpeg(input, stdout, rgbx, downscale)) {
		fprintf(stderr, "Error: JPEG Decoding Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;

	usage_exit:
	fprintf(stderr, "Usage: %s [OPTIONS] [FILE]\n", argv[0]);
	return 1;
}

/**
 * JPEG Encoder.
 */

static int wjpeg(FILE *input, FILE *output)
{
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	uint32_t i;
	uint8_t *buf;
	size_t buf_len, io_len;
	int ret;
	struct header hdr;

	if (read_header(input, &hdr)) {
		fprintf(stderr, "read header failed!\n");
		return 1;
	}

	ret = 0;
	buf_len = hdr.width * hdr.cmp;

	buf = malloc(buf_len);

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, output);
	cinfo.image_width = hdr.width;
	cinfo.image_height = hdr.height;
	cinfo.input_components = hdr.cmp;

	switch (hdr.cmp) {
	case 1:
		cinfo.in_color_space = JCS_GRAYSCALE;
		break;
	case 3:
		cinfo.in_color_space = JCS_RGB;
		break;
#ifdef JCS_EXTENSIONS
	case 4:
		cinfo.in_color_space = JCS_EXT_RGBX;
		break;
#endif
	default:
		goto cleanup_exit;
	}

	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 95, FALSE);
	jpeg_start_compress(&cinfo, TRUE);

	for (i=0; i<hdr.height; i++) {
		io_len = fread(buf, 1, buf_len, input);
		if (io_len != buf_len) {
			fprintf(stderr, "ERROR: Unable to read image.\n");
			ret = 1;
			goto cleanup_exit;
		}
		jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&buf, 1);
	}

	jpeg_finish_compress(&cinfo);

	cleanup_exit:
	jpeg_destroy_compress(&cinfo);
	free(buf);
	return ret;
}

int wjpeg_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (wjpeg(input, stdout)) {
		fprintf(stderr, "Error: JPEG Compression Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

/**
 * PNG info.
 */

static char *ctype_to_str(png_byte ctype)
{
	switch (ctype) {
	case PNG_COLOR_TYPE_GRAY:
		return "PNG_COLOR_TYPE_GRAY";
	case PNG_COLOR_TYPE_PALETTE:
		return "PNG_COLOR_TYPE_PALETTE";
	case PNG_COLOR_TYPE_RGB:
		return "PNG_COLOR_TYPE_RGB";
	case PNG_COLOR_TYPE_RGB_ALPHA:
		return "PNG_COLOR_TYPE_RGB_ALPHA";
	case PNG_COLOR_TYPE_GRAY_ALPHA:
		return "PNG_COLOR_TYPE_GRAY_ALPHA";
	}
	return "UNKNOWN";
}

static int pnginfo(FILE *input, FILE *output)
{
	png_structp png;
	png_infop info;
	char *cs;

	png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	info = png_create_info_struct(png);
	png_init_io(png, input);
	png_read_info(png, info);

	cs = ctype_to_str(png_get_color_type(png, info));

	printf("width:       %20lu\n", (unsigned long)png_get_image_width(png, info));
	printf("height:      %20lu\n", (unsigned long)png_get_image_height(png, info));
	printf("color_space: %20s\n", cs);

	png_destroy_read_struct(&png, &info, NULL);
	return 0;
}

int pnginfo_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (pnginfo(input, stdout)) {
		fprintf(stderr, "Error: PNG Info Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

/**
 * PNG Decoder.
 */

int rpng_interlaced(png_structp png, png_infop info, FILE *output)
{
	uint8_t **sl;
	uint32_t i, height;
	size_t buf_len, io_len;
	int ret;

	ret = 0;

	height = png_get_image_height(png, info);
	sl = malloc(height * sizeof(uint8_t *));

	buf_len = png_get_rowbytes(png, info);
	for (i=0; i<height; i++) {
		sl[i] = malloc(buf_len);
	}

	if (setjmp(png_jmpbuf(png))) {
		ret = 1;
		goto cleanup_exit;
	}

	png_read_image(png, sl);

	for (i=0; i<height; i++) {
		io_len = fwrite(sl[i], 1, buf_len, output);
		if (io_len != buf_len) {
			ret = 1;
			break;
		}
	}

	cleanup_exit:

	for (i=0; i<height; i++) {
		free(sl[i]);
	}
	free(sl);
	return ret;
}

int rpng_noninterlaced(png_structp png, png_infop info, FILE *output)
{
	uint8_t *buf;
	uint32_t i, height;
	size_t buf_len, io_len;

	buf_len = png_get_rowbytes(png, info);
	buf = malloc(buf_len);

	if (setjmp(png_jmpbuf(png))) {
		free(buf);
		return 1;
	}

	height = png_get_image_height(png, info);

	for (i=0; i<height; i++) {
		png_read_row(png, buf, NULL);
		io_len = fwrite(buf, 1, buf_len, output);
		if (io_len != buf_len) {
			return 1;
		}
	}

	free(buf);
	return 0;
}

static int rpng(FILE *input, FILE *output)
{
	png_structp png;
	png_infop info;
	int ret, opts;
	png_uint_32 width, height;
	png_byte channels;

	ret = 0;
	opts = 0;

	png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (setjmp(png_jmpbuf(png))) {
		ret = 1;
		goto cleanup_exit;
	}

	info = png_create_info_struct(png);
	png_init_io(png, input);
	png_read_info(png, info);

	png_set_packing(png);
	png_set_strip_16(png);
	png_set_expand(png);

	if (png_get_color_type(png, info) == PNG_COLOR_TYPE_RGB) {
		png_set_filler(png, 0, PNG_FILLER_AFTER);
		opts = OIL_FILLER;
	}
	png_read_update_info(png, info);

	width = png_get_image_width(png, info);
	height = png_get_image_height(png, info);
	channels = png_get_channels(png, info);

	if (write_header(output, width, height, channels, opts)) {
		ret = 1;
		goto cleanup_exit;
	}

	switch (png_get_interlace_type(png, info)) {
	case PNG_INTERLACE_NONE:
		ret = rpng_noninterlaced(png, info, output);
		break;
	case PNG_INTERLACE_ADAM7:
		ret = rpng_interlaced(png, info, output);
		break;
	default:
		ret = 1;
	}

	cleanup_exit:
	png_destroy_read_struct(&png, &info, NULL);
	return ret;
}

int rpng_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (rpng(input, stdout)) {
		fprintf(stderr, "Error: PNG Decoding Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

/**
 * PNG Encoder.
 */

static int wpng(FILE *input, FILE *output)
{
	png_structp png;
	png_infop info;
	png_byte ctype;
	uint32_t i;
	uint8_t *buf;
	size_t buf_len, io_len;
	int ret;
	struct header hdr;

	if (read_header(input, &hdr)) {
		fprintf(stderr, "Error: Read header failed!\n");
		return 1;
	}

	ret = 0;
	buf_len = hdr.width * hdr.cmp;
	buf = malloc(buf_len);

	png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	info = png_create_info_struct(png);
	png_init_io(png, output);

	if (hdr.cmp == 1) {
		ctype = PNG_COLOR_TYPE_GRAY;
	} else if (hdr.cmp == 2) {
		ctype = PNG_COLOR_TYPE_GRAY_ALPHA;
	} else if (hdr.cmp == 3 || (hdr.cmp == 4 && hdr.opts & OIL_FILLER)) {
		ctype = PNG_COLOR_TYPE_RGB;
	} else if (hdr.cmp == 4) {
		ctype = PNG_COLOR_TYPE_RGB_ALPHA;
	} else {
		goto cleanup_exit;
	}

	png_set_IHDR(png, info, hdr.width, hdr.height, 8, ctype,
		PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);

	png_write_info(png, info);

	if (hdr.opts & OIL_FILLER) {
		png_set_filler(png, 0, PNG_FILLER_AFTER);
	}

	for (i=0; i<hdr.height; i++) {
		io_len = fread(buf, 1, buf_len, input);
		if (io_len != buf_len) {
			fprintf(stderr, "Error: Unable to read.\n");
			ret = 1;
			goto cleanup_exit;
		}
		png_write_row(png, buf);
	}

	png_write_end(png, info);

	cleanup_exit:
	png_destroy_write_struct(&png, &info);
	free(buf);
	return ret;
}

int wpng_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (wpng(input, stdout)) {
		fprintf(stderr, "Error: PNG Compression Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

/**
 * GIF support for 5.0.x versions, e.g. FreeBSD.
 */

#ifdef GIFLIB_MAJOR

static int gifinfo(FILE *input, FILE *output)
{
	GifFileType *gif;
	GifRecordType record_type;
	int ext_type;
	GifByteType *ext_data;
	uint16_t delay;
	int num_images, giferr;

	num_images = 0;
	delay = 0;

	gif = DGifOpenFileHandle(fileno(input), &giferr);

	if (!gif) {
		fprintf(stderr, "Error: Unable to read gif file.\n");
		return 1;
	}

	while(1) {
		DGifGetRecordType(gif, &record_type);

		switch (record_type) {
		case EXTENSION_RECORD_TYPE:
			DGifGetExtension(gif, &ext_type, &ext_data);
			if (ext_type == GRAPHICS_EXT_FUNC_CODE) {
				delay = *(uint16_t *)(ext_data + 2);
			}
			while (ext_data != NULL) {
				if (DGifGetExtensionNext(gif, &ext_data) == GIF_ERROR) {
					goto end;
				}
			}
			break;
		case TERMINATE_RECORD_TYPE:
			goto end;
			break;
		case IMAGE_DESC_RECORD_TYPE:
			num_images++;
			break;
		case SCREEN_DESC_RECORD_TYPE:
		case UNDEFINED_RECORD_TYPE:
			break;
		}
	};

	end:

	fprintf(output, "width:       %20d\n", gif->SWidth);
	fprintf(output, "height:      %20d\n", gif->SHeight);
	fprintf(output, "delay:       %20d\n", delay);
	fprintf(output, "image count: %20d\n", num_images);

	DGifCloseFile(gif);
	return 0;
}

static int rgif(FILE *input, FILE *output)
{
	GifFileType *gif;
	GifRecordType record_type;
	GifPixelType *gif_buf;
	uint8_t *buf;
	uint32_t i, j, width, height;
	GifColorType *color, *colors;
	int ret, giferr;
	size_t buf_len, io_len;

	ret = 0;

	gif = DGifOpenFileHandle(fileno(input), &giferr);

	do {
		DGifGetRecordType(gif, &record_type);
		switch (record_type) {
		case IMAGE_DESC_RECORD_TYPE:
			DGifGetImageDesc(gif);

			if (gif->Image.ColorMap) {
				colors = gif->Image.ColorMap->Colors;
			} else if (gif->SColorMap) {
				colors = gif->SColorMap->Colors;
			} else {
				fprintf(stderr, "Error: GIF image has no color map.\n");
				ret = 1;
				goto cleanup_exit;
			}
			width = gif->Image.Width;
			height = gif->Image.Height;

			if (write_header(output, width, height, 4, OIL_FILLER)) {
				ret = 1;
				goto cleanup_exit;
			}

			gif_buf = malloc(width * sizeof(GifPixelType));
			buf_len = width * 4;
			buf = malloc(buf_len);

			for(i=0; i<height; i++) {
				DGifGetLine(gif, gif_buf, width);
				for (j=0; j<width; j++) {
					color = &colors[gif_buf[j]];
					buf[j*4] = color->Red;
					buf[j*4+1] = color->Green;
					buf[j*4+2] = color->Blue;
					buf[j*4+3] = 0;
				}

				io_len = fwrite(buf, 1, buf_len, output);
				if (io_len != buf_len) {
					fprintf(stderr, "ERROR: unable to write output.\n");
					return 1;
				}
			}
			free(buf);
			free(gif_buf);
			break;
		case SCREEN_DESC_RECORD_TYPE:
		case EXTENSION_RECORD_TYPE:
		case UNDEFINED_RECORD_TYPE:
		case TERMINATE_RECORD_TYPE:
			break;
		}
	} while (record_type != TERMINATE_RECORD_TYPE && record_type != IMAGE_DESC_RECORD_TYPE);

	cleanup_exit:
	DGifCloseFile(gif);
	return ret;
}

static int wgif(FILE *input, FILE *output)
{
	struct header hdr;
	struct quant quant;
	uint8_t **inbufs, *outbuf;
	uint32_t i;
	size_t row_len, io_len;
	int ret, count, giferr;
	ColorMapObject *cmap;
	GifColorType *colors;
	GifFileType *gif;

	ret = 0;

	if (read_header(input, &hdr)) {
		fprintf(stderr, "Error: Read header failed!\n");
		return 1;
	}

	if (hdr.cmp != 3) {
		fprintf(stderr, "quant only supports images with 3 channels.\n");
		return 1;
	}

	row_len = hdr.width * hdr.cmp;

	inbufs = malloc(hdr.height * sizeof(uint8_t *));
	outbuf = malloc(hdr.width);

	quant.output_width = hdr.width;
	quant.desired_number_of_colors = 256;

	quant_init(&quant);

	for (i=0; i<hdr.height; i++) {
		inbufs[i] = malloc(row_len);
		io_len = fread(inbufs[i], 1, row_len, input);
		if (io_len != row_len) {
			fprintf(stderr, "Error: Unable to read!\n");
			ret = 1;
			goto cleanup_exit;
		}
		quant_index(&quant, inbufs[i]);
	}

	quant_gen_palette(&quant);

	/* Prepare GIF Color Map */
	count = quant.actual_number_of_colors;
	cmap = GifMakeMapObject(count, NULL);
	colors = cmap->Colors;
	for (i=0; i<count; i++) {
		colors[i].Red = quant.colormap[0][i];
		colors[i].Green = quant.colormap[1][i];
		colors[i].Blue = quant.colormap[2][i];
	}

	/* Write GIF */
	gif = EGifOpenFileHandle(fileno(output), &giferr);
	EGifPutScreenDesc(gif, hdr.width, hdr.height, count, 0, cmap);
	EGifPutImageDesc(gif, 0, 0, hdr.width, hdr.height, FALSE, NULL);

	for(i=0; i<hdr.height; i++) {
		quant_map(&quant, inbufs[i], outbuf);
		EGifPutLine(gif, outbuf, hdr.width);
	}

	EGifCloseFile(gif);

	/* Free GIF Color Map */
	GifFreeMapObject(cmap);

	cleanup_exit:

	free(outbuf);
	for (i=0; i<hdr.height; i++) {
		free(inbufs[i]);
	}
	free(inbufs);

	quant_free(&quant);

	return ret;
}

/**
 * GIF support for 4.1.x versions, e.g. Fedora & Ubuntu.
 */
#else

static int gifinfo(FILE *input, FILE *output)
{
	GifFileType *gif;
	GifRecordType record_type;
	int ext_type;
	GifByteType *ext_data;
	uint16_t delay;
	int num_images;

	num_images = 0;
	delay = 0;

	gif = DGifOpenFileHandle(fileno(input));

	if (!gif) {
		fprintf(stderr, "Error: Unable to read gif file.\n");
		return 1;
	}

	while(1) {
		DGifGetRecordType(gif, &record_type);

		switch (record_type) {
		case EXTENSION_RECORD_TYPE:
			DGifGetExtension(gif, &ext_type, &ext_data);
			if (ext_type == GRAPHICS_EXT_FUNC_CODE) {
				delay = *(uint16_t *)(ext_data + 2);
			}
			while (ext_data != NULL) {
				if (DGifGetExtensionNext(gif, &ext_data) == GIF_ERROR) {
					goto end;
				}
			}
			break;
		case TERMINATE_RECORD_TYPE:
			goto end;
			break;
		case IMAGE_DESC_RECORD_TYPE:
			num_images++;
			break;
		case SCREEN_DESC_RECORD_TYPE:
		case UNDEFINED_RECORD_TYPE:
			break;
		}
	};

	end:

	fprintf(output, "width:       %20d\n", gif->SWidth);
	fprintf(output, "height:      %20d\n", gif->SHeight);
	fprintf(output, "delay:       %20d\n", delay);
	fprintf(output, "image count: %20d\n", num_images);

	DGifCloseFile(gif);
	return 0;
}

static int rgif(FILE *input, FILE *output)
{
	GifFileType *gif;
	GifRecordType record_type;
	GifPixelType *gif_buf;
	uint8_t *buf;
	uint32_t i, j, width, height;
	GifColorType *color, *colors;
	int ret;
	size_t buf_len, io_len;

	ret = 0;

	gif = DGifOpenFileHandle(fileno(input));

	do {
		DGifGetRecordType(gif, &record_type);
		switch (record_type) {
		case IMAGE_DESC_RECORD_TYPE:
			DGifGetImageDesc(gif);

			if (gif->Image.ColorMap) {
				colors = gif->Image.ColorMap->Colors;
			} else if (gif->SColorMap) {
				colors = gif->SColorMap->Colors;
			} else {
				fprintf(stderr, "Error: GIF image has no color map.\n");
				ret = 1;
				goto cleanup_exit;
			}
			width = gif->Image.Width;
			height = gif->Image.Height;

			if (write_header(output, width, height, 4, OIL_FILLER)) {
				ret = 1;
				goto cleanup_exit;
			}

			gif_buf = malloc(width * sizeof(GifPixelType));
			buf_len = width * 4;
			buf = malloc(buf_len);

			for(i=0; i<height; i++) {
				DGifGetLine(gif, gif_buf, width);
				for (j=0; j<width; j++) {
					color = &colors[gif_buf[j]];
					buf[j*4] = color->Red;
					buf[j*4+1] = color->Green;
					buf[j*4+2] = color->Blue;
					buf[j*4+3] = 0;
				}

				io_len = fwrite(buf, 1, buf_len, output);
				if (io_len != buf_len) {
					fprintf(stderr, "ERROR: unable to write output.\n");
					return 1;
				}
			}
			free(buf);
			free(gif_buf);
			break;
		case SCREEN_DESC_RECORD_TYPE:
		case EXTENSION_RECORD_TYPE:
		case UNDEFINED_RECORD_TYPE:
		case TERMINATE_RECORD_TYPE:
			break;
		}
	} while (record_type != TERMINATE_RECORD_TYPE && record_type != IMAGE_DESC_RECORD_TYPE);

	cleanup_exit:
	DGifCloseFile(gif);
	return ret;
}

static int wgif(FILE *input, FILE *output)
{
	struct header hdr;
	struct quant quant;
	uint8_t **inbufs, *outbuf;
	uint32_t i;
	size_t row_len, io_len;
	int ret, count;
	ColorMapObject *cmap;
	GifColorType *colors;
	GifFileType *gif;

	ret = 0;

	if (read_header(input, &hdr)) {
		fprintf(stderr, "Error: Read header failed!\n");
		return 1;
	}

	if (hdr.cmp != 3) {
		fprintf(stderr, "quant only supports images with 3 channels.\n");
		return 1;
	}

	row_len = hdr.width * hdr.cmp;

	inbufs = malloc(hdr.height * sizeof(uint8_t *));
	outbuf = malloc(hdr.width);

	quant.output_width = hdr.width;
	quant.desired_number_of_colors = 256;

	quant_init(&quant);

	for (i=0; i<hdr.height; i++) {
		inbufs[i] = malloc(row_len);
		io_len = fread(inbufs[i], 1, row_len, input);
		if (io_len != row_len) {
			fprintf(stderr, "Error: Unable to read!\n");
			ret = 1;
			goto cleanup_exit;
		}
		quant_index(&quant, inbufs[i]);
	}

	quant_gen_palette(&quant);

	/* Prepare GIF Color Map */
	count = quant.actual_number_of_colors;
	cmap = MakeMapObject(count, NULL);
	colors = cmap->Colors;
	for (i=0; i<count; i++) {
		colors[i].Red = quant.colormap[0][i];
		colors[i].Green = quant.colormap[1][i];
		colors[i].Blue = quant.colormap[2][i];
	}

	/* Write GIF */
	gif = EGifOpenFileHandle(fileno(output));
	EGifPutScreenDesc(gif, hdr.width, hdr.height, count, 0, cmap);
	EGifPutImageDesc(gif, 0, 0, hdr.width, hdr.height, FALSE, NULL);

	for(i=0; i<hdr.height; i++) {
		quant_map(&quant, inbufs[i], outbuf);
		EGifPutLine(gif, outbuf, hdr.width);
	}

	EGifCloseFile(gif);

	/* Free GIF Color Map */
	FreeMapObject(cmap);

	cleanup_exit:

	free(outbuf);
	for (i=0; i<hdr.height; i++) {
		free(inbufs[i]);
	}
	free(inbufs);

	quant_free(&quant);

	return ret;
}
#endif

/**
 * GIF commands.
 */

static int gifinfo_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (gifinfo(input, stdout)) {
		fprintf(stderr, "Error: GIF Info Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

static int rgif_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (rgif(input, stdout)) {
		fprintf(stderr, "Error: GIF Decoding Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

int wgif_cmd(int argc, char *argv[])
{
	char *arg;
	FILE *input;

	arg = NULL;

	if (argc == 2) {
		arg = argv[1];
	}

	if (argc > 2 || (argc == 2 && !strcmp(arg, "--help"))) {
		fprintf(stderr, "Usage: %s [FILE]\n", argv[0]);
		return 1;
	}

	input = arg ? fopen(arg, "r+") : stdin;
	if (!input) {
		fprintf(stderr, "Error: Unable to open file.\n");
		return 1;
	}

	if (wgif(input, stdout)) {
		fprintf(stderr, "Error: GIF Compression Failed.\n");
		fclose(input);
		return 1;
	}

	fclose(input);
	return 0;
}

/**
 * Main.
 */

int main(int argc, char *argv[])
{
	char *action, *cmd;
	FILE *usage;

	cmd = argv[0];
	usage = stderr;

	if (argc < 2) {
		goto usage_exit;
	}

	if (!strcmp(argv[1], "--help")) {
		usage = stdout;
		goto usage_exit;
	}

	action = argv[1];
	argv++;
	argc--;

	if (!strcmp(action, "scalex")) {
		return scalex_cmd(argc, argv);
	} else if (!strcmp(action, "scaley")) {
		return scaley_cmd(argc, argv);
	} else if (!strcmp(action, "rjpeg")) {
		return rjpeg_cmd(argc, argv);
	} else if (!strcmp(action, "wjpeg")) {
		return wjpeg_cmd(argc, argv);
	} else if (!strcmp(action, "jpeginfo")) {
		return jpeginfo_cmd(argc, argv);
	} else if (!strcmp(action, "rpng")) {
		return rpng_cmd(argc, argv);
	} else if (!strcmp(action, "wpng")) {
		return wpng_cmd(argc, argv);
	} else if (!strcmp(action, "pnginfo")) {
		return pnginfo_cmd(argc, argv);
	} else if (!strcmp(action, "gifinfo")) {
		return gifinfo_cmd(argc, argv);
	} else if (!strcmp(action, "rgif")) {
		return rgif_cmd(argc, argv);
	} else if (!strcmp(action, "wgif")) {
		return wgif_cmd(argc, argv);
	}

	usage_exit:

	fprintf(usage, "Usage: %s [--help] <command> [<args>]\n\n", cmd);
	fprintf(usage, "Commands:\n");
	fprintf(usage, "   jpeginfo  Print information about a jpeg file\n");
	fprintf(usage, "   rjpeg     Read a jpeg file\n");
	fprintf(usage, "   wjpeg     Write a jpeg file\n");
	fprintf(usage, "   pnginfo   Print information about a png file\n");
	fprintf(usage, "   rpng      Read a png file\n");
	fprintf(usage, "   wpng      Write a png file\n");
	fprintf(usage, "   gifinfo   Print information about a gif file\n");
	fprintf(usage, "   rgif      Read a gif file\n");
	fprintf(usage, "   wgif      Write a gif file\n");
	fprintf(usage, "   scaley    Scale image height\n");
	fprintf(usage, "   scalex    Scale image width\n");
	return 1;
}
