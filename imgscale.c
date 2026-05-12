#include "oil_libjpeg.h"
#include "oil_libpng.h"
#include "oil_resample.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <jpeglib.h>
#include <png.h>

enum file_format { FMT_PNG, FMT_JPEG };

struct backend_entry {
	const char *flag;
	int (*scale_in)(struct oil_scale *, unsigned char *);
	int (*scale_out)(struct oil_scale *, unsigned char *);
};

static const struct backend_entry backends[] = {
	{"--scalar", oil_scale_in, oil_scale_out},
#if defined(__x86_64__)
	{"--sse2",   oil_scale_in_sse2, oil_scale_out_sse2},
	{"--avx2",   oil_scale_in_avx2, oil_scale_out_avx2},
#elif defined(__aarch64__)
	{"--neon",   oil_scale_in_neon, oil_scale_out_neon},
#endif
};

static const struct backend_entry *find_backend(const char *flag) {
	size_t i;
	for (i = 0; i < sizeof(backends)/sizeof(backends[0]); i++) {
		if (strcmp(backends[i].flag, flag) == 0) return &backends[i];
	}
	return NULL;
}

static const struct backend_entry *default_backend(void) {
#if defined(__x86_64__)
	return find_backend(__builtin_cpu_supports("avx2") ? "--avx2" : "--sse2");
#elif defined(__aarch64__)
	return find_backend("--neon");
#else
	return find_backend("--scalar");
#endif
}

static enum oil_colorspace nogamma_cs(enum oil_colorspace cs) {
	switch (cs) {
	case OIL_CS_RGB:  return OIL_CS_RGB_NOGAMMA;
	case OIL_CS_RGBA: return OIL_CS_RGBA_NOGAMMA;
	case OIL_CS_RGBX: return OIL_CS_RGBX_NOGAMMA;
	default: return cs;
	}
}

static int oil_cs_to_png_ctype(enum oil_colorspace cs) {
	switch (cs) {
	case OIL_CS_G:
		return PNG_COLOR_TYPE_GRAY;
	case OIL_CS_GA:
		return PNG_COLOR_TYPE_GRAY_ALPHA;
	case OIL_CS_RGB:
	case OIL_CS_RGB_NOGAMMA:
	case OIL_CS_RGBX:
	case OIL_CS_RGBX_NOGAMMA:
		return PNG_COLOR_TYPE_RGB;
	case OIL_CS_RGBA:
	case OIL_CS_RGBA_NOGAMMA:
		return PNG_COLOR_TYPE_RGB_ALPHA;
	default:
		return -1;
	}
}

static enum file_format detect_out_format(const char *path, enum file_format fallback) {
	const char *ext;
	if (!path) return fallback;
	ext = strrchr(path, '.');
	if (!ext) return fallback;
	if (strcasecmp(ext, ".png") == 0) return FMT_PNG;
	if (strcasecmp(ext, ".jpg") == 0 ||
	    strcasecmp(ext, ".jpeg") == 0 ||
	    strcasecmp(ext, ".jpe") == 0) return FMT_JPEG;
	return fallback;
}

static void png_writer_open(png_structp *wpng, png_infop *winfo, FILE *output,
	int width, int height, int ctype)
{
	*wpng = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	*winfo = png_create_info_struct(*wpng);
	png_init_io(*wpng, output);
	png_set_IHDR(*wpng, *winfo, width, height, 8, ctype, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_write_info(*wpng, *winfo);
}

static void jpeg_writer_open(struct jpeg_compress_struct *cinfo,
	struct jpeg_error_mgr *jerr, FILE *output, int width, int height,
	enum oil_colorspace cs, jpeg_saved_marker_ptr markers)
{
	jpeg_saved_marker_ptr marker;

	cinfo->err = jpeg_std_error(jerr);
	jpeg_create_compress(cinfo);
	jpeg_stdio_dest(cinfo, output);
	cinfo->image_width = width;
	cinfo->image_height = height;
	cinfo->input_components = OIL_CMP(cs);
	cinfo->in_color_space = oil_cs_to_jpeg(cs);
	jpeg_set_defaults(cinfo);
	jpeg_set_quality(cinfo, 94, FALSE);

	/* Disable chroma subsampling (4:4:4) for higher fidelity. Only meaningful
	 * for color JPEGs that get internally converted to YCbCr.
	 */
	if (cinfo->num_components >= 3) {
		cinfo->comp_info[0].h_samp_factor = 1;
		cinfo->comp_info[0].v_samp_factor = 1;
	}

	jpeg_start_compress(cinfo, TRUE);

	for (marker = markers; marker; marker = marker->next) {
		jpeg_write_marker(cinfo, marker->marker, marker->data,
			marker->data_length);
	}
}

static void process_png_input(FILE *input, FILE *output, int width, int height,
	const struct backend_entry *be, int no_gamma, enum file_format out_fmt)
{
	int i, in_width, in_height, ret, ol_inited = 0, interlaced;
	png_structp rpng, wpng = NULL;
	png_infop rinfo, winfo = NULL;
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	int jpeg_started = 0;
	struct oil_libpng ol;
	unsigned char *outbuf = NULL;
	int out_ctype;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!rpng) {
		fprintf(stderr, "Unable to create PNG read struct.\n");
		exit(1);
	}

	rinfo = png_create_info_struct(rpng);
	if (!rinfo) {
		png_destroy_read_struct(&rpng, NULL, NULL);
		fprintf(stderr, "Unable to create PNG info struct.\n");
		exit(1);
	}

	if (setjmp(png_jmpbuf(rpng))) {
		free(outbuf);
		if (ol_inited)
			oil_libpng_free(&ol);
		if (out_fmt == FMT_PNG)
			png_destroy_write_struct(&wpng, &winfo);
		else if (jpeg_started)
			jpeg_destroy_compress(&cinfo);
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		fprintf(stderr, "PNG Decoding Error.\n");
		exit(1);
	}
	png_init_io(rpng, input);
	png_read_info(rpng, rinfo);

	png_set_packing(rpng);
	png_set_strip_16(rpng);
	png_set_expand(rpng);
	if (out_fmt == FMT_JPEG) {
		/* JPEG cannot carry alpha. Strip it before scaling. */
		png_set_strip_alpha(rpng);
	}
	png_set_interlace_handling(rpng);
	png_read_update_info(rpng, rinfo);

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);
	oil_fix_ratio(in_width, in_height, &width, &height);

	ret = oil_libpng_init(&ol, rpng, rinfo, width, height);
	if (ret!=0) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}
	ol_inited = 1;
	if (no_gamma) ol.os.cs = nogamma_cs(ol.os.cs);
	interlaced = png_get_interlace_type(rpng, rinfo) == PNG_INTERLACE_ADAM7;

	if (out_fmt == FMT_PNG) {
		out_ctype = oil_cs_to_png_ctype(ol.os.cs);
		if (out_ctype < 0) {
			fprintf(stderr, "Cannot encode colorspace as PNG.\n");
			exit(1);
		}
		png_writer_open(&wpng, &winfo, output, width, height, out_ctype);
	} else {
		jpeg_writer_open(&cinfo, &jerr, output, width, height, ol.os.cs, NULL);
		jpeg_started = 1;
	}

	outbuf = malloc(width * OIL_CMP(ol.os.cs));
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	for(i=0; i<height; i++) {
		while (oil_scale_slots(&ol.os) > 0) {
			if (interlaced) {
				be->scale_in(&ol.os, ol.inimage[ol.in_vpos++]);
			} else {
				png_read_row(rpng, ol.inbuf, NULL);
				be->scale_in(&ol.os, ol.inbuf);
			}
		}
		be->scale_out(&ol.os, outbuf);
		if (out_fmt == FMT_PNG) {
			png_write_row(wpng, outbuf);
		} else {
			jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
		}
	}

	if (out_fmt == FMT_PNG) {
		png_write_end(wpng, winfo);
		png_destroy_write_struct(&wpng, &winfo);
	} else {
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);
	}
	png_destroy_read_struct(&rpng, &rinfo, NULL);

	free(outbuf);
	oil_libpng_free(&ol);
}

static void process_jpeg_input(FILE *input, FILE *output, int width_out,
	int height_out, const struct backend_entry *be, int no_gamma,
	enum file_format out_fmt)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	png_structp wpng = NULL;
	png_infop winfo = NULL;
	unsigned char *outbuf;
	int i, ret, out_ctype;
	long j;
	struct oil_libjpeg ol;

	dinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&dinfo);
	jpeg_stdio_src(&dinfo, input);

	/* Save custom headers for the compressor, but ignore APP0 & APP14 so
	 * libjpeg can handle them.
	 */
	jpeg_save_markers(&dinfo, JPEG_COM, 0xFFFF);
	for (j=1; j<14; j++) {
		jpeg_save_markers(&dinfo, JPEG_APP0+j, 0xFFFF);
	}
	jpeg_save_markers(&dinfo, JPEG_APP0+15, 0xFFFF);
	jpeg_read_header(&dinfo, TRUE);

	/* PNG has no native CMYK; ask libjpeg to convert to RGB on decode. */
	if (out_fmt == FMT_PNG && dinfo.jpeg_color_space == JCS_CMYK) {
		dinfo.out_color_space = JCS_RGB;
	}

	jpeg_start_decompress(&dinfo);

	oil_fix_ratio(dinfo.output_width, dinfo.output_height, &width_out, &height_out);

	ret = oil_libjpeg_init(&ol, &dinfo, width_out, height_out);
	if (ret!=0) {
		fprintf(stderr, "Unable to initialize scaler.");
		jpeg_destroy_decompress(&dinfo);
		fclose(input);
		fclose(output);
		exit(1);
	}
	if (no_gamma) ol.os.cs = nogamma_cs(ol.os.cs);

	outbuf = malloc(width_out * OIL_CMP(ol.os.cs));
	if (!outbuf) {
		fprintf(stderr, "Unable to allocate buffers.");
		oil_libjpeg_free(&ol);
		jpeg_destroy_decompress(&dinfo);
		exit(1);
	}

	if (out_fmt == FMT_PNG) {
		out_ctype = oil_cs_to_png_ctype(ol.os.cs);
		if (out_ctype < 0) {
			fprintf(stderr, "Cannot encode colorspace as PNG.\n");
			exit(1);
		}
		png_writer_open(&wpng, &winfo, output, width_out, height_out, out_ctype);
	} else {
		jpeg_writer_open(&cinfo, &jerr, output, width_out, height_out,
			ol.os.cs, dinfo.marker_list);
	}

	for(i=height_out; i>0; i--) {
		while (oil_scale_slots(&ol.os) > 0) {
			jpeg_read_scanlines(&dinfo, &ol.inbuf, 1);
			be->scale_in(&ol.os, ol.inbuf);
		}
		be->scale_out(&ol.os, outbuf);
		if (out_fmt == FMT_PNG) {
			png_write_row(wpng, outbuf);
		} else {
			jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
		}
	}

	if (out_fmt == FMT_PNG) {
		png_write_end(wpng, winfo);
		png_destroy_write_struct(&wpng, &winfo);
	} else {
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);
	}

	jpeg_finish_decompress(&dinfo);
	jpeg_destroy_decompress(&dinfo);
	free(outbuf);
	oil_libjpeg_free(&ol);
}

static int looks_like_png(FILE *io)
{
	int peek;
	peek = getc(io);
	ungetc(peek, io);
	return peek == 137;
}

int main(int argc, char *argv[])
{
	int width, height, argi, n_pos = 0;
	int no_gamma = 0;
	char *end;
	char *pos[4] = {NULL, NULL, NULL, NULL};
	const struct backend_entry *be = NULL;
	FILE *io_in, *io_out;
	enum file_format input_fmt, output_fmt;

	io_in = stdin;
	io_out = stdout;

	for (argi = 1; argi < argc; argi++) {
		if (strcmp(argv[argi], "--no-gamma") == 0) {
			no_gamma = 1;
		} else if (argv[argi][0] == '-' && argv[argi][1] == '-') {
			be = find_backend(argv[argi]);
			if (!be) {
				fprintf(stderr, "Error: unknown or unavailable backend: %s\n", argv[argi]);
				return 1;
			}
		} else {
			if (n_pos >= 4) {
				fprintf(stderr, "Error: too many positional arguments.\n");
				return 1;
			}
			pos[n_pos++] = argv[argi];
		}
	}
	if (!be) be = default_backend();

	if (n_pos < 2) {
		fprintf(stderr, "Usage: %s [--scalar|--sse2|--avx2|--neon] [--no-gamma] WIDTH HEIGHT [file] [file]\n", argv[0]);
		return 1;
	}

	width = strtol(pos[0], &end, 10);
	if (*end || width <= 0) {
		fprintf(stderr, "Error: Invalid width.\n");
		return 1;
	}

	height = strtol(pos[1], &end, 10);
	if (*end || height <= 0) {
		fprintf(stderr, "Error: Invalid height.\n");
		return 1;
	}

	if (pos[2]) {
		io_in = fopen(pos[2], "rb");
		if (!io_in) {
			fprintf(stderr, "Unable to open source file.\n");
			return 1;
		}
	}

	if (pos[3]) {
		io_out = fopen(pos[3], "wb");
		if (!io_out) {
			fprintf(stderr, "Unable to open destination file.\n");
			return 1;
		}
	}

	input_fmt = looks_like_png(io_in) ? FMT_PNG : FMT_JPEG;
	output_fmt = detect_out_format(pos[3], input_fmt);

	if (input_fmt == FMT_PNG) {
		process_png_input(io_in, io_out, width, height, be, no_gamma, output_fmt);
	} else {
		process_jpeg_input(io_in, io_out, width, height, be, no_gamma, output_fmt);
	}

	if (io_in != stdin)
		fclose(io_in);
	if (io_out != stdout)
		fclose(io_out);
	return 0;
}
