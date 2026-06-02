#include "oil_libjpeg.h"
#include "oil_libpng.h"
#include "oil_jxl.h"
#include "oil_resample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <jpeglib.h>
#include <png.h>
#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>
#include <pthread.h>

enum file_format { FMT_PNG, FMT_JPEG, FMT_JXL };

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
	if (strcasecmp(ext, ".jxl") == 0) return FMT_JXL;
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

/* libjxl's encoder, like its decoder, has no scanline-streaming API: a frame is
 * handed over whole. So the writer accumulates scaled rows into a full-image
 * buffer and encodes once, at finish. (PNG/JPEG output is genuinely streamed;
 * JXL is buffered.) */
struct jxl_writer {
	unsigned char *buf;   /* width*height*cmp, filled row by row */
	size_t stride;        /* width*cmp */
	int width, height, row;
	int ncolor, alpha_bits;
};

static int oil_cs_jxl_channels(enum oil_colorspace cs, int *ncolor,
	int *alpha_bits)
{
	switch (cs) {
	case OIL_CS_G:            *ncolor = 1; *alpha_bits = 0; return 0;
	case OIL_CS_GA:           *ncolor = 1; *alpha_bits = 8; return 0;
	case OIL_CS_RGB:
	case OIL_CS_RGB_NOGAMMA:  *ncolor = 3; *alpha_bits = 0; return 0;
	case OIL_CS_RGBA:
	case OIL_CS_RGBA_NOGAMMA: *ncolor = 3; *alpha_bits = 8; return 0;
	default:                  return -1;   /* e.g. CMYK, RGBX */
	}
}

/* Returns 0, or -1 if the colorspace can't be written as JXL / allocation
 * failed (the caller reports it). */
static int jxl_writer_open(struct jxl_writer *w, int width, int height,
	enum oil_colorspace cs)
{
	if (oil_cs_jxl_channels(cs, &w->ncolor, &w->alpha_bits) != 0)
		return -1;
	w->stride = (size_t)width * OIL_CMP(cs);
	w->width = width;
	w->height = height;
	w->row = 0;
	w->buf = malloc(w->stride * (size_t)height);
	return w->buf ? 0 : -1;
}

static void jxl_writer_row(struct jxl_writer *w, const unsigned char *row)
{
	memcpy(w->buf + (size_t)w->row * w->stride, row, w->stride);
	w->row++;
}

static void jxl_writer_free(struct jxl_writer *w)
{
	free(w->buf);
	w->buf = NULL;
}

/* Encode the accumulated buffer and write it to @output. Returns 0 on success.
 * Lossy at distance 1.0 ("visually lossless"), the JXL analog of the JPEG
 * writer's quality 94; XYB is left enabled (uses_original_profile = FALSE). */
static int jxl_writer_finish(struct jxl_writer *w, FILE *output)
{
	JxlEncoder *enc = NULL;
	void *runner = NULL;
	JxlEncoderFrameSettings *fs;
	JxlBasicInfo info;
	JxlColorEncoding color;
	JxlPixelFormat fmt;
	unsigned char *out = NULL, *next;
	size_t cap, used;
	int ret = -1;

	runner = JxlThreadParallelRunnerCreate(NULL,
		JxlThreadParallelRunnerDefaultNumWorkerThreads());
	enc = JxlEncoderCreate(NULL);
	if (!runner || !enc)
		goto done;
	if (JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner, runner)
			!= JXL_ENC_SUCCESS)
		goto done;

	JxlEncoderInitBasicInfo(&info);
	info.xsize = w->width;
	info.ysize = w->height;
	info.bits_per_sample = 8;
	info.exponent_bits_per_sample = 0;
	info.num_color_channels = w->ncolor;
	info.alpha_bits = w->alpha_bits;
	info.alpha_exponent_bits = 0;
	/* A plain alpha channel needs only num_extra_channels=1; the encoder reads
	 * it from the interleaved buffer (no SetExtraChannelInfo required). */
	info.num_extra_channels = w->alpha_bits ? 1 : 0;
	info.uses_original_profile = JXL_FALSE;
	if (JxlEncoderSetBasicInfo(enc, &info) != JXL_ENC_SUCCESS)
		goto done;

	JxlColorEncodingSetToSRGB(&color, w->ncolor == 1);
	if (JxlEncoderSetColorEncoding(enc, &color) != JXL_ENC_SUCCESS)
		goto done;

	fmt.num_channels = w->ncolor + (w->alpha_bits ? 1 : 0);
	fmt.data_type    = JXL_TYPE_UINT8;
	fmt.endianness   = JXL_NATIVE_ENDIAN;
	fmt.align        = 0;

	fs = JxlEncoderFrameSettingsCreate(enc, NULL);
	if (!fs)
		goto done;
	if (JxlEncoderSetFrameDistance(fs, 1.0) != JXL_ENC_SUCCESS)
		goto done;

	if (JxlEncoderAddImageFrame(fs, &fmt, w->buf,
			w->stride * (size_t)w->height) != JXL_ENC_SUCCESS)
		goto done;
	JxlEncoderCloseInput(enc);

	cap = 1 << 16;
	out = malloc(cap);
	if (!out)
		goto done;
	used = 0;
	for (;;) {
		JxlEncoderStatus s;
		size_t avail;
		next = out + used;
		avail = cap - used;
		s = JxlEncoderProcessOutput(enc, &next, &avail);
		used = cap - avail;
		if (s == JXL_ENC_SUCCESS)
			break;
		if (s != JXL_ENC_NEED_MORE_OUTPUT)
			goto done;
		{
			unsigned char *nb = realloc(out, cap * 2);
			if (!nb)
				goto done;
			out = nb;
			cap *= 2;
		}
	}

	if (fwrite(out, 1, used, output) == used)
		ret = 0;
done:
	free(out);
	if (enc) JxlEncoderDestroy(enc);
	if (runner) JxlThreadParallelRunnerDestroy(runner);
	return ret;
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
	struct jxl_writer jw = {0};
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
		else if (out_fmt == FMT_JXL)
			jxl_writer_free(&jw);
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
	} else if (out_fmt == FMT_JXL) {
		if (jxl_writer_open(&jw, width, height, ol.os.cs) != 0) {
			fprintf(stderr, "Cannot encode as JXL.\n");
			exit(1);
		}
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
		} else if (out_fmt == FMT_JXL) {
			jxl_writer_row(&jw, outbuf);
		} else {
			jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
		}
	}

	if (out_fmt == FMT_PNG) {
		png_write_end(wpng, winfo);
		png_destroy_write_struct(&wpng, &winfo);
	} else if (out_fmt == FMT_JXL) {
		if (jxl_writer_finish(&jw, output) != 0) {
			fprintf(stderr, "JXL encoding failed.\n");
			exit(1);
		}
		jxl_writer_free(&jw);
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
	unsigned char *outbuf, *inrow;
	int i, ret, out_ctype;
	long j;
	struct oil_libjpeg ol;
	struct jxl_writer jw = {0};

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

	/* Neither PNG nor JXL output carries CMYK here; convert to RGB on decode. */
	if (out_fmt != FMT_JPEG && dinfo.jpeg_color_space == JCS_CMYK) {
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
	inrow = malloc((size_t)ol.fed_width * ol.components);
	if (!outbuf || !inrow) {
		fprintf(stderr, "Unable to allocate buffers.");
		free(outbuf);
		free(inrow);
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
	} else if (out_fmt == FMT_JXL) {
		if (jxl_writer_open(&jw, width_out, height_out, ol.os.cs) != 0) {
			fprintf(stderr, "Cannot encode as JXL.\n");
			exit(1);
		}
	} else {
		jpeg_writer_open(&cinfo, &jerr, output, width_out, height_out,
			ol.os.cs, dinfo.marker_list);
	}

	for(i=height_out; i>0; i--) {
		while (oil_scale_slots(&ol.os) > 0) {
			oil_libjpeg_decode_row(&ol, inrow);
			be->scale_in(&ol.os, inrow);
		}
		be->scale_out(&ol.os, outbuf);
		if (out_fmt == FMT_PNG) {
			png_write_row(wpng, outbuf);
		} else if (out_fmt == FMT_JXL) {
			jxl_writer_row(&jw, outbuf);
		} else {
			jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
		}
	}

	if (out_fmt == FMT_PNG) {
		png_write_end(wpng, winfo);
		png_destroy_write_struct(&wpng, &winfo);
	} else if (out_fmt == FMT_JXL) {
		if (jxl_writer_finish(&jw, output) != 0) {
			fprintf(stderr, "JXL encoding failed.\n");
			exit(1);
		}
		jxl_writer_free(&jw);
	} else {
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);
	}

	jpeg_finish_decompress(&dinfo);
	jpeg_destroy_decompress(&dinfo);
	free(outbuf);
	free(inrow);
	oil_libjpeg_free(&ol);
}

/* Read an entire stream (regular file or pipe) into a malloc'd buffer. libjxl
 * has no incremental input API, so the wrapper needs the whole codestream. */
static unsigned char *slurp_stream(FILE *io, size_t *size_out)
{
	size_t cap = 1 << 16, len = 0;
	unsigned char *buf = malloc(cap);
	if (!buf) return NULL;
	for (;;) {
		size_t n;
		if (len == cap) {
			unsigned char *nb = realloc(buf, cap * 2);
			if (!nb) { free(buf); return NULL; }
			buf = nb;
			cap *= 2;
		}
		n = fread(buf + len, 1, cap - len, io);
		len += n;
		if (n == 0) break;
	}
	if (ferror(io)) { free(buf); return NULL; }
	*size_out = len;
	return buf;
}

/* Path-B reference consumer: libjxl is a parts kit, not a wrapper, so imgscale
 * composes the helpers itself -- a reorder buffer fed by a decode thread it
 * owns, pulled and scaled (streaming to the encoder) on the main thread. */
struct jxl_drive { JxlDecoder *dec; const JxlPixelFormat *fmt;
	struct oil_jxl_rowbuf *rb; };
static void *jxl_drive_thread(void *arg)
{
	struct jxl_drive *d = arg;
	oil_jxl_run_decode(d->dec, d->fmt, d->rb);
	return NULL;
}

static void process_jxl_input(FILE *input, FILE *output, int width_out,
	int height_out, const struct backend_entry *be, int no_gamma,
	enum file_format out_fmt)
{
	unsigned char *data, *zero, *outbuf;
	size_t insize, vpos = 0;
	void *runner;
	JxlDecoder *dec;
	JxlBasicInfo info;
	struct oil_scale os;
	struct oil_jxl_rowbuf *rb;
	struct oil_jxl_waiter *waiter;
	struct jxl_drive drv;
	pthread_t driver;
	JxlPixelFormat fmt;
	enum oil_colorspace cs;
	int cmp, fed_x, fed_y, fed_w, fed_h;
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	struct jxl_writer jw = {0};
	png_structp wpng = NULL;
	png_infop winfo = NULL;
	int i, out_ctype;

	data = slurp_stream(input, &insize);
	if (!data) {
		fprintf(stderr, "Unable to read input.\n");
		exit(1);
	}

	runner = JxlThreadParallelRunnerCreate(NULL,
		JxlThreadParallelRunnerDefaultNumWorkerThreads());
	dec = JxlDecoderCreate(NULL);
	if (!runner || !dec) {
		fprintf(stderr, "Unable to create JXL decoder.\n");
		exit(1);
	}

	/* oil wants straight alpha; SetUnpremultiplyAlpha must precede the first
	 * ProcessInput and is a no-op for images without associated alpha. */
	if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner)
			!= JXL_DEC_SUCCESS ||
	    JxlDecoderSubscribeEvents(dec,
			JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS ||
	    JxlDecoderSetUnpremultiplyAlpha(dec, JXL_TRUE) != JXL_DEC_SUCCESS) {
		fprintf(stderr, "JXL decoder setup failed.\n");
		exit(1);
	}
	JxlDecoderSetInput(dec, data, insize);
	JxlDecoderCloseInput(dec);

	if (JxlDecoderProcessInput(dec) != JXL_DEC_BASIC_INFO ||
	    JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
		fprintf(stderr, "JXL Decoding Error.\n");
		exit(1);
	}

	oil_fix_ratio(info.xsize, info.ysize, &width_out, &height_out);

	cs = jxl_cs_to_oil(&info);
	if (cs == OIL_CS_UNKNOWN) {
		fprintf(stderr, "Unsupported JXL colorspace.\n");
		exit(1);
	}
	if (no_gamma) cs = nogamma_cs(cs);
	cmp = OIL_CMP(cs);

	/* JPEG cannot carry alpha, and there is no flatten step. The no-gamma
	 * variant (RGBA -> RGBA_NOGAMMA) is already folded into cs above. */
	if (out_fmt == FMT_JPEG &&
	    (cs == OIL_CS_GA || cs == OIL_CS_RGBA || cs == OIL_CS_RGBA_NOGAMMA)) {
		fprintf(stderr, "Cannot encode alpha as JPEG.\n");
		exit(1);
	}

	/* Full-image source rect; oil_required_input_rect clamps the halo to the
	 * image bounds, so the fed rect is the whole image. */
	if (oil_required_input_rect(info.ysize, info.xsize,
		0.0, (double)info.ysize, 0.0, (double)info.xsize,
		height_out, width_out, &fed_y, &fed_h, &fed_x, &fed_w) < 0 ||
	    oil_scale_init_ex(&os, fed_h, height_out, fed_w, width_out,
		0.0 - fed_y, (double)info.ysize, 0.0 - fed_x, (double)info.xsize,
		cs) != 0) {
		fprintf(stderr, "Unable to initialize scaler.\n");
		exit(1);
	}

	zero = calloc((size_t)fed_w * cmp, 1);   /* fed on decode failure */
	outbuf = malloc((size_t)width_out * cmp);
	waiter = oil_jxl_condvar_waiter_create();
	rb = waiter ? oil_jxl_rowbuf_create(fed_x, fed_y, fed_w, fed_h, cmp, 256,
		waiter) : NULL;
	if (!zero || !outbuf || !waiter || !rb) {
		fprintf(stderr, "Unable to allocate buffers.\n");
		exit(1);
	}

	if (out_fmt == FMT_PNG) {
		out_ctype = oil_cs_to_png_ctype(cs);
		if (out_ctype < 0) {
			fprintf(stderr, "Cannot encode colorspace as PNG.\n");
			exit(1);
		}
		png_writer_open(&wpng, &winfo, output, width_out, height_out, out_ctype);
	} else if (out_fmt == FMT_JXL) {
		if (jxl_writer_open(&jw, width_out, height_out, cs) != 0) {
			fprintf(stderr, "Cannot encode as JXL.\n");
			exit(1);
		}
	} else {
		jpeg_writer_open(&cinfo, &jerr, output, width_out, height_out,
			cs, NULL);
	}

	/* Spawn the decode driver; pull + scale its rows as they arrive. */
	fmt.num_channels = cmp;
	fmt.data_type    = JXL_TYPE_UINT8;
	fmt.endianness   = JXL_NATIVE_ENDIAN;
	fmt.align        = 0;
	drv.dec = dec; drv.fmt = &fmt; drv.rb = rb;
	if (pthread_create(&driver, NULL, jxl_drive_thread, &drv) != 0) {
		fprintf(stderr, "Unable to start JXL decode thread.\n");
		exit(1);
	}

	for (i = 0; i < height_out; i++) {
		while (oil_scale_slots(&os) > 0) {
			unsigned char *row = oil_jxl_rowbuf_wait_row(rb, vpos);
			be->scale_in(&os, row ? row : zero);
			if (row)
				oil_jxl_rowbuf_release_row(rb, vpos);
			vpos++;
		}
		be->scale_out(&os, outbuf);
		if (out_fmt == FMT_PNG) {
			png_write_row(wpng, outbuf);
		} else if (out_fmt == FMT_JXL) {
			jxl_writer_row(&jw, outbuf);
		} else {
			jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&outbuf, 1);
		}
	}

	if (out_fmt == FMT_PNG) {
		png_write_end(wpng, winfo);
		png_destroy_write_struct(&wpng, &winfo);
	} else if (out_fmt == FMT_JXL) {
		if (jxl_writer_finish(&jw, output) != 0) {
			fprintf(stderr, "JXL encoding failed.\n");
			exit(1);
		}
		jxl_writer_free(&jw);
	} else {
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);
	}

	/* Release any back-pressure-parked producer so the decode thread can join
	 * even if the scaler consumed fewer fed rows than were buffered. */
	oil_jxl_rowbuf_abort(rb);
	pthread_join(driver, NULL);
	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	oil_scale_free(&os);
	JxlDecoderDestroy(dec);
	JxlThreadParallelRunnerDestroy(runner);
	free(data);
	free(zero);
	free(outbuf);
}

/* Sniff the input format from its leading bytes without consuming them: PNG
 * starts 0x89, a JXL container 0x00, a raw JXL codestream 0xFF 0x0A, and JPEG
 * 0xFF 0xD8. Anything else falls back to JPEG. */
static enum file_format detect_input_format(FILE *io)
{
	int b0, b1;
	b0 = getc(io);
	if (b0 == 0x89) { ungetc(b0, io); return FMT_PNG; }
	if (b0 == 0x00) { ungetc(b0, io); return FMT_JXL; }
	if (b0 == 0xFF) {
		b1 = getc(io);
		if (b1 != EOF) ungetc(b1, io);
		ungetc(b0, io);
		return b1 == 0x0A ? FMT_JXL : FMT_JPEG;
	}
	if (b0 != EOF) ungetc(b0, io);
	return FMT_JPEG;
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

	input_fmt = detect_input_format(io_in);
	output_fmt = detect_out_format(pos[3], input_fmt);

	if (input_fmt == FMT_PNG) {
		process_png_input(io_in, io_out, width, height, be, no_gamma, output_fmt);
	} else if (input_fmt == FMT_JXL) {
		process_jxl_input(io_in, io_out, width, height, be, no_gamma, output_fmt);
	} else {
		process_jpeg_input(io_in, io_out, width, height, be, no_gamma, output_fmt);
	}

	if (io_in != stdin)
		fclose(io_in);
	if (io_out != stdout)
		fclose(io_out);
	return 0;
}
