#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <SDL3/SDL.h>
#include <jpeglib.h>
#include <png.h>

#include "oil_resample.h"
#include "oil_libjpeg.h"
#include "oil_libpng.h"

#define LINE_QUEUE_DEPTH 16
#define MIN_DRAG_PIXELS 5.0f

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

struct resumable_resize {
	FILE *io;
	int is_png;
	int png_interlaced;

	int surface_width;
	int surface_height;

	int img_width;
	int img_height;

	/* Logical source rect within the full image (may be fractional). */
	double src_x;
	double src_y;
	double src_w;
	double src_h;

	/* Fed input rect: integer-pixel sub-image inside the full image that
	 * the decoder will read and hand to the scaler. */
	int fed_x;
	int fed_y;
	int fed_width;
	int fed_height;

	int out_width;
	int out_height;

	int cmp;
	enum oil_colorspace cs;
	int in_rowbytes;
	int slot_rowbytes;

	int threaded;
	int no_gamma;
	int n_slots;

	unsigned char *outbuf;
	unsigned char *scaled_buf;
	unsigned char *scratch;
	unsigned char **inimage;

	struct oil_scale os;
	int (*scale_in)(struct oil_scale *, unsigned char *);
	int (*scale_out)(struct oil_scale *, unsigned char *);
	void (*decode_row)(struct resumable_resize *, unsigned char *, int);
	void (*format_end)(struct resumable_resize *);

	struct jpeg_decompress_struct *dinfo;
	png_structp rpng;
	png_infop rinfo;

	SDL_Thread *decoder_thread;
	SDL_Thread *scaler_thread;
	SDL_Thread *worker_thread;
	SDL_Mutex *mutex;
	SDL_Condition *cv;
	unsigned char *slots[LINE_QUEUE_DEPTH];
	int head;
	int tail;
	int count;
	int ypos;
	int decoder_done;
	int scaler_done;
	int aborted;
};

static void translate(unsigned char *in, unsigned char *out, int width, int cmp) {
	int i;
	for (i=0; i<width; i++) {
		if (cmp <= 2) {
			out[0] = out[1] = out[2] = in[0];
			out[3] = cmp == 2 ? in[1] : 0xFF;
		} else {
			out[0] = in[2];
			out[1] = in[1];
			out[2] = in[0];
			out[3] = cmp >= 4 ? in[3] : 0xFF;
		}
		in += cmp;
		out += 4;
	}
}

static int looks_like_png(FILE *io)
{
	int peek;
	peek = getc(io);
	ungetc(peek, io);
	return peek == 137;
}

static void decode_png_interlaced(struct resumable_resize *rr, unsigned char *slot, int row) {
	memcpy(slot, rr->inimage[rr->fed_y + row] + rr->fed_x * rr->cmp, rr->slot_rowbytes);
}

static void decode_png_streaming(struct resumable_resize *rr, unsigned char *slot, int row) {
	(void)row;
	png_read_row(rr->rpng, rr->scratch, NULL);
	memcpy(slot, rr->scratch + rr->fed_x * rr->cmp, rr->slot_rowbytes);
}

static void decode_jpeg_row(struct resumable_resize *rr, unsigned char *slot, int row) {
	(void)row;
	jpeg_read_scanlines(rr->dinfo, &rr->scratch, 1);
	memcpy(slot, rr->scratch + rr->fed_x * rr->cmp, rr->slot_rowbytes);
}

static unsigned char **alloc_image(int height, int rowbytes) {
	unsigned char **img;
	int i, j;
	img = malloc((size_t)height * sizeof(unsigned char *));
	if (!img) return NULL;
	for (i=0; i<height; i++) {
		img[i] = malloc(rowbytes);
		if (!img[i]) {
			for (j=0; j<i; j++) free(img[j]);
			free(img);
			return NULL;
		}
	}
	return img;
}

static void free_image(unsigned char **img, int height) {
	int i;
	if (!img) return;
	for (i=0; i<height; i++) free(img[i]);
	free(img);
}

static void clamp_crop(struct resumable_resize *rr) {
	if (rr->src_w <= 0 || rr->src_h <= 0) {
		rr->src_x = 0;
		rr->src_y = 0;
		rr->src_w = rr->img_width;
		rr->src_h = rr->img_height;
		return;
	}
	if (rr->src_x < 0) {
		rr->src_w += rr->src_x;
		rr->src_x = 0;
	}
	if (rr->src_y < 0) {
		rr->src_h += rr->src_y;
		rr->src_y = 0;
	}
	if (rr->src_x + rr->src_w > rr->img_width) rr->src_w = rr->img_width - rr->src_x;
	if (rr->src_y + rr->src_h > rr->img_height) rr->src_h = rr->img_height - rr->src_y;
}

static int compute_fed_and_out(struct resumable_resize *rr) {
	int src_iw, src_ih;
	src_iw = (int)(rr->src_w + 0.5);
	src_ih = (int)(rr->src_h + 0.5);
	if (src_iw < 1) src_iw = 1;
	if (src_ih < 1) src_ih = 1;
	rr->out_width = rr->surface_width;
	rr->out_height = rr->surface_height;
	if (oil_fix_ratio(src_iw, src_ih, &rr->out_width, &rr->out_height) < 0) return -1;
	if (rr->out_width < 1) rr->out_width = 1;
	if (rr->out_height < 1) rr->out_height = 1;

	return oil_required_input_rect(rr->img_height, rr->img_width,
		rr->src_y, rr->src_h, rr->src_x, rr->src_w,
		rr->out_height, rr->out_width,
		&rr->fed_y, &rr->fed_height, &rr->fed_x, &rr->fed_width);
}

static int init_scaler(struct resumable_resize *rr) {
	return oil_scale_init_ex(&rr->os, rr->fed_height, rr->out_height,
		rr->fed_width, rr->out_width,
		rr->src_y - rr->fed_y, rr->src_h,
		rr->src_x - rr->fed_x, rr->src_w,
		rr->cs);
}

static void png_end(struct resumable_resize *rr);
static void jpeg_end(struct resumable_resize *rr);

static int png_start(struct resumable_resize *rr) {
	png_structp rpng;
	png_infop rinfo;
	int i;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!rpng) {
		return -1;
	}

	rinfo = png_create_info_struct(rpng);
	if (!rinfo) {
		png_destroy_read_struct(&rpng, NULL, NULL);
		return -1;
	}

	if (setjmp(png_jmpbuf(rpng))) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}

	rr->rpng = rpng;
	rr->rinfo = rinfo;

	png_init_io(rpng, rr->io);
	png_read_info(rpng, rinfo);

	png_set_packing(rpng);
	png_set_strip_16(rpng);
	png_set_expand(rpng);
	png_set_interlace_handling(rpng);
	png_read_update_info(rpng, rinfo);

	rr->img_width = png_get_image_width(rpng, rinfo);
	rr->img_height = png_get_image_height(rpng, rinfo);
	rr->cs = png_cs_to_oil(png_get_color_type(rpng, rinfo));
	if (rr->cs == OIL_CS_UNKNOWN) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}
	if (rr->no_gamma) rr->cs = nogamma_cs(rr->cs);
	rr->cmp = OIL_CMP(rr->cs);
	rr->in_rowbytes = png_get_rowbytes(rpng, rinfo);
	rr->png_interlaced = (png_get_interlace_type(rpng, rinfo) == PNG_INTERLACE_ADAM7);

	clamp_crop(rr);
	if (rr->src_w < 1 || rr->src_h < 1) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}
	if (compute_fed_and_out(rr) < 0) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}
	rr->slot_rowbytes = rr->fed_width * rr->cmp;

	if (init_scaler(rr) != 0) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}

	rr->outbuf = malloc((size_t)rr->out_width * rr->cmp);
	if (!rr->outbuf) goto fail_os;
	rr->scratch = malloc(rr->in_rowbytes);
	if (!rr->scratch) goto fail_outbuf;

	if (rr->png_interlaced) {
		rr->inimage = alloc_image(rr->img_height, rr->in_rowbytes);
		if (!rr->inimage) goto fail_scratch;
		png_read_image(rpng, rr->inimage);
		rr->decode_row = decode_png_interlaced;
	} else {
		for (i = 0; i < rr->fed_y; i++) {
			png_read_row(rpng, rr->scratch, NULL);
		}
		rr->decode_row = decode_png_streaming;
	}

	rr->format_end = png_end;
	return 0;

fail_scratch:
	free(rr->scratch);
fail_outbuf:
	free(rr->outbuf);
fail_os:
	oil_scale_free(&rr->os);
	png_destroy_read_struct(&rpng, &rinfo, NULL);
	return -1;
}

static void png_end(struct resumable_resize *rr) {
	if (rr->inimage) free_image(rr->inimage, rr->img_height);
	free(rr->scratch);
	free(rr->outbuf);
	oil_scale_free(&rr->os);
	png_destroy_read_struct(&rr->rpng, &rr->rinfo, NULL);
}

struct jpeg_err {
	struct jpeg_error_mgr mgr;
	jmp_buf jmpbuf;
};

static void jpeg_err_exit(j_common_ptr cinfo)
{
	struct jpeg_err *err = (struct jpeg_err *)cinfo->err;
	longjmp(err->jmpbuf, 1);
}

static int jpeg_start(struct resumable_resize *rr)
{
	struct jpeg_decompress_struct *dinfo;
	struct jpeg_err *jerr;
	int i;

	dinfo = malloc(sizeof(struct jpeg_decompress_struct));
	if (!dinfo) {
		return -1;
	}
	jerr = malloc(sizeof(struct jpeg_err));
	if (!jerr) {
		free(dinfo);
		return -1;
	}

	rr->dinfo = dinfo;
	dinfo->err = jpeg_std_error(&jerr->mgr);
	jerr->mgr.error_exit = jpeg_err_exit;
	jpeg_create_decompress(dinfo);

	if (setjmp(jerr->jmpbuf)) {
		jpeg_destroy_decompress(dinfo);
		free(jerr);
		free(dinfo);
		return -1;
	}

	jpeg_stdio_src(dinfo, rr->io);
	jpeg_read_header(dinfo, TRUE);
	jpeg_calc_output_dimensions(dinfo);

	jpeg_start_decompress(dinfo);

	rr->img_width = dinfo->output_width;
	rr->img_height = dinfo->output_height;
	rr->cs = jpeg_cs_to_oil(dinfo->out_color_space);
	if (rr->cs == OIL_CS_UNKNOWN) goto fail_jpeg;
	if (rr->no_gamma) rr->cs = nogamma_cs(rr->cs);
	rr->cmp = dinfo->output_components;
	rr->in_rowbytes = dinfo->output_width * dinfo->output_components;

	clamp_crop(rr);
	if (rr->src_w < 1 || rr->src_h < 1) goto fail_jpeg;
	if (compute_fed_and_out(rr) < 0) goto fail_jpeg;
	rr->slot_rowbytes = rr->fed_width * rr->cmp;

	if (init_scaler(rr) != 0) goto fail_jpeg;

	rr->outbuf = malloc((size_t)rr->out_width * rr->cmp);
	if (!rr->outbuf) goto fail_os;
	rr->scratch = malloc(rr->in_rowbytes);
	if (!rr->scratch) goto fail_outbuf;

	for (i = 0; i < rr->fed_y; i++) {
		jpeg_read_scanlines(dinfo, &rr->scratch, 1);
	}

	rr->decode_row = decode_jpeg_row;
	rr->format_end = jpeg_end;
	return 0;

fail_outbuf:
	free(rr->outbuf);
fail_os:
	oil_scale_free(&rr->os);
fail_jpeg:
	jpeg_destroy_decompress(dinfo);
	free(jerr);
	free(dinfo);
	return -1;
}

static void jpeg_end(struct resumable_resize *rr) {
	free(rr->scratch);
	free(rr->outbuf);
	oil_scale_free(&rr->os);
	free(rr->dinfo->err);
	jpeg_destroy_decompress(rr->dinfo);
	free(rr->dinfo);
}

static int decoder_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	int row;

	for (row = 0; row < rr->fed_height; row++) {
		unsigned char *slot;

		SDL_LockMutex(rr->mutex);
		while (rr->count == LINE_QUEUE_DEPTH && !rr->aborted) {
			SDL_WaitCondition(rr->cv, rr->mutex);
		}
		if (rr->aborted) {
			SDL_UnlockMutex(rr->mutex);
			return 0;
		}
		slot = rr->slots[rr->head];
		SDL_UnlockMutex(rr->mutex);

		rr->decode_row(rr, slot, row);

		SDL_LockMutex(rr->mutex);
		rr->head = (rr->head + 1) % LINE_QUEUE_DEPTH;
		rr->count++;
		SDL_SignalCondition(rr->cv);
		SDL_UnlockMutex(rr->mutex);
	}

	SDL_LockMutex(rr->mutex);
	rr->decoder_done = 1;
	SDL_SignalCondition(rr->cv);
	SDL_UnlockMutex(rr->mutex);
	return 0;
}

static int scaler_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	int local_ypos = 0;

	while (local_ypos < rr->out_height) {
		unsigned char *tmp;

		while (oil_scale_slots(&rr->os) > 0) {
			unsigned char *slot;

			SDL_LockMutex(rr->mutex);
			while (rr->count == 0 && !rr->decoder_done && !rr->aborted) {
				SDL_WaitCondition(rr->cv, rr->mutex);
			}
			if (rr->aborted || rr->count == 0) {
				SDL_UnlockMutex(rr->mutex);
				goto done;
			}
			slot = rr->slots[rr->tail];
			SDL_UnlockMutex(rr->mutex);

			rr->scale_in(&rr->os, slot);

			SDL_LockMutex(rr->mutex);
			rr->tail = (rr->tail + 1) % LINE_QUEUE_DEPTH;
			rr->count--;
			SDL_SignalCondition(rr->cv);
			SDL_UnlockMutex(rr->mutex);
		}

		rr->scale_out(&rr->os, rr->outbuf);
		tmp = rr->scaled_buf + local_ypos * rr->out_width * 4;
		translate(rr->outbuf, tmp, rr->out_width, rr->cmp);
		local_ypos++;

		SDL_LockMutex(rr->mutex);
		rr->ypos = local_ypos;
		SDL_UnlockMutex(rr->mutex);
	}

done:
	SDL_LockMutex(rr->mutex);
	rr->scaler_done = 1;
	SDL_UnlockMutex(rr->mutex);
	return 0;
}

static int worker_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	unsigned char *inbuf = rr->slots[0];
	int local_ypos = 0;
	int row = 0;
	int aborted;

	while (local_ypos < rr->out_height) {
		unsigned char *tmp;

		SDL_LockMutex(rr->mutex);
		aborted = rr->aborted;
		SDL_UnlockMutex(rr->mutex);
		if (aborted) goto done;

		while (oil_scale_slots(&rr->os) > 0) {
			rr->decode_row(rr, inbuf, row++);
			rr->scale_in(&rr->os, inbuf);
		}

		rr->scale_out(&rr->os, rr->outbuf);
		tmp = rr->scaled_buf + local_ypos * rr->out_width * 4;
		translate(rr->outbuf, tmp, rr->out_width, rr->cmp);
		local_ypos++;

		SDL_LockMutex(rr->mutex);
		rr->ypos = local_ypos;
		SDL_UnlockMutex(rr->mutex);
	}

done:
	SDL_LockMutex(rr->mutex);
	rr->scaler_done = 1;
	SDL_UnlockMutex(rr->mutex);
	return 0;
}

static int resumable_resize_start(struct resumable_resize *rr, char *path,
                                  int surface_width, int surface_height,
                                  const struct backend_entry *be, int threaded,
                                  int no_gamma,
                                  double src_x, double src_y,
                                  double src_w, double src_h)
{
	int i;
	int is_png;

	memset(rr, 0, sizeof(*rr));
	rr->surface_width = surface_width;
	rr->surface_height = surface_height;
	rr->scale_in = be->scale_in;
	rr->scale_out = be->scale_out;
	rr->threaded = threaded;
	rr->no_gamma = no_gamma;
	rr->n_slots = threaded ? LINE_QUEUE_DEPTH : 1;
	rr->src_x = src_x;
	rr->src_y = src_y;
	rr->src_w = src_w;
	rr->src_h = src_h;

	rr->io = fopen(path, "r");
	if (!rr->io) {
		fprintf(stderr, "Error: unable to open %s\n", path);
		return -1;
	}
	is_png = looks_like_png(rr->io);

	if (is_png ? png_start(rr) : jpeg_start(rr)) {
		goto fail_io;
	}

	rr->scaled_buf = malloc((size_t)rr->out_width * rr->out_height * 4);
	if (!rr->scaled_buf) goto fail_decoder;

	for (i = 0; i < rr->n_slots; i++) {
		rr->slots[i] = malloc(rr->slot_rowbytes);
		if (!rr->slots[i]) goto fail_slots;
	}

	rr->mutex = SDL_CreateMutex();
	rr->cv = SDL_CreateCondition();
	if (!rr->mutex || !rr->cv) {
		fprintf(stderr, "Error: SDL sync init failed: %s\n", SDL_GetError());
		goto fail_sync;
	}

	if (rr->threaded) {
		rr->decoder_thread = SDL_CreateThread(decoder_thread_fn, "oil-decoder", rr);
		rr->scaler_thread = SDL_CreateThread(scaler_thread_fn, "oil-scaler", rr);
		if (!rr->decoder_thread || !rr->scaler_thread) {
			fprintf(stderr, "Error: SDL_CreateThread failed: %s\n", SDL_GetError());
			SDL_LockMutex(rr->mutex);
			rr->aborted = 1;
			SDL_BroadcastCondition(rr->cv);
			SDL_UnlockMutex(rr->mutex);
			SDL_WaitThread(rr->decoder_thread, NULL);
			SDL_WaitThread(rr->scaler_thread, NULL);
			goto fail_sync;
		}
	} else {
		rr->worker_thread = SDL_CreateThread(worker_thread_fn, "oil-worker", rr);
		if (!rr->worker_thread) {
			fprintf(stderr, "Error: SDL_CreateThread failed: %s\n", SDL_GetError());
			goto fail_sync;
		}
	}

	return 0;

fail_sync:
	SDL_DestroyMutex(rr->mutex);
	SDL_DestroyCondition(rr->cv);
fail_slots:
	for (i = 0; i < rr->n_slots; i++) free(rr->slots[i]);
	free(rr->scaled_buf);
fail_decoder:
	rr->format_end(rr);
fail_io:
	fclose(rr->io);
	return -1;
}

static void resumable_resize_end(struct resumable_resize *rr)
{
	int i;

	SDL_LockMutex(rr->mutex);
	rr->aborted = 1;
	SDL_BroadcastCondition(rr->cv);
	SDL_UnlockMutex(rr->mutex);

	if (rr->threaded) {
		SDL_WaitThread(rr->decoder_thread, NULL);
		SDL_WaitThread(rr->scaler_thread, NULL);
	} else {
		SDL_WaitThread(rr->worker_thread, NULL);
	}

	SDL_DestroyMutex(rr->mutex);
	SDL_DestroyCondition(rr->cv);

	for (i = 0; i < rr->n_slots; i++) {
		free(rr->slots[i]);
	}
	free(rr->scaled_buf);

	rr->format_end(rr);
	fclose(rr->io);
}

static void compute_letterbox(int win_w, int win_h, int src_w, int src_h, SDL_FRect *dst)
{
	float sx = (float)win_w / src_w;
	float sy = (float)win_h / src_h;
	float scale = sx < sy ? sx : sy;
	dst->w = src_w * scale;
	dst->h = src_h * scale;
	dst->x = (win_w - dst->w) / 2.0f;
	dst->y = (win_h - dst->h) / 2.0f;
}

static void present_frame(SDL_Renderer *renderer, SDL_Texture *display_tex,
                          const SDL_FRect *overlay)
{
	int win_w, win_h;
	SDL_GetRenderOutputSize(renderer, &win_w, &win_h);

	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);

	if (display_tex) {
		float tex_w, tex_h;
		SDL_FRect dst;
		SDL_GetTextureSize(display_tex, &tex_w, &tex_h);
		compute_letterbox(win_w, win_h, (int)tex_w, (int)tex_h, &dst);
		SDL_RenderTexture(renderer, display_tex, NULL, &dst);
	}

	if (overlay) {
		SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
		SDL_RenderRect(renderer, overlay);
	}

	SDL_RenderPresent(renderer);
}

static void update_texture_rows(SDL_Texture *tex, struct resumable_resize *rr, int y0, int y1)
{
	SDL_Rect rect;
	rect.x = 0;
	rect.y = y0;
	rect.w = rr->out_width;
	rect.h = y1 - y0;
	SDL_UpdateTexture(tex, &rect,
		rr->scaled_buf + y0 * rr->out_width * 4,
		rr->out_width * 4);
}

static SDL_Texture *create_blank_texture(SDL_Renderer *renderer, int w, int h)
{
	SDL_Texture *tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING, w, h);
	if (tex) {
		void *pixels;
		int pitch;
		if (SDL_LockTexture(tex, NULL, &pixels, &pitch)) {
			memset(pixels, 0, (size_t)pitch * h);
			SDL_UnlockTexture(tex);
		}
	}
	return tex;
}

static int start_resize_session(struct resumable_resize *rr, char *path,
                                SDL_Renderer *renderer, SDL_Texture **display_tex,
                                const struct backend_entry *be, int threaded,
                                int no_gamma,
                                double src_x, double src_y,
                                double src_w, double src_h)
{
	int rw, rh;
	SDL_GetRenderOutputSize(renderer, &rw, &rh);
	if (resumable_resize_start(rr, path, rw, rh, be, threaded, no_gamma,
	                           src_x, src_y, src_w, src_h) < 0) return -1;
	if (*display_tex) SDL_DestroyTexture(*display_tex);
	*display_tex = create_blank_texture(renderer, rr->out_width, rr->out_height);
	return 0;
}

static SDL_FRect make_drag_rect(float ax, float ay, float bx, float by)
{
	SDL_FRect r;
	r.x = ax < bx ? ax : bx;
	r.y = ay < by ? ay : by;
	r.w = fabsf(bx - ax);
	r.h = fabsf(by - ay);
	return r;
}

#define STASH_DEPTH 16

struct view_stash {
	SDL_Texture *tex;
	double crop_x;
	double crop_y;
	double crop_w;
	double crop_h;
};

static void stash_clear(struct view_stash *s, int *count)
{
	int i;
	for (i = 0; i < *count; i++) {
		if (s[i].tex) SDL_DestroyTexture(s[i].tex);
	}
	*count = 0;
}

static void stash_invalidate_textures(struct view_stash *s, int count)
{
	int i;
	for (i = 0; i < count; i++) {
		if (s[i].tex) {
			SDL_DestroyTexture(s[i].tex);
			s[i].tex = NULL;
		}
	}
}

static void stash_push(struct view_stash *s, int *count, SDL_Texture *tex,
                       double cx, double cy, double cw, double ch)
{
	if (*count == STASH_DEPTH) {
		SDL_DestroyTexture(s[0].tex);
		memmove(&s[0], &s[1], (STASH_DEPTH - 1) * sizeof(*s));
		(*count)--;
	}
	s[*count].tex = tex;
	s[*count].crop_x = cx;
	s[*count].crop_y = cy;
	s[*count].crop_w = cw;
	s[*count].crop_h = ch;
	(*count)++;
}

/* Map a render-coord rect to a logical source rect inside the current crop,
 * clipped to the visible image area. Returns 0 on success, -1 if the rect
 * collapses to nothing inside the image. */
static int drag_rect_to_crop(SDL_Renderer *renderer, SDL_Texture *display_tex,
                             SDL_FRect drag, double crop_x, double crop_y,
                             double crop_w, double crop_h,
                             double *out_x, double *out_y,
                             double *out_w, double *out_h)
{
	int win_w, win_h;
	float tex_w, tex_h;
	SDL_FRect dst;
	double sx, sy, sw, sh;

	if (!display_tex) return -1;
	SDL_GetRenderOutputSize(renderer, &win_w, &win_h);
	SDL_GetTextureSize(display_tex, &tex_w, &tex_h);
	compute_letterbox(win_w, win_h, (int)tex_w, (int)tex_h, &dst);

	if (drag.x < dst.x) { drag.w -= dst.x - drag.x; drag.x = dst.x; }
	if (drag.y < dst.y) { drag.h -= dst.y - drag.y; drag.y = dst.y; }
	if (drag.x + drag.w > dst.x + dst.w) drag.w = dst.x + dst.w - drag.x;
	if (drag.y + drag.h > dst.y + dst.h) drag.h = dst.y + dst.h - drag.y;
	if (drag.w <= 0 || drag.h <= 0 || dst.w <= 0 || dst.h <= 0) return -1;

	sx = (drag.x - dst.x) / dst.w;
	sy = (drag.y - dst.y) / dst.h;
	sw = drag.w / dst.w;
	sh = drag.h / dst.h;
	*out_x = crop_x + sx * crop_w;
	*out_y = crop_y + sy * crop_h;
	*out_w = sw * crop_w;
	*out_h = sh * crop_h;
	return (*out_w >= 1.0 && *out_h >= 1.0) ? 0 : -1;
}

int main(int argc, char **argv) {
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *display_tex = NULL;
	SDL_Event event;
	char *path;
	int event_happened, render_in_progress;
	int last_displayed_ypos;
	struct resumable_resize rr;
	Uint64 resize_start_time, elapsed_time;
	int resize_pending = 0;
	int argi;
	int threaded = 1;
	int no_gamma = 0;
	const struct backend_entry *be = NULL;
	int img_w = 0;
	double crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
	int drag_active = 0;
	float drag_start_x = 0, drag_start_y = 0;
	float drag_cur_x = 0, drag_cur_y = 0;
	int need_present;
	struct view_stash stash[STASH_DEPTH];
	int stash_count = 0;

	path = NULL;
	for (argi = 1; argi < argc; argi++) {
		if (strcmp(argv[argi], "--no-threaded") == 0) {
			threaded = 0;
		} else if (strcmp(argv[argi], "--no-gamma") == 0) {
			no_gamma = 1;
		} else if (argv[argi][0] == '-' && argv[argi][1] == '-') {
			be = find_backend(argv[argi]);
			if (!be) {
				fprintf(stderr, "Error: unknown or unavailable backend: %s\n", argv[argi]);
				return 1;
			}
		} else {
			path = argv[argi];
		}
	}
	if (!path) {
		fprintf(stderr, "Usage: %s [--scalar|--sse2|--avx2|--neon] [--no-threaded] [--no-gamma] <image>\n", argv[0]);
		return 1;
	}
	if (!be) be = default_backend();

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}
	window = SDL_CreateWindow(path, 640, 480, SDL_WINDOW_RESIZABLE);
	if (!window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		SDL_Quit();
		return 1;
	}
	renderer = SDL_CreateRenderer(window, NULL);
	if (!renderer) {
		fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}

	last_displayed_ypos = 0;
	resize_start_time = SDL_GetTicks();
	if (start_resize_session(&rr, path, renderer, &display_tex, be, threaded, no_gamma,
	                         crop_x, crop_y, crop_w, crop_h) < 0) {
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}
	render_in_progress = 1;
	img_w = rr.img_width;
	crop_x = rr.src_x;
	crop_y = rr.src_y;
	crop_w = rr.src_w;
	crop_h = rr.src_h;

	while (1) {
		int timeout = render_in_progress ? 16 : -1;
		event_happened = SDL_WaitEventTimeout(&event, timeout);
		need_present = 0;

		if (event_happened) {
			if (event.type == SDL_EVENT_QUIT) {
				if (render_in_progress) {
					resumable_resize_end(&rr);
				}
				stash_clear(stash, &stash_count);
				SDL_DestroyTexture(display_tex);
				SDL_DestroyRenderer(renderer);
				SDL_DestroyWindow(window);
				SDL_Quit();
				return 0;
			}
			if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
			    (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F5)) {
				if (render_in_progress) {
					resumable_resize_end(&rr);
					render_in_progress = 0;
				}
				drag_active = 0;
				if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
					stash_invalidate_textures(stash, stash_count);
					present_frame(renderer, display_tex, NULL);
				}
				resize_pending = 1;
			}
			if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
			    event.button.button == SDL_BUTTON_LEFT) {
				float rx, ry;
				SDL_RenderCoordinatesFromWindow(renderer,
					event.button.x, event.button.y, &rx, &ry);
				drag_start_x = drag_cur_x = rx;
				drag_start_y = drag_cur_y = ry;
				drag_active = 1;
				need_present = 1;
			}
			if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
			    event.button.button == SDL_BUTTON_RIGHT) {
				drag_active = 0;
				if (stash_count > 0) {
					if (render_in_progress) {
						resumable_resize_end(&rr);
						render_in_progress = 0;
					}
					stash_count--;
					crop_x = stash[stash_count].crop_x;
					crop_y = stash[stash_count].crop_y;
					crop_w = stash[stash_count].crop_w;
					crop_h = stash[stash_count].crop_h;
					if (stash[stash_count].tex) {
						SDL_DestroyTexture(display_tex);
						display_tex = stash[stash_count].tex;
						need_present = 1;
					} else {
						resize_pending = 1;
					}
				}
			}
			if (event.type == SDL_EVENT_MOUSE_MOTION && drag_active) {
				float rx, ry;
				SDL_RenderCoordinatesFromWindow(renderer,
					event.motion.x, event.motion.y, &rx, &ry);
				drag_cur_x = rx;
				drag_cur_y = ry;
				need_present = 1;
			}
			if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
			    event.button.button == SDL_BUTTON_LEFT && drag_active) {
				float rx, ry;
				SDL_FRect drag;
				double nx, ny, nw, nh;
				drag_active = 0;
				SDL_RenderCoordinatesFromWindow(renderer,
					event.button.x, event.button.y, &rx, &ry);
				drag_cur_x = rx;
				drag_cur_y = ry;
				drag = make_drag_rect(drag_start_x, drag_start_y, drag_cur_x, drag_cur_y);
				if (drag.w >= MIN_DRAG_PIXELS && drag.h >= MIN_DRAG_PIXELS &&
				    img_w > 0 &&
				    drag_rect_to_crop(renderer, display_tex, drag,
				                      crop_x, crop_y, crop_w, crop_h,
				                      &nx, &ny, &nw, &nh) == 0) {
					if (render_in_progress) {
						resumable_resize_end(&rr);
						render_in_progress = 0;
					}
					stash_push(stash, &stash_count, display_tex,
					           crop_x, crop_y, crop_w, crop_h);
					display_tex = NULL;
					crop_x = nx;
					crop_y = ny;
					crop_w = nw;
					crop_h = nh;
					resize_pending = 1;
				}
				need_present = 1;
			}
		}

		if (resize_pending && !render_in_progress &&
		    SDL_GetMouseState(NULL, NULL) == 0) {
			resize_pending = 0;
			last_displayed_ypos = 0;
			resize_start_time = SDL_GetTicks();
			if (start_resize_session(&rr, path, renderer, &display_tex, be, threaded, no_gamma,
			                         crop_x, crop_y, crop_w, crop_h) == 0) {
				render_in_progress = 1;
				if (img_w == 0) {
					img_w = rr.img_width;
				}
				crop_x = rr.src_x;
				crop_y = rr.src_y;
				crop_w = rr.src_w;
				crop_h = rr.src_h;
			}
		}

		if (render_in_progress) {
			int local_ypos, local_done;

			SDL_LockMutex(rr.mutex);
			local_ypos = rr.ypos;
			local_done = rr.scaler_done;
			SDL_UnlockMutex(rr.mutex);

			if (local_done) {
				render_in_progress = 0;
				if (last_displayed_ypos < rr.out_height) {
					update_texture_rows(display_tex, &rr, last_displayed_ypos, rr.out_height);
				}
				resumable_resize_end(&rr);
				need_present = 1;
				elapsed_time = SDL_GetTicks() - resize_start_time;
				fprintf(stderr, "Resize ticks: %llu\n", (unsigned long long)elapsed_time);
			} else if (local_ypos > last_displayed_ypos) {
				update_texture_rows(display_tex, &rr, last_displayed_ypos, local_ypos);
				last_displayed_ypos = local_ypos;
				need_present = 1;
			}
		}

		if (need_present) {
			SDL_FRect drag, *overlay = NULL;
			if (drag_active) {
				drag = make_drag_rect(drag_start_x, drag_start_y, drag_cur_x, drag_cur_y);
				overlay = &drag;
			}
			present_frame(renderer, display_tex, overlay);
		}
	}
}
