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
#define STASH_DEPTH 16
#define PNG_FIRST_MAGIC_BYTE 0x89  /* first byte of the 8-byte PNG signature */

/* ===========================================================================
 * Backend selection
 * =========================================================================== */

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

/* ===========================================================================
 * Colorspace helper
 * =========================================================================== */

static enum oil_colorspace nogamma_cs(enum oil_colorspace cs) {
	switch (cs) {
	case OIL_CS_RGB:  return OIL_CS_RGB_NOGAMMA;
	case OIL_CS_RGBA: return OIL_CS_RGBA_NOGAMMA;
	case OIL_CS_RGBX: return OIL_CS_RGBX_NOGAMMA;
	default: return cs;
	}
}

/* ===========================================================================
 * Resumable resize — types
 * =========================================================================== */

/* Caller-supplied request, carried in struct resumable_resize. src_* may be
 * normalized during start: when is_zoomed, the rect is expanded to window
 * aspect, and callers should read the adjusted values back from rr->cfg
 * after start. */
struct rr_config {
	int surface_width;
	int surface_height;
	int threaded;
	int no_gamma;
	int is_zoomed;
	double src_x;
	double src_y;
	double src_w;
	double src_h;
};

/* Threaded pipeline state. Fields below the barrier comment are mutated by
 * multiple threads and must only be touched while holding `mutex`. */
struct rr_pipeline {
	SDL_Thread *decoder_thread;
	SDL_Thread *scaler_thread;
	SDL_Thread *worker_thread;
	SDL_Mutex *mutex;
	SDL_Condition *cv;
	unsigned char *slots[LINE_QUEUE_DEPTH];
	int n_slots;
	/* mutex-protected from here down */
	int head;
	int tail;
	int count;
	int ypos;
	int decoder_done;
	int scaler_done;
	int aborted;
};

struct resumable_resize {
	struct rr_config cfg;

	FILE *io;

	int img_width;
	int img_height;

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
	int slot_rowbytes;

	unsigned char *outbuf;
	unsigned char *scaled_buf;

	/* scale_os points at the active wrapper's oil_scale (olj.os for
	 * JPEG, olp.os for PNG). */
	struct oil_scale *scale_os;
	int (*scale_in)(struct oil_scale *, unsigned char *);
	int (*scale_out)(struct oil_scale *, unsigned char *);
	void (*decode_row)(struct resumable_resize *, unsigned char *, int);
	void (*format_end)(struct resumable_resize *);

	struct oil_libjpeg olj;
	struct oil_libpng olp;

	struct rr_pipeline pipe;
};

/* ===========================================================================
 * Resumable resize — crop math
 * =========================================================================== */

static void clamp_crop(struct resumable_resize *rr) {
	struct rr_config *c = &rr->cfg;
	if (c->src_w <= 0 || c->src_h <= 0) {
		c->src_x = 0;
		c->src_y = 0;
		c->src_w = rr->img_width;
		c->src_h = rr->img_height;
		return;
	}
	if (c->src_x < 0) {
		c->src_w += c->src_x;
		c->src_x = 0;
	}
	if (c->src_y < 0) {
		c->src_h += c->src_y;
		c->src_y = 0;
	}
	if (c->src_x + c->src_w > rr->img_width) c->src_w = rr->img_width - c->src_x;
	if (c->src_y + c->src_h > rr->img_height) c->src_h = rr->img_height - c->src_y;
}

/* When zoomed, expand the logical source rect to match the window aspect
 * ratio so the output texture fills the window: the user's selection is a
 * minimum, extra content is captured on the under-sized axis (centered on
 * the selection center, shifted to stay inside the image). If expansion
 * would exceed the image, clamp to the image and shrink the perpendicular
 * axis to keep window aspect. The adjusted rect is written back to
 * rr->cfg.src_*; the caller mirrors it for the next session.
 *
 * When not zoomed, preserve the full image via oil_fix_ratio (letterbox if
 * window aspect differs from image aspect); rr->cfg.src_* is left alone. */
static int compute_fed_and_out(struct resumable_resize *rr) {
	struct rr_config *c = &rr->cfg;
	if (c->is_zoomed) {
		double win_aspect, sel_aspect, cx, cy, new_w, new_h, new_x, new_y;

		rr->out_width = c->surface_width;
		rr->out_height = c->surface_height;
		if (rr->out_width < 1) rr->out_width = 1;
		if (rr->out_height < 1) rr->out_height = 1;

		win_aspect = (double)rr->out_width / rr->out_height;
		sel_aspect = c->src_w / c->src_h;
		cx = c->src_x + c->src_w / 2.0;
		cy = c->src_y + c->src_h / 2.0;

		if (sel_aspect < win_aspect) {
			new_h = c->src_h;
			new_w = new_h * win_aspect;
		} else {
			new_w = c->src_w;
			new_h = new_w / win_aspect;
		}
		if (new_w > rr->img_width) {
			new_w = rr->img_width;
			new_h = new_w / win_aspect;
		}
		if (new_h > rr->img_height) {
			new_h = rr->img_height;
			new_w = new_h * win_aspect;
		}

		new_x = cx - new_w / 2.0;
		new_y = cy - new_h / 2.0;
		if (new_x < 0) new_x = 0;
		if (new_y < 0) new_y = 0;
		if (new_x + new_w > rr->img_width) new_x = rr->img_width - new_w;
		if (new_y + new_h > rr->img_height) new_y = rr->img_height - new_h;

		c->src_x = new_x;
		c->src_y = new_y;
		c->src_w = new_w;
		c->src_h = new_h;
	} else {
		int src_iw = (int)(c->src_w + 0.5);
		int src_ih = (int)(c->src_h + 0.5);
		if (src_iw < 1) src_iw = 1;
		if (src_ih < 1) src_ih = 1;
		rr->out_width = c->surface_width;
		rr->out_height = c->surface_height;
		if (oil_fix_ratio(src_iw, src_ih, &rr->out_width, &rr->out_height) < 0) return -1;
		if (rr->out_width < 1) rr->out_width = 1;
		if (rr->out_height < 1) rr->out_height = 1;
	}

	return oil_required_input_rect(rr->img_height, rr->img_width,
		c->src_y, c->src_h, c->src_x, c->src_w,
		rr->out_height, rr->out_width,
		&rr->fed_y, &rr->fed_height, &rr->fed_x, &rr->fed_width);
}

/* ===========================================================================
 * Resumable resize — PNG decoder
 * =========================================================================== */

static void decode_png_row(struct resumable_resize *rr, unsigned char *slot, int row) {
	(void)row;
	oil_libpng_decode_row(&rr->olp, slot);
}

static void png_end(struct resumable_resize *rr) {
	png_structp rpng = rr->olp.rpng;
	png_infop rinfo = rr->olp.rinfo;
	free(rr->outbuf);
	oil_libpng_free(&rr->olp);
	png_destroy_read_struct(&rpng, &rinfo, NULL);
}

static int png_start(struct resumable_resize *rr) {
	png_structp rpng;
	png_infop rinfo;

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
	if (rr->cfg.no_gamma) rr->cs = nogamma_cs(rr->cs);
	rr->cmp = OIL_CMP(rr->cs);

	clamp_crop(rr);
	if (rr->cfg.src_w < 1 || rr->cfg.src_h < 1) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}
	if (compute_fed_and_out(rr) < 0) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}
	rr->slot_rowbytes = rr->fed_width * rr->cmp;

	if (oil_libpng_init_ex(&rr->olp, rpng, rinfo, rr->out_width, rr->out_height,
		rr->cfg.src_x, rr->cfg.src_y, rr->cfg.src_w, rr->cfg.src_h,
		rr->cs) != 0) {
		png_destroy_read_struct(&rpng, &rinfo, NULL);
		return -1;
	}
	rr->scale_os = &rr->olp.os;

	rr->outbuf = malloc((size_t)rr->out_width * rr->cmp);
	if (!rr->outbuf) goto fail_olp;

	rr->decode_row = decode_png_row;
	rr->format_end = png_end;
	return 0;

fail_olp:
	oil_libpng_free(&rr->olp);
	png_destroy_read_struct(&rpng, &rinfo, NULL);
	return -1;
}

/* ===========================================================================
 * Resumable resize — JPEG decoder
 * =========================================================================== */

struct jpeg_err {
	struct jpeg_error_mgr mgr;
	jmp_buf jmpbuf;
};

static void jpeg_err_exit(j_common_ptr cinfo)
{
	struct jpeg_err *err = (struct jpeg_err *)cinfo->err;
	longjmp(err->jmpbuf, 1);
}

static void decode_jpeg_row(struct resumable_resize *rr, unsigned char *slot, int row) {
	(void)row;
	oil_libjpeg_decode_row(&rr->olj, slot);
}

static void jpeg_end(struct resumable_resize *rr) {
	struct jpeg_decompress_struct *dinfo = rr->olj.dinfo;
	free(rr->outbuf);
	oil_libjpeg_free(&rr->olj);
	free(dinfo->err);
	jpeg_destroy_decompress(dinfo);
	free(dinfo);
}

static int jpeg_start(struct resumable_resize *rr)
{
	struct jpeg_decompress_struct *dinfo;
	struct jpeg_err *jerr;

	dinfo = malloc(sizeof(struct jpeg_decompress_struct));
	if (!dinfo) {
		return -1;
	}
	jerr = malloc(sizeof(struct jpeg_err));
	if (!jerr) {
		free(dinfo);
		return -1;
	}

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
	if (rr->cfg.no_gamma) rr->cs = nogamma_cs(rr->cs);
	rr->cmp = dinfo->output_components;

	clamp_crop(rr);
	if (rr->cfg.src_w < 1 || rr->cfg.src_h < 1) goto fail_jpeg;
	if (compute_fed_and_out(rr) < 0) goto fail_jpeg;

	if (oil_libjpeg_init_ex(&rr->olj, dinfo, rr->out_width, rr->out_height,
		rr->cfg.src_x, rr->cfg.src_y, rr->cfg.src_w, rr->cfg.src_h,
		rr->cs) != 0) {
		goto fail_jpeg;
	}
	/* jpeg_crop_scanline (turbo) may widen fed_w to an iMCU boundary; the
	 * authoritative value is what the wrapper recorded. */
	rr->fed_width = rr->olj.fed_width;
	rr->slot_rowbytes = rr->fed_width * rr->cmp;
	rr->scale_os = &rr->olj.os;

	rr->outbuf = malloc((size_t)rr->out_width * rr->cmp);
	if (!rr->outbuf) goto fail_olj;

	rr->decode_row = decode_jpeg_row;
	rr->format_end = jpeg_end;
	return 0;

fail_olj:
	oil_libjpeg_free(&rr->olj);
fail_jpeg:
	jpeg_destroy_decompress(dinfo);
	free(jerr);
	free(dinfo);
	return -1;
}

/* ===========================================================================
 * Resumable resize — worker threads
 * =========================================================================== */

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

/* Emit one scaled row from rr->outbuf into rr->scaled_buf at row index
 * local_ypos, then publish the new progress count. */
static void finish_output_row(struct resumable_resize *rr, int local_ypos)
{
	unsigned char *tmp;
	rr->scale_out(rr->scale_os, rr->outbuf);
	tmp = rr->scaled_buf + local_ypos * rr->out_width * 4;
	translate(rr->outbuf, tmp, rr->out_width, rr->cmp);

	SDL_LockMutex(rr->pipe.mutex);
	rr->pipe.ypos = local_ypos + 1;
	SDL_UnlockMutex(rr->pipe.mutex);
}

static int decoder_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	struct rr_pipeline *p = &rr->pipe;
	int row;

	for (row = 0; row < rr->fed_height; row++) {
		unsigned char *slot;

		SDL_LockMutex(p->mutex);
		while (p->count == LINE_QUEUE_DEPTH && !p->aborted) {
			SDL_WaitCondition(p->cv, p->mutex);
		}
		if (p->aborted) {
			SDL_UnlockMutex(p->mutex);
			return 0;
		}
		slot = p->slots[p->head];
		SDL_UnlockMutex(p->mutex);

		rr->decode_row(rr, slot, row);

		SDL_LockMutex(p->mutex);
		p->head = (p->head + 1) % LINE_QUEUE_DEPTH;
		p->count++;
		SDL_SignalCondition(p->cv);
		SDL_UnlockMutex(p->mutex);
	}

	SDL_LockMutex(p->mutex);
	p->decoder_done = 1;
	SDL_SignalCondition(p->cv);
	SDL_UnlockMutex(p->mutex);
	return 0;
}

static int scaler_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	struct rr_pipeline *p = &rr->pipe;
	int local_ypos = 0;

	while (local_ypos < rr->out_height) {
		while (oil_scale_slots(rr->scale_os) > 0) {
			unsigned char *slot;

			SDL_LockMutex(p->mutex);
			while (p->count == 0 && !p->decoder_done && !p->aborted) {
				SDL_WaitCondition(p->cv, p->mutex);
			}
			if (p->aborted || p->count == 0) {
				SDL_UnlockMutex(p->mutex);
				goto done;
			}
			slot = p->slots[p->tail];
			SDL_UnlockMutex(p->mutex);

			rr->scale_in(rr->scale_os, slot);

			SDL_LockMutex(p->mutex);
			p->tail = (p->tail + 1) % LINE_QUEUE_DEPTH;
			p->count--;
			SDL_SignalCondition(p->cv);
			SDL_UnlockMutex(p->mutex);
		}

		finish_output_row(rr, local_ypos);
		local_ypos++;
	}

done:
	SDL_LockMutex(p->mutex);
	p->scaler_done = 1;
	SDL_UnlockMutex(p->mutex);
	return 0;
}

static int worker_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	struct rr_pipeline *p = &rr->pipe;
	unsigned char *inbuf = p->slots[0];
	int local_ypos = 0;
	int row = 0;
	int aborted;

	while (local_ypos < rr->out_height) {
		SDL_LockMutex(p->mutex);
		aborted = p->aborted;
		SDL_UnlockMutex(p->mutex);
		if (aborted) goto done;

		while (oil_scale_slots(rr->scale_os) > 0) {
			rr->decode_row(rr, inbuf, row++);
			rr->scale_in(rr->scale_os, inbuf);
		}

		finish_output_row(rr, local_ypos);
		local_ypos++;
	}

done:
	SDL_LockMutex(p->mutex);
	p->scaler_done = 1;
	SDL_UnlockMutex(p->mutex);
	return 0;
}

/* ===========================================================================
 * Resumable resize — session lifecycle
 * =========================================================================== */

static int looks_like_png(FILE *io)
{
	int peek;
	peek = getc(io);
	ungetc(peek, io);
	return peek == PNG_FIRST_MAGIC_BYTE;
}

static int resumable_resize_start(struct resumable_resize *rr, char *path,
                                  const struct backend_entry *be,
                                  const struct rr_config *cfg)
{
	struct rr_pipeline *p;
	int i;
	int is_png;

	memset(rr, 0, sizeof(*rr));
	rr->cfg = *cfg;
	rr->scale_in = be->scale_in;
	rr->scale_out = be->scale_out;
	p = &rr->pipe;
	p->n_slots = rr->cfg.threaded ? LINE_QUEUE_DEPTH : 1;

	rr->io = fopen(path, "rb");
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

	for (i = 0; i < p->n_slots; i++) {
		p->slots[i] = malloc(rr->slot_rowbytes);
		if (!p->slots[i]) goto fail_slots;
	}

	p->mutex = SDL_CreateMutex();
	p->cv = SDL_CreateCondition();
	if (!p->mutex || !p->cv) {
		fprintf(stderr, "Error: SDL sync init failed: %s\n", SDL_GetError());
		goto fail_sync;
	}

	if (rr->cfg.threaded) {
		p->decoder_thread = SDL_CreateThread(decoder_thread_fn, "oil-decoder", rr);
		p->scaler_thread = SDL_CreateThread(scaler_thread_fn, "oil-scaler", rr);
		if (!p->decoder_thread || !p->scaler_thread) {
			fprintf(stderr, "Error: SDL_CreateThread failed: %s\n", SDL_GetError());
			SDL_LockMutex(p->mutex);
			p->aborted = 1;
			SDL_BroadcastCondition(p->cv);
			SDL_UnlockMutex(p->mutex);
			if (p->decoder_thread) SDL_WaitThread(p->decoder_thread, NULL);
			if (p->scaler_thread) SDL_WaitThread(p->scaler_thread, NULL);
			goto fail_sync;
		}
	} else {
		p->worker_thread = SDL_CreateThread(worker_thread_fn, "oil-worker", rr);
		if (!p->worker_thread) {
			fprintf(stderr, "Error: SDL_CreateThread failed: %s\n", SDL_GetError());
			goto fail_sync;
		}
	}

	return 0;

fail_sync:
	SDL_DestroyMutex(p->mutex);
	SDL_DestroyCondition(p->cv);
fail_slots:
	for (i = 0; i < p->n_slots; i++) free(p->slots[i]);
	free(rr->scaled_buf);
fail_decoder:
	rr->format_end(rr);
fail_io:
	fclose(rr->io);
	return -1;
}

static void resumable_resize_end(struct resumable_resize *rr)
{
	struct rr_pipeline *p = &rr->pipe;
	int i;

	SDL_LockMutex(p->mutex);
	p->aborted = 1;
	SDL_BroadcastCondition(p->cv);
	SDL_UnlockMutex(p->mutex);

	if (rr->cfg.threaded) {
		SDL_WaitThread(p->decoder_thread, NULL);
		SDL_WaitThread(p->scaler_thread, NULL);
	} else {
		SDL_WaitThread(p->worker_thread, NULL);
	}

	SDL_DestroyMutex(p->mutex);
	SDL_DestroyCondition(p->cv);

	for (i = 0; i < p->n_slots; i++) {
		free(p->slots[i]);
	}
	free(rr->scaled_buf);

	rr->format_end(rr);
	fclose(rr->io);
}

/* ===========================================================================
 * Presentation
 * =========================================================================== */

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

/* ===========================================================================
 * Drag-to-crop
 * =========================================================================== */

static SDL_FRect make_drag_rect(float ax, float ay, float bx, float by)
{
	SDL_FRect r;
	r.x = ax < bx ? ax : bx;
	r.y = ay < by ? ay : by;
	r.w = fabsf(bx - ax);
	r.h = fabsf(by - ay);
	return r;
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

/* ===========================================================================
 * View stash (undo history of zoom states)
 * =========================================================================== */

struct view_stash {
	SDL_Texture *tex;
	double crop_x;
	double crop_y;
	double crop_w;
	double crop_h;
	int is_zoomed;
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
                       double cx, double cy, double cw, double ch, int zoomed)
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
	s[*count].is_zoomed = zoomed;
	(*count)++;
}

/* ===========================================================================
 * App state and event handlers
 * =========================================================================== */

struct app_state {
	SDL_Window *window;
	SDL_Renderer *renderer;
	SDL_Texture *display_tex;
	char *path;
	const struct backend_entry *be;

	struct resumable_resize rr;
	struct rr_config cfg;     /* carried across sessions */
	int img_w;

	int render_in_progress;
	int resize_pending;
	int last_displayed_ypos;
	Uint64 resize_start_time;

	int drag_active;
	float drag_start_x, drag_start_y;
	float drag_cur_x, drag_cur_y;

	struct view_stash stash[STASH_DEPTH];
	int stash_count;

	int need_present;
	int should_exit;
};

static int start_resize_session(struct app_state *app)
{
	int rw, rh;
	SDL_GetRenderOutputSize(app->renderer, &rw, &rh);
	app->cfg.surface_width = rw;
	app->cfg.surface_height = rh;
	if (resumable_resize_start(&app->rr, app->path, app->be, &app->cfg) < 0) return -1;
	if (app->display_tex) SDL_DestroyTexture(app->display_tex);
	app->display_tex = create_blank_texture(app->renderer, app->rr.out_width, app->rr.out_height);
	app->cfg = app->rr.cfg;            /* mirror back any expanded crop */
	app->img_w = app->rr.img_width;
	return 0;
}

static void abort_current_session(struct app_state *app)
{
	if (app->render_in_progress) {
		resumable_resize_end(&app->rr);
		app->render_in_progress = 0;
	}
}

static void handle_redraw_request(struct app_state *app, int from_window_resize)
{
	abort_current_session(app);
	app->drag_active = 0;
	if (from_window_resize) {
		stash_invalidate_textures(app->stash, app->stash_count);
		present_frame(app->renderer, app->display_tex, NULL);
	}
	app->resize_pending = 1;
}

static void handle_left_button_down(struct app_state *app, const SDL_Event *ev)
{
	float rx, ry;
	SDL_RenderCoordinatesFromWindow(app->renderer,
		ev->button.x, ev->button.y, &rx, &ry);
	app->drag_start_x = app->drag_cur_x = rx;
	app->drag_start_y = app->drag_cur_y = ry;
	app->drag_active = 1;
	app->need_present = 1;
}

static void handle_right_button_down(struct app_state *app)
{
	struct view_stash *top;

	app->drag_active = 0;
	if (app->stash_count == 0) return;

	abort_current_session(app);
	app->stash_count--;
	top = &app->stash[app->stash_count];
	app->cfg.src_x = top->crop_x;
	app->cfg.src_y = top->crop_y;
	app->cfg.src_w = top->crop_w;
	app->cfg.src_h = top->crop_h;
	app->cfg.is_zoomed = top->is_zoomed;
	if (top->tex) {
		SDL_DestroyTexture(app->display_tex);
		app->display_tex = top->tex;
		app->need_present = 1;
	} else {
		app->resize_pending = 1;
	}
}

static void handle_motion(struct app_state *app, const SDL_Event *ev)
{
	float rx, ry;
	SDL_RenderCoordinatesFromWindow(app->renderer,
		ev->motion.x, ev->motion.y, &rx, &ry);
	app->drag_cur_x = rx;
	app->drag_cur_y = ry;
	app->need_present = 1;
}

static void handle_left_button_up(struct app_state *app, const SDL_Event *ev)
{
	float rx, ry;
	SDL_FRect drag;
	double nx, ny, nw, nh;

	app->drag_active = 0;
	SDL_RenderCoordinatesFromWindow(app->renderer,
		ev->button.x, ev->button.y, &rx, &ry);
	app->drag_cur_x = rx;
	app->drag_cur_y = ry;
	drag = make_drag_rect(app->drag_start_x, app->drag_start_y,
	                      app->drag_cur_x, app->drag_cur_y);
	if ((drag.w >= MIN_DRAG_PIXELS || drag.h >= MIN_DRAG_PIXELS) &&
	    app->img_w > 0 &&
	    drag_rect_to_crop(app->renderer, app->display_tex, drag,
	                      app->cfg.src_x, app->cfg.src_y,
	                      app->cfg.src_w, app->cfg.src_h,
	                      &nx, &ny, &nw, &nh) == 0) {
		abort_current_session(app);
		stash_push(app->stash, &app->stash_count, app->display_tex,
		           app->cfg.src_x, app->cfg.src_y,
		           app->cfg.src_w, app->cfg.src_h, app->cfg.is_zoomed);
		app->display_tex = NULL;
		app->cfg.src_x = nx;
		app->cfg.src_y = ny;
		app->cfg.src_w = nw;
		app->cfg.src_h = nh;
		app->cfg.is_zoomed = 1;
		app->resize_pending = 1;
	}
	app->need_present = 1;
}

static void dispatch_event(struct app_state *app, const SDL_Event *ev)
{
	switch (ev->type) {
	case SDL_EVENT_QUIT:
		app->should_exit = 1;
		break;
	case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
		handle_redraw_request(app, 1);
		break;
	case SDL_EVENT_KEY_DOWN:
		if (ev->key.key == SDLK_F5) handle_redraw_request(app, 0);
		break;
	case SDL_EVENT_MOUSE_BUTTON_DOWN:
		if (ev->button.button == SDL_BUTTON_LEFT) handle_left_button_down(app, ev);
		else if (ev->button.button == SDL_BUTTON_RIGHT) handle_right_button_down(app);
		break;
	case SDL_EVENT_MOUSE_MOTION:
		if (app->drag_active) handle_motion(app, ev);
		break;
	case SDL_EVENT_MOUSE_BUTTON_UP:
		if (ev->button.button == SDL_BUTTON_LEFT && app->drag_active)
			handle_left_button_up(app, ev);
		break;
	}
}

static void maybe_start_session(struct app_state *app)
{
	if (!app->resize_pending || app->render_in_progress) return;
	if (SDL_GetMouseState(NULL, NULL) != 0) return;
	app->resize_pending = 0;
	app->last_displayed_ypos = 0;
	app->resize_start_time = SDL_GetTicks();
	if (start_resize_session(app) == 0) app->render_in_progress = 1;
}

static void pump_render(struct app_state *app)
{
	struct resumable_resize *rr = &app->rr;
	int local_ypos, local_done;

	if (!app->render_in_progress) return;

	SDL_LockMutex(rr->pipe.mutex);
	local_ypos = rr->pipe.ypos;
	local_done = rr->pipe.scaler_done;
	SDL_UnlockMutex(rr->pipe.mutex);

	if (local_done) {
		Uint64 elapsed;
		app->render_in_progress = 0;
		if (app->last_displayed_ypos < rr->out_height) {
			update_texture_rows(app->display_tex, rr,
			                    app->last_displayed_ypos, rr->out_height);
		}
		resumable_resize_end(rr);
		app->need_present = 1;
		elapsed = SDL_GetTicks() - app->resize_start_time;
		fprintf(stderr, "Resize ticks: %llu\n", (unsigned long long)elapsed);
	} else if (local_ypos > app->last_displayed_ypos) {
		update_texture_rows(app->display_tex, rr,
		                    app->last_displayed_ypos, local_ypos);
		app->last_displayed_ypos = local_ypos;
		app->need_present = 1;
	}
}

static void present_current(struct app_state *app)
{
	SDL_FRect drag, *overlay = NULL;
	if (app->drag_active) {
		drag = make_drag_rect(app->drag_start_x, app->drag_start_y,
		                      app->drag_cur_x, app->drag_cur_y);
		overlay = &drag;
	}
	present_frame(app->renderer, app->display_tex, overlay);
	app->need_present = 0;
}

static void app_cleanup(struct app_state *app)
{
	if (app->render_in_progress) resumable_resize_end(&app->rr);
	stash_clear(app->stash, &app->stash_count);
	if (app->display_tex) SDL_DestroyTexture(app->display_tex);
	if (app->renderer) SDL_DestroyRenderer(app->renderer);
	if (app->window) SDL_DestroyWindow(app->window);
	SDL_Quit();
}

/* ===========================================================================
 * Main
 * =========================================================================== */

int main(int argc, char **argv)
{
	struct app_state app;
	SDL_Event event;
	int argi;

	memset(&app, 0, sizeof(app));
	app.cfg.threaded = 1;

	for (argi = 1; argi < argc; argi++) {
		if (strcmp(argv[argi], "--no-threaded") == 0) {
			app.cfg.threaded = 0;
		} else if (strcmp(argv[argi], "--no-gamma") == 0) {
			app.cfg.no_gamma = 1;
		} else if (argv[argi][0] == '-' && argv[argi][1] == '-') {
			app.be = find_backend(argv[argi]);
			if (!app.be) {
				fprintf(stderr, "Error: unknown or unavailable backend: %s\n", argv[argi]);
				return 1;
			}
		} else {
			app.path = argv[argi];
		}
	}
	if (!app.path) {
		fprintf(stderr, "Usage: %s [--scalar|--sse2|--avx2|--neon] [--no-threaded] [--no-gamma] <image>\n", argv[0]);
		return 1;
	}
	if (!app.be) app.be = default_backend();

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}
	app.window = SDL_CreateWindow(app.path, 640, 480, SDL_WINDOW_RESIZABLE);
	if (!app.window) {
		fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
		app_cleanup(&app);
		return 1;
	}
	app.renderer = SDL_CreateRenderer(app.window, NULL);
	if (!app.renderer) {
		fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
		app_cleanup(&app);
		return 1;
	}

	app.resize_start_time = SDL_GetTicks();
	if (start_resize_session(&app) < 0) {
		app_cleanup(&app);
		return 1;
	}
	app.render_in_progress = 1;

	while (!app.should_exit) {
		int timeout = app.render_in_progress ? 16 : -1;
		if (SDL_WaitEventTimeout(&event, timeout)) {
			dispatch_event(&app, &event);
			if (app.should_exit) break;
		}

		maybe_start_session(&app);
		pump_render(&app);
		if (app.need_present) present_current(&app);
	}

	app_cleanup(&app);
	return 0;
}
