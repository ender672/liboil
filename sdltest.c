#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL3/SDL.h>
#include <jpeglib.h>
#include <png.h>

#include "oil_resample.h"
#include "oil_libjpeg.h"
#include "oil_libpng.h"

#define LINE_QUEUE_DEPTH 16

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
	int surface_width;
	int surface_height;
	int out_width;
	int out_height;
	int in_height;
	int in_rowbytes;
	int cmp;
	int threaded;
	int no_gamma;
	int n_slots;
	unsigned char *outbuf;
	unsigned char *scaled_buf;

	struct oil_scale *os;
	int (*scale_in)(struct oil_scale *, unsigned char *);
	int (*scale_out)(struct oil_scale *, unsigned char *);
	void (*decode_row)(struct resumable_resize *, unsigned char *, int);
	void (*format_end)(struct resumable_resize *);

	struct oil_libjpeg *olj;
	struct jpeg_decompress_struct *dinfo;

	struct oil_libpng *olp;
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
	memcpy(slot, rr->olp->inimage[row], rr->in_rowbytes);
}

static void decode_png_streaming(struct resumable_resize *rr, unsigned char *slot, int row) {
	(void)row;
	png_read_row(rr->rpng, slot, NULL);
}

static void decode_jpeg_row(struct resumable_resize *rr, unsigned char *slot, int row) {
	(void)row;
	jpeg_read_scanlines(rr->dinfo, &slot, 1);
}

static void png_end(struct resumable_resize *rr);
static void jpeg_end(struct resumable_resize *rr);

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

	rr->rpng = rpng;
	rr->rinfo = rinfo;

	png_init_io(rpng, rr->io);
	png_read_info(rpng, rinfo);

	png_set_packing(rpng);
	png_set_strip_16(rpng);
	png_set_expand(rpng);
	png_set_interlace_handling(rpng);
	png_read_update_info(rpng, rinfo);

	rr->out_width = rr->surface_width;
	rr->out_height = rr->surface_height;
	oil_fix_ratio(png_get_image_width(rpng, rinfo), png_get_image_height(rpng, rinfo), &rr->out_width, &rr->out_height);

	rr->olp = malloc(sizeof(struct oil_libpng));
	oil_libpng_init(rr->olp, rpng, rinfo, rr->out_width, rr->out_height);
	if (rr->no_gamma) rr->olp->os.cs = nogamma_cs(rr->olp->os.cs);
	rr->cmp = OIL_CMP(rr->olp->os.cs);
	rr->outbuf = malloc(rr->out_width * rr->cmp);
	rr->in_height = png_get_image_height(rpng, rinfo);
	rr->in_rowbytes = png_get_rowbytes(rpng, rinfo);
	rr->os = &rr->olp->os;
	rr->decode_row = (png_get_interlace_type(rpng, rinfo) == PNG_INTERLACE_ADAM7)
		? decode_png_interlaced : decode_png_streaming;
	rr->format_end = png_end;
	return 0;
}

static void png_end(struct resumable_resize *rr) {
	free(rr->outbuf);
	oil_libpng_free(rr->olp);
	free(rr->olp);
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

	rr->out_width = rr->surface_width;
	rr->out_height = rr->surface_height;
	oil_fix_ratio(dinfo->output_width, dinfo->output_height, &rr->out_width, &rr->out_height);

	rr->olj = malloc(sizeof(struct oil_libjpeg));
	oil_libjpeg_init(rr->olj, dinfo, rr->out_width, rr->out_height);
	if (rr->no_gamma) rr->olj->os.cs = nogamma_cs(rr->olj->os.cs);
	rr->cmp = 3;
	rr->outbuf = malloc(rr->out_width * rr->cmp);
	rr->in_height = dinfo->output_height;
	rr->in_rowbytes = dinfo->output_width * dinfo->output_components;
	rr->os = &rr->olj->os;
	rr->decode_row = decode_jpeg_row;
	rr->format_end = jpeg_end;
	return 0;
}

static void jpeg_end(struct resumable_resize *rr) {
	free(rr->outbuf);
	oil_libjpeg_free(rr->olj);
	free(rr->olj);
	free(rr->dinfo->err);
	jpeg_destroy_decompress(rr->dinfo);
	free(rr->dinfo);
}

static int decoder_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	int row;

	for (row = 0; row < rr->in_height; row++) {
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

		while (oil_scale_slots(rr->os) > 0) {
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

			rr->scale_in(rr->os, slot);

			SDL_LockMutex(rr->mutex);
			rr->tail = (rr->tail + 1) % LINE_QUEUE_DEPTH;
			rr->count--;
			SDL_SignalCondition(rr->cv);
			SDL_UnlockMutex(rr->mutex);
		}

		rr->scale_out(rr->os, rr->outbuf);
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

		while (oil_scale_slots(rr->os) > 0) {
			rr->decode_row(rr, inbuf, row++);
			rr->scale_in(rr->os, inbuf);
		}

		rr->scale_out(rr->os, rr->outbuf);
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
                                  int no_gamma)
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
		rr->slots[i] = malloc(rr->in_rowbytes);
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

static void present_frame(SDL_Renderer *renderer, SDL_Texture *display_tex)
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
                                int no_gamma)
{
	int rw, rh;
	SDL_GetRenderOutputSize(renderer, &rw, &rh);
	if (resumable_resize_start(rr, path, rw, rh, be, threaded, no_gamma) < 0) return -1;
	if (*display_tex) SDL_DestroyTexture(*display_tex);
	*display_tex = create_blank_texture(renderer, rr->out_width, rr->out_height);
	return 0;
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
	if (start_resize_session(&rr, path, renderer, &display_tex, be, threaded, no_gamma) < 0) {
		SDL_DestroyRenderer(renderer);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 1;
	}
	render_in_progress = 1;

	while (1) {
		event_happened = SDL_WaitEventTimeout(&event, render_in_progress ? 16 : -1);

		if (event_happened) {
			if (event.type == SDL_EVENT_QUIT) {
				if (render_in_progress) {
					resumable_resize_end(&rr);
				}
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
				if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
					present_frame(renderer, display_tex);
				}
				resize_pending = 1;
			}
		}

		if (resize_pending && !render_in_progress &&
		    SDL_GetMouseState(NULL, NULL) == 0) {
			resize_pending = 0;
			last_displayed_ypos = 0;
			resize_start_time = SDL_GetTicks();
			if (start_resize_session(&rr, path, renderer, &display_tex, be, threaded, no_gamma) == 0) {
				render_in_progress = 1;
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
				present_frame(renderer, display_tex);
				elapsed_time = SDL_GetTicks() - resize_start_time;
				fprintf(stderr, "Resize ticks: %llu\n", (unsigned long long)elapsed_time);
			} else if (local_ypos > last_displayed_ypos) {
				update_texture_rows(display_tex, &rr, last_displayed_ypos, local_ypos);
				last_displayed_ypos = local_ypos;
				present_frame(renderer, display_tex);
			}
		}
	}
}
