#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL3/SDL.h>

#define LINE_QUEUE_DEPTH 16

enum backend { BACKEND_DEFAULT, BACKEND_SCALAR, BACKEND_SSE2, BACKEND_AVX2, BACKEND_NEON };
static enum backend backend;

#include "oil_resample.h"

static int (*scale_in_fn)(struct oil_scale *, unsigned char *);
static int (*scale_out_fn)(struct oil_scale *, unsigned char *);
#include "oil_libjpeg.h"
#include "oil_libpng.h"
#include <jpeglib.h>
#include <png.h>
struct resumable_resize {
	FILE *io;
	int looks_like_png;
	int interlaced_png;
	int surface_width;
	int surface_height;
	int out_width;
	int out_height;
	int in_height;
	int in_rowbytes;
	int cmp;
	unsigned char *outbuf;

	struct oil_libjpeg *olj;
	struct jpeg_decompress_struct *dinfo;

	struct oil_libpng *olp;
	png_structp rpng;
	png_infop rinfo;

	unsigned char *surface_pixels;

	SDL_Thread *decoder_thread;
	SDL_Thread *scaler_thread;
	SDL_Mutex *mutex;
	SDL_Condition *not_empty;
	SDL_Condition *not_full;
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

static int resize_center_offset(struct resumable_resize *rr)
{
	int x_center_offset, y_center_offset;

	x_center_offset = (rr->surface_width - rr->out_width) / 2 * 4;
	y_center_offset = (rr->surface_height - rr->out_height) / 2 * rr->surface_width * 4;
	return x_center_offset + y_center_offset;
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
	rr->cmp = OIL_CMP(rr->olp->os.cs);
	rr->outbuf = malloc(rr->out_width * rr->cmp);
	rr->in_height = png_get_image_height(rpng, rinfo);
	rr->in_rowbytes = png_get_rowbytes(rpng, rinfo);
	rr->interlaced_png = (png_get_interlace_type(rpng, rinfo) == PNG_INTERLACE_ADAM7);
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
	rr->cmp = 3;
	rr->outbuf = malloc(rr->out_width * rr->cmp);
	rr->in_height = dinfo->output_height;
	rr->in_rowbytes = dinfo->output_width * dinfo->output_components;
	rr->interlaced_png = 0;
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
			SDL_WaitCondition(rr->not_full, rr->mutex);
		}
		if (rr->aborted) {
			SDL_UnlockMutex(rr->mutex);
			return 0;
		}
		slot = rr->slots[rr->head];
		SDL_UnlockMutex(rr->mutex);

		if (rr->looks_like_png) {
			if (rr->interlaced_png) {
				memcpy(slot, rr->olp->inimage[row], rr->in_rowbytes);
			} else {
				png_read_row(rr->rpng, slot, NULL);
			}
		} else {
			jpeg_read_scanlines(rr->dinfo, &slot, 1);
		}

		SDL_LockMutex(rr->mutex);
		rr->head = (rr->head + 1) % LINE_QUEUE_DEPTH;
		rr->count++;
		SDL_BroadcastCondition(rr->not_empty);
		SDL_UnlockMutex(rr->mutex);
	}

	SDL_LockMutex(rr->mutex);
	rr->decoder_done = 1;
	SDL_BroadcastCondition(rr->not_empty);
	SDL_UnlockMutex(rr->mutex);
	return 0;
}

static int scaler_thread_fn(void *arg)
{
	struct resumable_resize *rr = arg;
	struct oil_scale *os;
	int center_offset;
	int local_ypos;

	os = rr->looks_like_png ? &rr->olp->os : &rr->olj->os;
	center_offset = resize_center_offset(rr);
	local_ypos = 0;

	while (local_ypos < rr->out_height) {
		unsigned char *tmp;

		while (oil_scale_slots(os) > 0) {
			unsigned char *slot;

			SDL_LockMutex(rr->mutex);
			while (rr->count == 0 && !rr->decoder_done && !rr->aborted) {
				SDL_WaitCondition(rr->not_empty, rr->mutex);
			}
			if (rr->aborted || rr->count == 0) {
				SDL_UnlockMutex(rr->mutex);
				goto done;
			}
			slot = rr->slots[rr->tail];
			SDL_UnlockMutex(rr->mutex);

			scale_in_fn(os, slot);

			SDL_LockMutex(rr->mutex);
			rr->tail = (rr->tail + 1) % LINE_QUEUE_DEPTH;
			rr->count--;
			SDL_BroadcastCondition(rr->not_full);
			SDL_UnlockMutex(rr->mutex);
		}

		scale_out_fn(os, rr->outbuf);
		tmp = rr->surface_pixels + center_offset + local_ypos * rr->surface_width * 4;
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

static int resumable_resize_start(struct resumable_resize *rr, char *path, int surface_width, int surface_height, unsigned char *surface_pixels)
{
	int i;

	rr->io = fopen(path, "r");
	if (!rr->io) {
		fprintf(stderr, "Error: unable to open %s\n", path);
		return -1;
	}
	rr->looks_like_png = looks_like_png(rr->io);
	rr->surface_width = surface_width;
	rr->surface_height = surface_height;
	rr->surface_pixels = surface_pixels;
	rr->ypos = 0;
	rr->head = 0;
	rr->tail = 0;
	rr->count = 0;
	rr->decoder_done = 0;
	rr->scaler_done = 0;
	rr->aborted = 0;
	rr->mutex = NULL;
	rr->not_empty = NULL;
	rr->not_full = NULL;
	rr->decoder_thread = NULL;
	rr->scaler_thread = NULL;
	for (i = 0; i < LINE_QUEUE_DEPTH; i++) {
		rr->slots[i] = NULL;
	}

	if (rr->looks_like_png) {
		if (png_start(rr) < 0) {
			fclose(rr->io);
			return -1;
		}
	} else {
		if (jpeg_start(rr) < 0) {
			fclose(rr->io);
			return -1;
		}
	}

	for (i = 0; i < LINE_QUEUE_DEPTH; i++) {
		rr->slots[i] = malloc(rr->in_rowbytes);
		if (!rr->slots[i]) {
			while (i-- > 0) free(rr->slots[i]);
			if (rr->looks_like_png) png_end(rr); else jpeg_end(rr);
			fclose(rr->io);
			return -1;
		}
	}

	rr->mutex = SDL_CreateMutex();
	rr->not_empty = SDL_CreateCondition();
	rr->not_full = SDL_CreateCondition();
	if (!rr->mutex || !rr->not_empty || !rr->not_full) {
		fprintf(stderr, "Error: SDL sync init failed: %s\n", SDL_GetError());
		if (rr->mutex) SDL_DestroyMutex(rr->mutex);
		if (rr->not_empty) SDL_DestroyCondition(rr->not_empty);
		if (rr->not_full) SDL_DestroyCondition(rr->not_full);
		for (i = 0; i < LINE_QUEUE_DEPTH; i++) free(rr->slots[i]);
		if (rr->looks_like_png) png_end(rr); else jpeg_end(rr);
		fclose(rr->io);
		return -1;
	}

	rr->decoder_thread = SDL_CreateThread(decoder_thread_fn, "oil-decoder", rr);
	rr->scaler_thread = SDL_CreateThread(scaler_thread_fn, "oil-scaler", rr);
	if (!rr->decoder_thread || !rr->scaler_thread) {
		fprintf(stderr, "Error: SDL_CreateThread failed: %s\n", SDL_GetError());
		SDL_LockMutex(rr->mutex);
		rr->aborted = 1;
		SDL_BroadcastCondition(rr->not_empty);
		SDL_BroadcastCondition(rr->not_full);
		SDL_UnlockMutex(rr->mutex);
		if (rr->decoder_thread) SDL_WaitThread(rr->decoder_thread, NULL);
		if (rr->scaler_thread) SDL_WaitThread(rr->scaler_thread, NULL);
		SDL_DestroyMutex(rr->mutex);
		SDL_DestroyCondition(rr->not_empty);
		SDL_DestroyCondition(rr->not_full);
		for (i = 0; i < LINE_QUEUE_DEPTH; i++) free(rr->slots[i]);
		if (rr->looks_like_png) png_end(rr); else jpeg_end(rr);
		fclose(rr->io);
		return -1;
	}

	return 0;
}

static int resumable_resize_start_from_surface(struct resumable_resize *rr, char *path, SDL_Surface *surface)
{
	return resumable_resize_start(rr, path, surface->w, surface->h, surface->pixels);
}

static void resumable_resize_end(struct resumable_resize *rr)
{
	int i;

	SDL_LockMutex(rr->mutex);
	rr->aborted = 1;
	SDL_BroadcastCondition(rr->not_empty);
	SDL_BroadcastCondition(rr->not_full);
	SDL_UnlockMutex(rr->mutex);

	SDL_WaitThread(rr->decoder_thread, NULL);
	SDL_WaitThread(rr->scaler_thread, NULL);

	SDL_DestroyMutex(rr->mutex);
	SDL_DestroyCondition(rr->not_empty);
	SDL_DestroyCondition(rr->not_full);

	for (i = 0; i < LINE_QUEUE_DEPTH; i++) {
		free(rr->slots[i]);
	}

	if (rr->looks_like_png) {
		png_end(rr);
	} else {
		jpeg_end(rr);
	}
	fclose(rr->io);
}

static void clear_surface(SDL_Surface *surface)
{
	memset(surface->pixels, 0, surface->pitch * surface->h);
}

static void letterbox_blit(SDL_Surface *snap, SDL_Surface *dest)
{
	float scale_x, scale_y, scale;
	SDL_Rect dst_rect;

	scale_x = (float)dest->w / snap->w;
	scale_y = (float)dest->h / snap->h;
	scale = scale_x < scale_y ? scale_x : scale_y;

	dst_rect.w = (int)(snap->w * scale);
	dst_rect.h = (int)(snap->h * scale);
	dst_rect.x = (dest->w - dst_rect.w) / 2;
	dst_rect.y = (dest->h - dst_rect.h) / 2;

	clear_surface(dest);
	SDL_BlitSurfaceScaled(snap, NULL, dest, &dst_rect, SDL_SCALEMODE_LINEAR);
}

int main(int argc, char **argv) {
	SDL_Window *window;
	SDL_Surface *surface;
	SDL_Event event;
	char *path;
	int event_happened, render_in_progress, surface_is_dirty;
	int last_displayed_ypos;
	struct resumable_resize rr;
	Uint64 lastUpdateTime, currentTime, elapsed_time, resize_start_time;
	SDL_Surface *snapshot = NULL;
	Uint64 pending_resize_time = 0;
	#define DEBOUNCE_MS 150
	int argi;

	path = NULL;
	backend = BACKEND_DEFAULT;
	for (argi = 1; argi < argc; argi++) {
		if (strcmp(argv[argi], "--scalar") == 0) {
			backend = BACKEND_SCALAR;
		} else if (strcmp(argv[argi], "--sse2") == 0) {
			backend = BACKEND_SSE2;
		} else if (strcmp(argv[argi], "--avx2") == 0) {
			backend = BACKEND_AVX2;
		} else if (strcmp(argv[argi], "--neon") == 0) {
			backend = BACKEND_NEON;
		} else {
			path = argv[argi];
		}
	}
	if (!path) {
		fprintf(stderr, "Usage: %s [--scalar|--sse2|--avx2|--neon] <image>\n", argv[0]);
		return 1;
	}

	if (backend == BACKEND_DEFAULT) {
#if defined(__x86_64__)
		if (__builtin_cpu_supports("avx2")) {
			backend = BACKEND_AVX2;
		} else {
			backend = BACKEND_SSE2;
		}
#elif defined(__aarch64__)
		backend = BACKEND_NEON;
#else
		backend = BACKEND_SCALAR;
#endif
	}

	switch (backend) {
	case BACKEND_DEFAULT:
		/* unreachable */
		break;
	case BACKEND_SCALAR:
		scale_in_fn = oil_scale_in;
		scale_out_fn = oil_scale_out;
		break;
	case BACKEND_SSE2:
#if defined(__x86_64__)
		scale_in_fn = oil_scale_in_sse2;
		scale_out_fn = oil_scale_out_sse2;
#else
		fprintf(stderr, "Error: SSE2 backend not available on this architecture\n");
		return 1;
#endif
		break;
	case BACKEND_AVX2:
#if defined(__x86_64__)
		scale_in_fn = oil_scale_in_avx2;
		scale_out_fn = oil_scale_out_avx2;
#else
		fprintf(stderr, "Error: AVX2 backend not available on this architecture\n");
		return 1;
#endif
		break;
	case BACKEND_NEON:
#if defined(__aarch64__)
		scale_in_fn = oil_scale_in_neon;
		scale_out_fn = oil_scale_out_neon;
#else
		fprintf(stderr, "Error: NEON backend not available on this architecture\n");
		return 1;
#endif
		break;
	}

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

	surface = SDL_GetWindowSurface(window);
	clear_surface(surface);
	lastUpdateTime = 0;
	resize_start_time = 0;
	last_displayed_ypos = 0;
	if (resumable_resize_start_from_surface(&rr, path, surface) < 0) {
		SDL_Quit();
		return 1;
	}
	render_in_progress = 1;
	surface_is_dirty = 1;

	while (1) {
		if (render_in_progress) {
			event_happened = SDL_WaitEventTimeout(&event, 1000 / 60);
		} else if (pending_resize_time) {
			event_happened = SDL_WaitEventTimeout(&event, DEBOUNCE_MS);
		} else {
			event_happened = SDL_WaitEvent(&event);
		}

		if (event_happened && event.type == SDL_EVENT_QUIT) {
			if (render_in_progress) {
				resumable_resize_end(&rr);
			}
			if (snapshot)
				SDL_DestroySurface(snapshot);
			SDL_Quit();
			return 0;
		}

		if (event_happened && event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
			if (render_in_progress) {
				resumable_resize_end(&rr);
				render_in_progress = 0;
			}
			surface = SDL_GetWindowSurface(window);
			if (snapshot) {
				letterbox_blit(snapshot, surface);
				SDL_UpdateWindowSurface(window);
				surface_is_dirty = 0;
			} else {
				clear_surface(surface);
				surface_is_dirty = 1;
			}
			pending_resize_time = SDL_GetTicks();
		}

		if (event_happened && event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F5) {
			if (render_in_progress) {
				resumable_resize_end(&rr);
				render_in_progress = 0;
			}
			pending_resize_time = 0;
			surface = SDL_GetWindowSurface(window);
			clear_surface(surface);
			resize_start_time = SDL_GetTicks();
			last_displayed_ypos = 0;
			if (resumable_resize_start_from_surface(&rr, path, surface) == 0) {
				render_in_progress = 1;
				surface_is_dirty = 1;
			}
		}

		if (!render_in_progress && pending_resize_time > 0) {
			currentTime = SDL_GetTicks();
			if (currentTime - pending_resize_time >= DEBOUNCE_MS) {
				pending_resize_time = 0;
				surface = SDL_GetWindowSurface(window);
				clear_surface(surface);
				resize_start_time = SDL_GetTicks();
				last_displayed_ypos = 0;
				if (resumable_resize_start_from_surface(&rr, path, surface) == 0) {
					render_in_progress = 1;
					surface_is_dirty = 1;
				}
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
				if (snapshot)
					SDL_DestroySurface(snapshot);
				snapshot = SDL_CreateSurface(rr.out_width, rr.out_height, surface->format);
				{
					int y, center_offset;
					center_offset = resize_center_offset(&rr);
					for (y = 0; y < rr.out_height; y++) {
						memcpy(
							(unsigned char *)snapshot->pixels + y * snapshot->pitch,
							(unsigned char *)surface->pixels + center_offset + y * surface->w * 4,
							rr.out_width * 4
						);
					}
				}
				resumable_resize_end(&rr);
				SDL_UpdateWindowSurface(window);
				surface_is_dirty = 0;
				lastUpdateTime = SDL_GetTicks();
				elapsed_time = SDL_GetTicks() - resize_start_time;
				fprintf(stderr, "Resize ticks: %llu\n", (unsigned long long)elapsed_time);
			} else if (local_ypos > last_displayed_ypos) {
				last_displayed_ypos = local_ypos;
				surface_is_dirty = 1;
			}
		}

		if (surface_is_dirty) {
			currentTime = SDL_GetTicks();
			elapsed_time = currentTime - lastUpdateTime;
			if (elapsed_time >= 1000 / 60) {
				SDL_UpdateWindowSurface(window);
				surface_is_dirty = 0;
				lastUpdateTime = currentTime;
			}
		}
	}
}
