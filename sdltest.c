#include <stdio.h>
#include <SDL2/SDL.h>
#include "oil_resample.h"
#include "oil_libjpeg.h"
#include "oil_libpng.h"
#include <jpeglib.h>
#include <png.h>
struct resumable_resize {
	FILE *io;
	int looks_like_png;
	int surface_width;
	int surface_height;
	int out_width;
	int out_height;
	int ypos;
	unsigned char *outbuf;

	struct oil_libjpeg *olj;
	struct jpeg_decompress_struct *dinfo;

	struct oil_libpng *olp;
	png_structp rpng;
	png_infop rinfo;

	unsigned char *surface_pixels;
};

static void translate(unsigned char *in, unsigned char *out, int width) {
	int i;
	for (i=0; i<width; i++) {
		out[0] = in[2];
		out[1] = in[1];
		out[2] = in[0];
		out[3] = 0xFF;
		in += 3;
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

static void png_start(struct resumable_resize *rr) {
	png_structp rpng;
	png_infop rinfo;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (setjmp(png_jmpbuf(rpng))) {
		return;
	}

	rinfo = png_create_info_struct(rpng);
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
	rr->outbuf = malloc(rr->out_width * OIL_CMP(rr->olp->os.cs));
}

static void png_end(struct resumable_resize *rr) {

}

static void jpeg_start(struct resumable_resize *rr)
{
	struct jpeg_decompress_struct *dinfo;

	dinfo = malloc(sizeof(struct jpeg_decompress_struct));
	rr->dinfo = dinfo;
	dinfo->err = malloc(sizeof(struct jpeg_error_mgr));
	jpeg_std_error(dinfo->err);
	jpeg_create_decompress(dinfo);

	jpeg_stdio_src(dinfo, rr->io);
	jpeg_read_header(dinfo, TRUE);
	jpeg_calc_output_dimensions(dinfo);

	jpeg_start_decompress(dinfo);

	rr->out_width = rr->surface_width;
	rr->out_height = rr->surface_height;
	oil_fix_ratio(dinfo->output_width, dinfo->output_height, &rr->out_width, &rr->out_height);

	rr->olj = malloc(sizeof(struct oil_libjpeg));
	oil_libjpeg_init(rr->olj, dinfo, rr->out_width, rr->out_height);
	rr->outbuf = malloc(rr->out_width * 3);
}

static void jpeg_end(struct resumable_resize *rr) {
	free(rr->outbuf);
	oil_libjpeg_free(rr->olj);
	jpeg_destroy_decompress(rr->dinfo);
	free(rr->olj);
	free(rr->dinfo->err);
	free(rr->dinfo);
}

static void resumable_resize_start(struct resumable_resize *rr, char *path, int surface_width, int surface_height, unsigned char *surface_pixels)
{
	rr->io = fopen(path, "r");
	rr->looks_like_png = looks_like_png(rr->io);
	rr->surface_width = surface_width;
	rr->surface_height = surface_height;
	rr->surface_pixels = surface_pixels;
	rr->ypos = 0;
	if (rr->looks_like_png) {
		png_start(rr);
	} else {
		jpeg_start(rr);
	}
}

static void resumable_resize_start_from_surface(struct resumable_resize *rr, char *path, SDL_Surface *surface)
{
	resumable_resize_start(rr, path, surface->w, surface->h, surface->pixels);
}

static int resumable_resize_do(struct resumable_resize *rr) {
	int scanline_ready, center_offset, line_buf_offset;
	unsigned char *tmp;

	center_offset = resize_center_offset(rr);
	line_buf_offset = rr->ypos * rr->surface_width * 4;
	tmp = rr->surface_pixels + center_offset + line_buf_offset;

	if (rr->looks_like_png) {
		scanline_ready = oil_libpng_proccess_scanline_part(rr->olp);
		if (!scanline_ready) {
			return -2;
		}
		oil_libpng_read_scanline(rr->olp, rr->outbuf);
		rr->ypos += 1;
	} else {
		scanline_ready = oil_libjpeg_proccess_scanline_part(rr->olj);
		if (!scanline_ready) {
			return -2;
		}
		oil_libjpeg_read_scanline(rr->olj, rr->outbuf);
				rr->ypos += 1;
	}
	translate(rr->outbuf, tmp, rr->out_width);
	return rr->ypos == rr->out_height ? 0 : -1;
}

static void resumable_resize_end(struct resumable_resize *rr)
{
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

int main(int argc, char **argv) {
	SDL_Window *window;
	SDL_Surface *surface;
	SDL_Event event;
	char *path;
	int ret, event_happened, render_in_progress, surface_is_dirty;
	struct resumable_resize rr;
	Uint32 lastUpdateTime, currentTime, elapsed_time, resize_start_time;

	path = argv[1];

	SDL_Init(SDL_INIT_VIDEO);
	window = SDL_CreateWindow(path, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 640, 480, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	surface = SDL_GetWindowSurface(window);
	clear_surface(surface);
	lastUpdateTime = 0;
	resize_start_time = 0;
	resumable_resize_start_from_surface(&rr, path, surface);
	render_in_progress = 1;
	surface_is_dirty = 1;

	while (1) {
		event_happened = render_in_progress ? SDL_PollEvent(&event) : SDL_WaitEvent(&event);

		if (event_happened && event.type == SDL_QUIT) {
			if (render_in_progress) {
				resumable_resize_end(&rr);
			}
			SDL_Quit();
			return 0;
		}

		if (
			(event_happened && event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
			|| (event_happened && event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_F5)
		) {
			if (render_in_progress) {
				resumable_resize_end(&rr);
			}
			surface = SDL_GetWindowSurface(window);
			clear_surface(surface);
			resize_start_time = SDL_GetTicks();
			resumable_resize_start_from_surface(&rr, path, surface);
			render_in_progress = 1;
			surface_is_dirty = 1;
		}

		if (render_in_progress) {
			ret = resumable_resize_do(&rr);
			if (ret == 0) { // 0 means the image resize finished
				render_in_progress = 0;
				resumable_resize_end(&rr);
				SDL_UpdateWindowSurface(window);
				surface_is_dirty = 0;
				lastUpdateTime = SDL_GetTicks();
				elapsed_time = SDL_GetTicks() - resize_start_time;
				fprintf(stderr, "Resize ticks: %d\n", elapsed_time);
			} else if (ret == -1) { // -1 means one scanline finished
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
