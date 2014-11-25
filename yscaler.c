#include "yscaler.h"
#include "resample.h"
#include <stdlib.h>
#include <stdint.h>

/**
 * Initializes a strip of height scanlines.
 *
 * Scanlines are allocated with a size of len bytes.
 */
static void strip_init(struct strip *st, uint32_t height, size_t buflen)
{
	uint32_t i;

	st->height = height;
	st->rr = 0;
	st->sl = malloc(height * sizeof(uint8_t *));
	st->virt = malloc(height * sizeof(uint8_t *));
	for (i=0; i<height; i++) {
		st->sl[i] = malloc(buflen);
		st->virt[i] = st->sl[0];
	}
}

/**
 * Frees memory allocated at initialization.
 */
static void strip_free(struct strip *st)
{
	uint32_t i;

	for (i=0; i<st->height; i++) {
		free(st->sl[i]);
	}
	free(st->sl);
	free(st->virt);
}

/**
 * Shifts the virtual scanline pointers downwards by one.
 *
 * The bottmomost virtual scanline pointer is left untouched since we may have
 * reached the end of the source image and we may want to leave it pointing at
 * the bottom.
 *
 * If we have not reached the bottom of the source image this call should be
 * followed up by strip_bottom_shift() to complete the strip shift.
 */
static void strip_top_shift(struct strip *st)
{
	uint32_t i;
	for (i=1; i<st->height; i++) {
		st->virt[i - 1] = st->virt[i];
	}
}

/**
 * Points the bottommost virtual scanline to the next allocated scanline,
 * discarding the topmost scanline.
 *
 * The caller is expected to fill the bottom virtual scanline with image data
 * after this call.
 */
static void strip_bottom_shift(struct strip *st)
{
	st->rr = (st->rr + 1) % st->height;
	st->virt[st->height - 1] = st->sl[st->rr];
}

/**
 * Returns a pointer to the bottommost virtual scanline.
 */
static void *strip_bottom(struct strip *st)
{
	return st->virt[st->height - 1];
}

static void yscaler_map_pos(struct yscaler *ys)
{
	uint32_t target;
	target = split_map(ys->in_height, ys->out_height, ys->out_pos, &ys->ty);
	ys->in_target = target + ys->strip.height / 2 + 1;
}

void yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	uint32_t buflen)
{
	uint32_t taps;

	ys->in_height = in_height;
	ys->out_height = out_height;
	ys->in_pos = 0;
	ys->out_pos = 0;

	taps = calc_taps(in_height, out_height);
	strip_init(&ys->strip, taps, buflen);
	yscaler_map_pos(ys);
}

void yscaler_free(struct yscaler *ys)
{
	strip_free(&ys->strip);
}

unsigned char *yscaler_next(struct yscaler *ys)
{
	struct strip *st;

	st = &ys->strip;

	/* We need the first scanline for top padding. */
	if (ys->in_pos == 0) {
		ys->in_pos++;
		return strip_bottom(st);
	}

	while (ys->in_pos < ys->in_target) {
		ys->in_pos++;
		strip_top_shift(&ys->strip);
		if (ys->in_pos <= ys->in_height) {
			strip_bottom_shift(st);
			return strip_bottom(st);
		}
	}

	return 0;
}

void yscaler_scale(struct yscaler *ys, uint8_t *out, uint32_t width,
	uint8_t cmp, uint8_t opts)
{
	struct strip *st;

	st = &ys->strip;
	strip_scale((void **)st->virt, st->height, width, (void *)out, ys->ty,
		cmp, opts);
	ys->out_pos++;
	yscaler_map_pos(ys);
}

void yscaler_prealloc_scale(uint32_t in_height, uint32_t out_height,
	uint8_t **in, uint8_t *out, uint32_t pos, uint32_t width, uint8_t cmp,
	uint8_t opts)
{
	uint32_t i, taps;
	int32_t smp_i, strip_pos;
	uint8_t **virt;
	float ty;

	taps = calc_taps(in_height, out_height);
	virt = malloc(taps * sizeof(uint8_t *));
	smp_i = split_map(in_height, out_height, pos, &ty);
	strip_pos = smp_i + 1 - taps / 2;

	for (i=0; i<taps; i++) {
		if (strip_pos < 0) {
			virt[i] = in[0];
		} else if ((uint32_t)strip_pos > in_height - 1) {
			virt[i] = in[in_height - 1];
		} else {
			virt[i] = in[strip_pos];
		}
		strip_pos++;
	}

	strip_scale((void **)virt, taps, width, (void *)out, ty, cmp, opts);
}
