#include "yscaler.h"
#include "resample.h"
#include <stdlib.h>
#include <stdint.h>

static void yscaler_map_pos(struct yscaler *ys, uint32_t pos)
{
	uint32_t target;
	target = split_map(ys->in_height, ys->out_height, pos, &ys->ty);
	ys->target = target + ys->taps / 2 + 1;
}

void yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	uint32_t scanline_len)
{
	ys->in_height = in_height;
	ys->out_height = out_height;
	ys->sl_count = 0;
	ys->taps = calc_taps(in_height, out_height);
	ys->scanline_len = scanline_len;
	ys->strip = malloc(scanline_len * ys->taps);
	yscaler_map_pos(ys, 0);
}

void yscaler_free(struct yscaler *ys)
{
	free(ys->strip);
}

uint8_t *yscaler_next(struct yscaler *ys)
{
	if (ys->sl_count == ys->in_height || ys->sl_count >= ys->target + 1) {
		return 0;
	}

	return ys->strip + (ys->sl_count++ % ys->taps) * ys->scanline_len;
}

void yscaler_scale(struct yscaler *ys, uint8_t *out, uint32_t width,
	uint8_t cmp, uint8_t opts, uint32_t pos)
{
	void **virt, *ptr;
	uint32_t i, target_safe, taps;

	taps = ys->taps;
	virt = malloc(sizeof(uint8_t *) * taps);
	for (i=0; i<taps; i++) {
		target_safe = ys->target < i ? 0 : ys->target - i;
		if (target_safe > ys->in_height) {
			target_safe = ys->in_height;
		}
		ptr = ys->strip + (target_safe % taps) * ys->scanline_len;
		virt[taps - i - 1] = ptr;
	}
	strip_scale(virt, taps, width, (void *)out, ys->ty, cmp, opts);
	yscaler_map_pos(ys, pos + 1);
	free(virt);
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
	free(virt);
}
