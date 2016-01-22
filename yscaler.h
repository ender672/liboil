#ifndef YSCALER_H
#define YSCALER_H

#include <stdint.h>

struct yscaler {
	uint32_t in_height;
	uint32_t out_height;
	uint32_t taps;
	uint32_t scanline_len;
	uint8_t *strip;
	uint32_t sl_count;
	uint32_t target;
	float ty;
};

void yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	uint32_t scanline_len);
void yscaler_free(struct yscaler *ys);
unsigned char *yscaler_next(struct yscaler *ys);
void yscaler_scale(struct yscaler *ys, uint8_t *out,  uint32_t width,
	uint8_t cmp, uint8_t opts, uint32_t pos);
void yscaler_prealloc_scale(uint32_t in_height, uint32_t out_height,
	uint8_t **in, uint8_t *out, uint32_t pos, uint32_t width, uint8_t cmp,
	uint8_t opts);

#endif
