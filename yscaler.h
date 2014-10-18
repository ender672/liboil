#ifndef YSCALER_H
#define YSCALER_H

#include "resample.h"
#include <stdint.h>

struct strip {
	uint32_t height;
	uint8_t **sl;
	uint8_t **virt;
	uint32_t rr;
};

struct yscaler {
	uint32_t in_height;
	uint32_t out_height;
	uint32_t width;
	enum oil_fmt fmt;

	struct strip strip;
	uint32_t in_pos;
	uint32_t out_pos;
	uint32_t in_target;
	float ty;
};

void yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	uint32_t width, enum oil_fmt fmt);
void yscaler_free(struct yscaler *ys);
unsigned char *yscaler_next(struct yscaler *ys);
void yscaler_scale(struct yscaler *ys, uint8_t *out);

#endif
