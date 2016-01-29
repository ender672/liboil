/**
 * Copyright (c) 2014-2016 Timothy Elliott
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef RESAMPLE_H
#define RESAMPLE_H

#include <stdint.h>

/**
 * Use this for the opts argument to indicate that our samples have a 'filler'
 * component to improve memory alignment, for example RGBX.
 *
 * When given, this can result in a modest speed improvement since oil can
 * ignore the extra component.
 */
#define OIL_FILLER 1

/**
 * Scale scanline in to the scanline out.
 */
void xscale(unsigned char *in, long in_width, unsigned char *out,
	long out_width, int cmp, int opts);

/**
 * Indicate how many taps will be required to scale an image. The number of taps
 * required indicates how tall a strip needs to be.
 */
long calc_taps(long dim_in, long dim_out);

/**
 * Given input & output dimensions and an output position, return the
 * corresponding input position and put the sub-pixel remainder in rest.
 */
long split_map(unsigned long dim_in, unsigned long dim_out, unsigned long pos,
	float *rest);

/**
 * Scale a strip. The height parameter indicates the height of the strip, not
 * the height of the image.
 *
 * The strip_height parameter indicates how many scanlines we are passing in. It
 * must be a multiple of 4.
 *
 * The in parameter points to an array of scanlines, each with width samples in
 * sample_fmt format. There must be at least strip_height scanlines in the
 * array.
 *
 * The ty parameter indicates how far our mapped sampling position is from the
 * center of the strip.
 *
 * Note that all scanlines in the strip must be populated, even when this
 * requires scanlines that are less than 0 or larger than the height of the
 * source image.
 */
void strip_scale(void **in, long strip_height, long width, void *out, float ty,
	int cmp, int opts);

/**
 * struct sl_rbuf manages scanlines for y scaling. It implements a ring buffer
 * for storing scanlines.
 */
struct sl_rbuf {
	uint32_t height; // number of scanlines that the ring buffer can hold
	uint32_t length; // width in bytes of each scanline in the buffer
	uint32_t count; // total no. of scanlines that have been fed in
	uint8_t *buf; // buffer for the ring buffer
	uint8_t **virt; // space to provide scanline pointers for scaling
};

/**
 * Initialize a yscaler struct. Calculates how large the scanline ring buffer
 * will need to be and allocates it.
 */
void sl_rbuf_init(struct sl_rbuf *rb, uint32_t height, uint32_t sl_len);

/**
 * Free a sl_rbuf struct, including the ring buffer.
 */
void sl_rbuf_free(struct sl_rbuf *rb);

/**
 * Get a pointer to the next scanline to be filled in the ring buffer and
 * advance the internal counter of scanlines in the buffer.
 */
uint8_t *sl_rbuf_next(struct sl_rbuf *rb);

/**
 * Return an ordered array of scanline pointers for use in scaling.
 */
uint8_t **sl_rbuf_virt(struct sl_rbuf *rb, long target);

/**
 * Struct to hold state for y-scaling.
 */
struct yscaler {
	struct sl_rbuf rb; // ring buffer holding scanlines.
	uint32_t in_height; // input image height.
	uint32_t out_height; // output image height.
	uint32_t target; // where the ring buffer should be on next scaling.
	float ty; // sub-pixel offset for next scaling.
};

/**
 * Initialize a yscaler struct. Calculates how large the scanline ring buffer
 * will need to be and allocates it.
 */
void yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	uint32_t scanline_len);

/**
 * Free a yscaler struct, including the ring buffer.
 */
void yscaler_free(struct yscaler *ys);

/**
 * Get a pointer to the next scanline to be filled in the ring buffer. Returns
 * null if no more scanlines are needed to perform scaling.
 */
unsigned char *yscaler_next(struct yscaler *ys);

/**
 * Scale the buffered contents of the yscaler to produce the next scaled output
 * scanline.
 *
 * Scaled scanline will be written to the out parameter.
 * The width parameter is the nuber of samples in each scanline.
 * The cmp parameter is the number of components per sample (3 for RGB).
 * The opts parameter is a bit mask for options. Currently OIL_FILLER is the
 *   only one.
 * The pos parameter is the position of the output scanline.
 */
void yscaler_scale(struct yscaler *ys, uint8_t *out,  uint32_t width,
	uint8_t cmp, uint8_t opts, uint32_t pos);

/**
 * Helper for scaling an image that sits fully in memory.
 */
void yscaler_prealloc_scale(uint32_t in_height, uint32_t out_height,
	uint8_t **in, uint8_t *out, uint32_t pos, uint32_t width, uint8_t cmp,
	uint8_t opts);

#endif
