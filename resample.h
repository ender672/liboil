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
#include <stddef.h>

/**
 * Calculate the required length for a padded scanline and get the offset at
 * which the image samples should be be filled in.
 *
 * in_width, out_width - input/output dimensions in samples.
 * cmp - components per sample
 * offset - the offset at which the image samples should be filled in is
 *   returned
 *
 * Example use:
 *   len = padded_sl_len_ofset(in_width, out_width, cmp, &offset);
 *   buf = malloc(len);
 *   psl_pos0 = buf + offset;
 *   // populate with in_width samples starting at psl_pos0
 *   padded_sl_extend_edges(buf, in_width, offset, cmp);
 *   xscale_padded(psl_pos0, in_width, outbuf, out_width, cmp, 0);
 *   ...
 */
size_t padded_sl_len_offset(uint32_t in_width, uint32_t out_width,
	uint8_t cmp, size_t *offset);

/**
 * Scale scanline in to the scanline out. This is the simplest way to perform
 * x-scaling on a scanline, but it currently involves an extry memory allocation
 * and copy that may be avoidable if you use the lower-level padded_len_offset()
 * and xscale_padded() functions.
 *
 * in - pointer to input scanline
 * in_width - width in samples of input scanline
 * out - pointer to buffer where the output scanline will be written. It must be
 *   at least (size_t)out_width * cmp bytes in length.
 * out_width - width in samles of output scanline
 * cmp - number of components per sample
 *
 * returns 0 on success, otherwise a negative integer:
 *
 * -1 - bad input parameter
 * -2 - unable to perform an allocation
 * Extend the first and last sample into the padded area of a padded scanline.
 */
void padded_sl_extend_edges(uint8_t *buf, uint32_t width, size_t pad_len,
	uint8_t cmp);

/**
 * Scale padded scanline in to scanline out.
 */
int xscale_padded(uint8_t *in, uint32_t in_width, uint8_t *out,
	uint32_t out_width, uint8_t cmp, int filler);

/**
 * Indicate how many taps will be required to scale an image. The number of taps
 * required indicates how tall a strip needs to be.
 *
 * When dim_in is very large and dim_out is very small, this can exceed the max
 * size of uint32_t, so returns a uint64_t.
 */
uint64_t calc_taps(uint32_t dim_in, uint32_t dim_out);

/**
 * Given input & output dimensions and an output position, return the
 * corresponding input position and put the sub-pixel remainder in rest.
 */
int32_t split_map(uint32_t dim_in, uint32_t dim_out, uint32_t pos, float *rest);

/**
 * Scale a strip. The height parameter indicates the height of the strip, not
 * the height of the image.
 *
 * The strip_height parameter indicates how many scanlines we are passing in. It
 * must be a multiple of 4.
 *
 * The in parameter points to an array of scanlines, each len bytes. There must
 * be at least strip_height scanlines in the array.
 *
 * The ty parameter indicates how far our mapped sampling position is from the
 * center of the strip.
 *
 * Note that all scanlines in the strip must be populated, even when this
 * requires scanlines that are less than 0 or larger than the height of the
 * source image.
 */
int strip_scale(uint8_t **in, uint32_t strip_height, size_t len, uint8_t *out,
	float ty, uint8_t cmp, int filler);

/**
 * Struct to hold state for x-scaling.
 */
struct xscaler {
	uint8_t *psl_buf;
	size_t psl_offset;
	uint32_t width_in;
	uint32_t width_out;
	uint8_t cmp;
	int filler;
};

int xscaler_init(struct xscaler *xs, uint32_t width_in, uint32_t width_out,
	uint8_t cmp, int filler);
void xscaler_free(struct xscaler *xs);
uint8_t *xscaler_psl_pos0(struct xscaler *xs);
void xscaler_scale(struct xscaler *xs, uint8_t *out_buf);

/**
 * struct sl_rbuf manages scanlines for y scaling. It implements a ring buffer
 * for storing scanlines.
 */
struct sl_rbuf {
	uint32_t height; // number of scanlines that the ring buffer can hold
	size_t length; // width in bytes of each scanline in the buffer
	uint32_t count; // total no. of scanlines that have been fed in
	uint8_t *buf; // buffer for the ring buffer
	uint8_t **virt; // space to provide scanline pointers for scaling
};

/**
 * Initialize a yscaler struct. Calculates how large the scanline ring buffer
 * will need to be and allocates it.
 */
int sl_rbuf_init(struct sl_rbuf *rb, uint32_t height, size_t sl_len);

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
uint8_t **sl_rbuf_virt(struct sl_rbuf *rb, uint32_t target);

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
int yscaler_init(struct yscaler *ys, uint32_t in_height, uint32_t out_height,
	size_t scanline_len);

/**
 * Free a yscaler struct, including the ring buffer.
 */
void yscaler_free(struct yscaler *ys);

/**
 * Get a pointer to the next scanline to be filled in the ring buffer. Returns
 * null if no more scanlines are needed to perform scaling.
 */
uint8_t *yscaler_next(struct yscaler *ys);

/**
 * Scale the buffered contents of the yscaler to produce the next scaled output
 * scanline.
 *
 * Scaled scanline will be written to the out parameter.
 * The width parameter is the nuber of samples in each scanline.
 * The cmp parameter is the number of components per sample (3 for RGB).
 * The pos parameter is the position of the output scanline.
 */
int yscaler_scale(struct yscaler *ys, uint8_t *out, uint32_t pos, uint8_t cmp,
	int filler);

/**
 * Helper for scaling an image that sits fully in memory.
 */
int yscaler_prealloc_scale(uint32_t in_height, uint32_t out_height,
	uint8_t **in, uint8_t *out, uint32_t pos, uint32_t width, uint8_t cmp,
	int filler);

/**
 * Utility helpers.
 */
void fix_ratio(uint32_t src_width, uint32_t src_height, uint32_t *out_width,
	uint32_t *out_height);
int cubic_scale_denom(uint32_t src_dim, uint32_t out_dim);

#endif
