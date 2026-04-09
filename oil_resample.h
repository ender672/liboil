/**
 * Copyright (c) 2014-2019 Timothy Elliott
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

#ifndef OIL_RESAMPLE_H
#define OIL_RESAMPLE_H

#define OIL_VERSION_MAJOR 0
#define OIL_VERSION_MINOR 2
#define OIL_VERSION_PATCH 0

/**
 * Color spaces currently supported by oil.
 */
enum oil_colorspace {
	// error
	OIL_CS_UNKNOWN = 0,

	// RGBA without sRGB linearization - premultiplied alpha, no gamma
	OIL_CS_RGBA_NOGAMMA = 0x0604,

	// RGBX without sRGB linearization - 4 bytes per pixel, 4th byte ignored
	OIL_CS_RGBX_NOGAMMA = 0x0704,
};

/**
 * Macro to get the number of components from an oil color space.
 */
#define OIL_CMP(x) ((x)&0xFF)

/**
 * Struct to hold state for scaling. Changing these will produce unpredictable
 * results.
 */
struct oil_scale {
	int in_height; // input image height.
	int out_height; // output image height.
	int in_width; // input image width.
	int out_width; // output image height.
	enum oil_colorspace cs; // color space of input & output.
	int in_pos; // current row of input image.
	int out_pos; // current row of output image.
	float *coeffs_y; // buffer for holding temporary y-coefficients.
	float *coeffs_x; // buffer for holding precalculated coefficients.
	int *borders_x; // holds precalculated coefficient rotation points.
	int *borders_y; // coefficient rotation points for y-scaling.
	float *sums_y; // buffer of intermediate sums for y-scaling.
	float *tmp_coeffs; // temporary buffer for calculating coeffs.
	void *buf; // single backing allocation for all buffers above.
	int sums_y_tap; // ring buffer offset for sums_y (0-3).
};

/**
 * Initialize static, pre-calculated tables. This only needs to be called once.
 * A call to oil_scale_init() will initialize these tables if not already done,
 * so explicityly calling oil_global_init() is only needed if there are
 * concurrency concerns.
 */
void oil_global_init(void);

/**
 * Calculate the buffer size needed for an oil scaler struct.
 * @in_height: Height, in pixels, of the input image.
 * @out_height: Height, in pixels, of the output image.
 * @in_width: Width, in pixels, of the input image.
 * @out_width: Width, in pixels, of the output image.
 * @cs: Color space of the input/output images.
 *
 * Returns the required buffer size in bytes.
 */
int oil_scale_alloc_size(int in_height, int out_height, int in_width,
	int out_width, enum oil_colorspace cs);

/**
 * Initialize an oil scaler struct with a pre-allocated buffer.
 * @os: Pointer to the scaler struct to be initialized.
 * @in_height: Height, in pixels, of the input image.
 * @out_height: Height, in pixels, of the output image.
 * @in_width: Width, in pixels, of the input image.
 * @out_width: Width, in pixels, of the output image.
 * @cs: Color space of the input/output images.
 * @buf: Pre-allocated buffer for internal use.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 */
int oil_scale_init_allocated(struct oil_scale *os, int in_height,
	int out_height, int in_width, int out_width, enum oil_colorspace cs,
	void *buf);

/**
 * Initialize an oil scaler struct.
 * @os: Pointer to the scaler struct to be initialized.
 * @in_height: Height, in pixels, of the input image.
 * @out_height: Height, in pixels, of the output image.
 * @in_width: Width, in pixels, of the input image.
 * @out_width: Width, in pixels, of the output image.
 * @cs: Color space of the input/output images.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 */
int oil_scale_init(struct oil_scale *os, int in_height, int out_height,
	int in_width, int out_width, enum oil_colorspace cs);

/**
 * Reset rows counters in an oil scaler struct.
 * @os: Pointer to the scaler struct to be reseted.
 */
void oil_scale_restart(struct oil_scale *);

/**
 * Free heap allocations associated with an oil scaler struct.
 * @os: Pointer to the scaler struct to be freed.
 */
void oil_scale_free(struct oil_scale *os);

/**
 * Return the number of input scanlines needed before the next output scanline
 * can be produced.
 * @os: Pointer to the oil scaler struct.
 *
 * Returns 0 if no more input lines are needed to produce the next output line.
 * Otherwise, returns the number of input lines that are needed.
 */
int oil_scale_slots(struct oil_scale *os);

/**
 * Ingest & buffer an input scanline. Input is unsigned chars.
 * @os: Pointer to the scaler struct.
 * @in: Pointer to the input buffer containing a scanline.
 *
 * Returns 0 on success.
 * Returns -1 if an output scanline is ready and must be consumed first via
 * oil_scale_out() or discarded via oil_scale_discard().
 */
int oil_scale_in(struct oil_scale *os, unsigned char *in);

/**
 * Scale previously ingested & buffered contents to produce the next scaled output
 * scanline.
 * @os: Pointer to the scaler struct.
 * @out: Pointer to the buffer where the output scanline will be written.
 *
 * Returns 0 on success.
 * Returns -1 if not enough input scanlines have been fed yet.
 */
int oil_scale_out(struct oil_scale *os, unsigned char *out);

/**
 * SSE2-optimized version of oil_scale_in().
 */
int oil_scale_in_sse2(struct oil_scale *os, unsigned char *in);

/**
 * SSE2-optimized version of oil_scale_out().
 */
int oil_scale_out_sse2(struct oil_scale *os, unsigned char *out);


/**
 * AVX2-optimized version of oil_scale_in().
 */
int oil_scale_in_avx2(struct oil_scale *os, unsigned char *in);

/**
 * AVX2-optimized version of oil_scale_out().
 */
int oil_scale_out_avx2(struct oil_scale *os, unsigned char *out);

/**
 * NEON-optimized version of oil_scale_in().
 */
int oil_scale_in_neon(struct oil_scale *os, unsigned char *in);

/**
 * NEON-optimized version of oil_scale_out().
 */
int oil_scale_out_neon(struct oil_scale *os, unsigned char *out);

/**
 * Discard the next output scanline without producing it. Advances internal
 * state so that input feeding can continue.
 * @os: Pointer to the scaler struct.
 *
 * Returns 0 on success.
 * Returns -1 if not enough input scanlines have been fed yet.
 */
int oil_scale_out_discard(struct oil_scale *os);

#endif
