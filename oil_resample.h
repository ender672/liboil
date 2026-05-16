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

	// greyscale - no sRGB gamma space conversions
	OIL_CS_G       = 0x0001,

	// greyscale w/ alpha - uses premultiplied alpha
	OIL_CS_GA      = 0x0002,

	// sRGB - input will be converted to linear RGB during processing
	OIL_CS_RGB     = 0x0003,

	// sRGB w/ alpha - sRGB to linear conversion and premultiplied alpha
	OIL_CS_RGBA    = 0x0104,

	// sRGB w/ alpha - alpha first, then sRGB to linear conversion and premultiplied alpha
	OIL_CS_ARGB    = 0x0304,

	// sRGB w/o alpha - 4 bytes per pixel, 4th byte (X) is ignored
	OIL_CS_RGBX    = 0x0404,

	// no color space conversions
	OIL_CS_CMYK    = 0x0204,

	// RGB without sRGB linearization - samples treated as raw values
	OIL_CS_RGB_NOGAMMA  = 0x0503,

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
	int in_height; // height of the fed input buffer (rows fed via oil_scale_in).
	int out_height; // output image height.
	int in_width; // width of the fed input buffer.
	int out_width; // output image width.
	double src_y_off; // start of logical source region within fed buffer (y).
	double src_x_off; // start of logical source region within fed buffer (x).
	double src_height; // logical source region height; drives y scale factor.
	double src_width; // logical source region width; drives x scale factor.
	enum oil_colorspace cs; // color space of input & output.
	int in_pos; // current row of input image.
	int out_pos; // current row of output image.
	float *coeffs_y; // buffer for holding temporary y-coefficients.
	float *coeffs_x; // buffer for holding precalculated coefficients.
	int *borders_x; // holds precalculated coefficient rotation points.
	int *borders_y; // coefficient rotation points for y-scaling.
	float *sums_y; // buffer of intermediate sums for y-scaling.
	float *rb; // ring buffer holding scanlines.
	float *tmp_coeffs; // temporary buffer for calculating coeffs.
	void *buf; // single backing allocation for all buffers above.
	int sums_y_tap; // ring buffer offset for sums_y (0-3).
	int slots_y; // live countdown into the current borders_y entry.
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
 * Returns the required buffer size in bytes. The caller must zero-initialize
 * the allocation before passing it to oil_scale_init_allocated.
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
 * @buf: Pre-allocated buffer for internal use; MUST be zero-initialized.
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
 * Initialize an oil scaler that consumes a fed input buffer of the given
 * dimensions but treats a sub-rectangle of that buffer as the logical source
 * for scaling. See oil_required_input_rect for sizing the fed buffer.
 *
 * @os: Pointer to the scaler struct to be initialized.
 * @in_height: Height of the fed input buffer (rows the caller will feed).
 * @out_height: Output image height in pixels.
 * @in_width: Width of the fed input buffer in pixels.
 * @out_width: Output image width in pixels.
 * @src_y: Start of the logical source rect within the fed buffer (rows;
 *         may be fractional).
 * @src_height: Logical source rect height; sets the y scale factor.
 * @src_x: Start of the logical source rect within the fed buffer (columns;
 *         may be fractional).
 * @src_width: Logical source rect width; sets the x scale factor.
 * @cs: Color space of the input/output images.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 *
 * Calling oil_scale_init_ex with src_y = src_x = 0, src_height = in_height,
 * src_width = in_width is equivalent to oil_scale_init.
 */
int oil_scale_init_ex(struct oil_scale *os, int in_height, int out_height,
	int in_width, int out_width, double src_y, double src_height,
	double src_x, double src_width, enum oil_colorspace cs);

/**
 * Same as oil_scale_init_ex but with a caller-supplied buffer. The buffer
 * MUST be zero-initialized (use calloc, or memset to 0).
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 */
int oil_scale_init_allocated_ex(struct oil_scale *os, int in_height,
	int out_height, int in_width, int out_width, double src_y,
	double src_height, double src_x, double src_width,
	enum oil_colorspace cs, void *buf);

/**
 * Calculate the buffer size for an oil scaler with the _ex parameters. The
 * returned size is the allocation that must be passed to
 * oil_scale_init_allocated_ex; the caller must zero-initialize the buffer.
 */
int oil_scale_alloc_size_ex(int in_height, int out_height, int in_width,
	int out_width, double src_height, double src_width,
	enum oil_colorspace cs);

/**
 * Compute the input rectangle the caller must feed to a scaler that operates
 * on a logical source rect inside a larger image. The returned rect is the
 * logical rect padded by the filter's half-support on each side and clamped
 * to the image bounds.
 *
 * @img_height/@img_width: Dimensions of the full source image.
 * @src_y, @src_height, @src_x, @src_width: The logical source rect.
 * @out_height, @out_width: Output image dimensions.
 * @fed_y, @fed_height, @fed_x, @fed_width: Output; the rect the caller's
 *         decoder must produce and feed into oil_scale_in.
 *
 * The caller passes (fed_y, fed_height, fed_x, fed_width) as the in_* args
 * to oil_scale_init_ex, and (src_y - fed_y, src_height, src_x - fed_x,
 * src_width) as the src_* args.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 */
int oil_required_input_rect(int img_height, int img_width, double src_y,
	double src_height, double src_x, double src_width, int out_height,
	int out_width, int *fed_y, int *fed_height, int *fed_x, int *fed_width);

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

/**
 * Calculate an output ratio that preserves the input aspect ratio.
 * @src_width: Width, in pixels, of the input image.
 * @src_height: Height, in pixels, of the input image.
 * @out_width: Width, in pixels, of the output bounding box.
 * @out_height: Height, in pixels, of the output bounding box.
 *
 * The out_width and out_height parameters will be modified, if necessary, to
 * maintain the input aspect ratio while staying within the given bounding box.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -3 if an adjusted dimension would be out of range.
 */
 int oil_fix_ratio(int src_width, int src_height, int *out_width,
	int *out_height);

/**
 * Gravity hints for oil_compute_cover_rect. The hint selects which edge or
 * corner of the source rect stays anchored when the rect is smaller than the
 * image on the cropped axis. Hints on a non-cropped axis are ignored.
 */
enum oil_gravity {
	OIL_GRAVITY_CENTER,
	OIL_GRAVITY_TOP,
	OIL_GRAVITY_BOTTOM,
	OIL_GRAVITY_LEFT,
	OIL_GRAVITY_RIGHT,
	OIL_GRAVITY_TOP_LEFT,
	OIL_GRAVITY_TOP_RIGHT,
	OIL_GRAVITY_BOTTOM_LEFT,
	OIL_GRAVITY_BOTTOM_RIGHT
};

/**
 * Compute a source crop rect that fills the output dimensions exactly,
 * preserving aspect (CSS "cover" semantics). The returned rect is the largest
 * sub-rect of the image that shares the output aspect ratio.
 *
 * Pair with oil_scale_init_ex (or the libjpeg/libpng init_ex wrappers) to
 * feed only the cropped region into the resampler.
 *
 * For "contain" semantics (fit the whole image inside the output, preserve
 * aspect, no crop), use oil_fix_ratio on the output dimensions instead. For
 * "stretch" semantics, pass the image dimensions as the src rect directly.
 *
 * @img_width, @img_height: Source image dimensions.
 * @out_width, @out_height: Desired output dimensions.
 * @gravity: Which edge of the source rect to anchor on the cropped axis.
 * @src_x, @src_y, @src_width, @src_height: Filled with the computed rect.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 */
int oil_compute_cover_rect(int img_width, int img_height,
	int out_width, int out_height, enum oil_gravity gravity,
	double *src_x, double *src_y, double *src_width, double *src_height);

/**
 * Compute a contain (bounding-box) source rect with exact aspect preservation.
 *
 * Adjusts the output dimensions to fit within the given bounding box while
 * approximating the input's aspect ratio (same rounding as oil_fix_ratio),
 * then returns a fractional source rect that, paired with those output dims,
 * gives exact aspect preservation by absorbing the rounding remainder as a
 * sub-pixel centered crop of the input.
 *
 * For bounding boxes whose aspect is close to the input's, the trim is
 * sub-pixel and the difference vs oil_fix_ratio alone is imperceptible. For
 * bounding boxes whose aspect differs sharply from the input (e.g. a
 * panorama scaled into a near-square box), the trim can remove a noticeable
 * slice of the input -- callers who want to preserve all input content
 * should use oil_fix_ratio alone and accept the mild aspect distortion.
 *
 * Pair with oil_scale_init_ex (or the libjpeg/libpng init_ex wrappers).
 *
 * @img_width, @img_height: Source image dimensions.
 * @out_width, @out_height: In/out. Caller passes the desired bounding box;
 *         these are adjusted to integer dims that approximate input aspect.
 * @src_x, @src_y, @src_width, @src_height: Filled with the computed rect.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -3 if an adjusted dimension would be out of range.
 */
int oil_compute_contain_rect(int img_width, int img_height,
	int *out_width, int *out_height,
	double *src_x, double *src_y, double *src_width, double *src_height);

#endif
