/* SPDX-License-Identifier: MIT */

#ifndef OIL_LIBPNG_H
#define OIL_LIBPNG_H

#include <stdio.h>
#include <png.h>
#include "oil_resample.h"

struct oil_libpng {
	/* Public. The configured scaler: feed it with oil_scale_in (or a SIMD
	 * entry point) and read scaled rows with oil_scale_out. */
	struct oil_scale os;

	/* Private -- internal decode state; do not read or write from outside
	 * the wrapper. @rpng/@rinfo are the borrowed libpng read structs;
	 * @inbuf is the wrapper's own scratch row buffer (non-interlaced);
	 * @inimage holds the fully decoded image (interlaced Adam7, which
	 * forbids row-skipping); @in_vpos is the next source row to serve;
	 * @inbuf_offset is the byte offset of the fed rect within a decoded
	 * row; @img_height sizes @inimage for freeing. */
	png_structp rpng;
	png_infop rinfo;
	int in_vpos;
	int inbuf_offset;
	int img_height;
	unsigned char *inbuf;
	unsigned char **inimage;

	/* Public. Size of one decoded input row, for callers that drive the
	 * decode themselves: oil_libpng_decode_row writes fed_width *
	 * components bytes into the caller-supplied buffer. */
	int fed_width;
	int components;
};

/**
 * Initialize an oil_libpng struct.
 * @ol: Pointer to the struct to be initialized.
 * @rpng, @rinfo: Active libpng read structs, with header already read.
 * @out_height: Desired height, in pixels, of the output image.
 * @out_width: Desired width, in pixels, of the output image.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 */
int oil_libpng_init(struct oil_libpng *ol, png_structp rpng, png_infop rinfo,
	int out_width, int out_height);

/**
 * Initialize an oil_libpng struct with a sub-pixel source rect.
 *
 * The wrapper computes the required fed input rect (with halo for the
 * Catmull-Rom kernel) via oil_required_input_rect and advances the libpng
 * decoder past rows outside the rect (non-interlaced). Interlaced PNGs are
 * always read in full at init time, since Adam7 forbids row-skipping.
 *
 * @ol: Pointer to the struct to be initialized.
 * @rpng, @rinfo: Active libpng read structs.
 * @out_width, @out_height: Desired output dimensions in pixels.
 * @src_x, @src_y, @src_width, @src_height: Source rect inside the full image,
 *     in source pixels (may be fractional). Must fit within the image bounds.
 * @cs_override: If OIL_CS_UNKNOWN, the wrapper derives the scaler's
 *     colorspace from png_get_color_type. Otherwise the override is
 *     passed to oil_scale_init_ex; this is how callers select the
 *     no-gamma variants when the decoded bytes are RGB(A) but should
 *     be scaled in the file's native gamma. The override must have the
 *     same OIL_CMP as the derived colorspace.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 * Returns -2 if unable to allocate memory.
 */
int oil_libpng_init_ex(struct oil_libpng *ol, png_structp rpng, png_infop rinfo,
	int out_width, int out_height,
	double src_x, double src_y, double src_width, double src_height,
	enum oil_colorspace cs_override);

/* Release the scaler and decode buffers. Does not touch the caller's
 * libpng read structs. */
void oil_libpng_free(struct oil_libpng *ol);

/**
 * Decode the next input row from the PNG into a caller-supplied buffer.
 *
 * @ol: Initialized wrapper.
 * @dst: Destination buffer of at least ol->fed_width * ol->components bytes.
 *     On success, holds one row of decoded pixels in the wrapper's
 *     colorspace, restricted to the cropped fed rect.
 *
 * Callers driving the scaler themselves (e.g., to use SIMD entry points or
 * to interpose a slot queue between decode and scale) use this in place of
 * the bundled oil_libpng_read_scanline. For interlaced PNGs the row is
 * served from the pre-decoded full-image buffer; for non-interlaced PNGs
 * it triggers one png_read_row.
 */
void oil_libpng_decode_row(struct oil_libpng *ol, unsigned char *dst);

/* Bundled all-in-one path: decode enough input rows to scale and emit one
 * output row into @outbuf. Drives the scaler internally (scalar entry points).
 */
void oil_libpng_read_scanline(struct oil_libpng *ol, unsigned char *outbuf);

/* Map a libpng color type to its oil colorspace, or OIL_CS_UNKNOWN. */
enum oil_colorspace png_cs_to_oil(png_byte cs);

#endif
