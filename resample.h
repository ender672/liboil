#ifndef RESAMPLE_H
#define RESAMPLE_H

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

#endif
