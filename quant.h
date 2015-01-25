#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>

#define HIST_C2_BITS  5 /* bits of precision in B/R histogram */
#define HIST_C2_ELEMS  (1<<HIST_C2_BITS)

typedef int16_t FSERROR;          /* 16 bits should be enough */
typedef FSERROR *FSERRPTR;      /* pointer to error array */

typedef uint16_t histcell;  /* histogram cell; prefer an unsigned type */
typedef histcell hist1d[HIST_C2_ELEMS]; /* typedefs for the array */
typedef hist1d * hist2d;    /* type for the 2nd-level pointers */
typedef hist2d * hist3d;    /* type for top-level pointer */

struct quant {
	/* Space for the eventually created colormap is stashed here */
	unsigned char **sv_colormap;  /* colormap allocated at init time */
	int desired;                  /* desired # of colors = size of colormap */

	/* Variables for accumulating image statistics */
	hist3d histogram;             /* pointer to the histogram */

	/* Variables for Floyd-Steinberg dithering */
	FSERRPTR fserrors;            /* accumulated errors */
	int on_odd_row;               /* flag to remember which row we are on */
	int * error_limiter;          /* table for clamping the applied error */

	struct jpeg_color_quantizer * cquantize;
	unsigned int output_width;
	unsigned char **colormap;
	int actual_number_of_colors;
	int desired_number_of_colors;
};

void quant_init(struct quant *cquantize);
void quant_index(struct quant *cquantize, unsigned char *ptr);
void quant_gen_palette(struct quant *cquantize);
void quant_map(struct quant *cquantize, unsigned char *inptr, unsigned char *outptr);
void quant_free(struct quant *cquantize);

#endif
