#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>

#define HIST_C2_BITS  5 /* bits of precision in B/R histogram */
#define HIST_C2_ELEMS  (1<<HIST_C2_BITS)

typedef int16_t FSERROR;          /* 16 bits should be enough */
typedef FSERROR *FSERRPTR;      /* pointer to error array */

typedef uint16_t histcell;  /* histogram cell; prefer an unsigned type */
typedef histcell * histptr; /* for pointers to histogram cells */
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

void jinit_2pass_quantizer(struct quant *cquantize);
void prescan_quantize(struct quant *cquantize, unsigned char **input_buf,
	unsigned char **output_buf, int num_rows);
void finish_pass1(struct quant *cquantize);
void start_pass_2_quant(struct quant *cquantize);
void pass2_fs_dither(struct quant *cquantize, unsigned char **input_buf,
	unsigned char **output_buf, int num_rows);
void quant_free(struct quant *cquantize);

#endif
