#ifndef OIL_RESAMPLE_INTERNAL_H
#define OIL_RESAMPLE_INTERNAL_H

#if defined(__x86_64__) && !defined(OIL_NO_SIMD)
#define OIL_USE_SSE2
#endif

/* Lookup tables shared between oil_resample.c and arch-specific files. */
extern float s2l_map[256];
extern float i2f_map[256];
extern unsigned char *l2s_map;
extern int l2s_len;

/* SSE2-optimized functions (oil_resample_sse2.c) */
#if defined(OIL_USE_SSE2)
void oil_shift_left_f_sse2(float *f);
void oil_yscale_out_nonlinear_sse2(float *sums, int len, unsigned char *out);
void oil_yscale_out_linear_sse2(float *sums, int len, unsigned char *out);
void oil_scale_down_g_sse2(unsigned char *in, float *sums_y, int out_width,
	float *coeffs_x, int *border_buf, float *coeffs_y);
void oil_scale_down_ga_sse2(unsigned char *in, float *sums_y, int out_width,
	float *coeffs_x, int *border_buf, float *coeffs_y);
void oil_scale_down_rgb_sse2(unsigned char *in, float *sums_y, int out_width,
	float *coeffs_x, int *border_buf, float *coeffs_y);
void oil_scale_down_rgbx_sse2(unsigned char *in, float *sums_y, int out_width,
	float *coeffs_x, int *border_buf, float *coeffs_y);
void oil_scale_down_rgba_sse2(unsigned char *in, float *sums_y, int out_width,
	float *coeffs_x, int *border_buf, float *coeffs_y);
void oil_xscale_up_g_sse2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf);
void oil_xscale_up_ga_sse2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf);
void oil_yscale_up_g_cmyk_sse2(float **in, int len, float *coeffs,
	unsigned char *out);
void oil_yscale_up_rgb_sse2(float **in, int len, float *coeffs,
	unsigned char *out);
void oil_xscale_up_rgb_sse2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf);
void oil_xscale_up_rgbx_sse2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf);
void oil_yscale_up_ga_sse2(float **in, int len, float *coeffs,
	unsigned char *out);
void oil_yscale_out_ga_sse2(float *sums, int width, unsigned char *out);
void oil_yscale_out_rgbx_sse2(float *sums, int width, unsigned char *out);
void oil_yscale_up_rgbx_sse2(float **in, int len, float *coeffs,
	unsigned char *out);
void oil_yscale_out_rgba_sse2(float *sums, int width, unsigned char *out);
void oil_yscale_up_rgba_sse2(float **in, int len, float *coeffs,
	unsigned char *out);
void oil_xscale_up_rgba_sse2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf);
void oil_scale_down_cmyk_sse2(unsigned char *in, float *sums_y, int out_width,
	float *coeffs_x, int *border_buf, float *coeffs_y);
void oil_yscale_out_cmyk_sse2(float *sums, int len, unsigned char *out);
#endif

#endif
