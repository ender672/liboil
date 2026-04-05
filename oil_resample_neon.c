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

#include "oil_resample_internal.h"

#ifdef OIL_USE_NEON

#include <arm_neon.h>

/* Helper: horizontal sum of a float32x4_t */
static inline float hsum_f32(float32x4_t v)
{
	float32x2_t lo = vget_low_f32(v);
	float32x2_t hi = vget_high_f32(v);
	float32x2_t sum = vadd_f32(lo, hi);
	sum = vpadd_f32(sum, sum);
	return vget_lane_f32(sum, 0);
}

/* Helper: pair-wise dot products of smp with c0 and c1.
 * Returns a float32x4_t where lane 0 = dot(smp,c0), lane 1 = dot(smp,c1). */
static inline float32x4_t dot2(float32x4_t smp, float32x4_t c0, float32x4_t c1)
{
	float32x4_t p0 = vmulq_f32(smp, c0);
	float32x4_t p1 = vmulq_f32(smp, c1);
	float32x4_t lo = vzip1q_f32(p0, p1);
	float32x4_t hh = vzip2q_f32(p0, p1);
	float32x4_t sum = vaddq_f32(lo, hh);
	float32x4_t t1 = vcombine_f32(vget_high_f32(sum), vget_high_f32(sum));
	float32x4_t t2 = vaddq_f32(sum, t1);
	return t2;
}

/* Helper: gather lane 0 from 4 float32x4_t vectors */
static inline float32x4_t gather_lane0(float32x4_t f0, float32x4_t f1,
	float32x4_t f2, float32x4_t f3)
{
	float32x4_t vals;
	vals = vsetq_lane_f32(vgetq_lane_f32(f0, 0), vdupq_n_f32(0), 0);
	vals = vsetq_lane_f32(vgetq_lane_f32(f1, 0), vals, 1);
	vals = vsetq_lane_f32(vgetq_lane_f32(f2, 0), vals, 2);
	vals = vsetq_lane_f32(vgetq_lane_f32(f3, 0), vals, 3);
	return vals;
}

/* Helper: shift float vector right by 1 lane (shift in zero) */
static inline float32x4_t shift_right_1(float32x4_t v)
{
	return vextq_f32(v, vdupq_n_f32(0), 1);
}

/* Helper: push_f - shift right by 1 lane, insert value at lane 3 */
static inline float32x4_t push_f(float32x4_t smp, float value)
{
	smp = vextq_f32(smp, vdupq_n_f32(0), 1);
	smp = vsetq_lane_f32(value, smp, 3);
	return smp;
}

void oil_shift_left_f_neon(float *f)
{
	float32x4_t v = vld1q_f32(f);
	v = vextq_f32(v, vdupq_n_f32(0), 1);
	vst1q_f32(f, v);
}

void oil_yscale_out_nonlinear_neon(float *sums, int len, unsigned char *out)
{
	int i;
	float32x4_t vals, f0, f1, f2, f3;
	float32x4_t scale, half, zero, one;
	int32x4_t idx;

	scale = vdupq_n_f32(255.0f);
	half = vdupq_n_f32(0.5f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);

	for (i=0; i+3<len; i+=4) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		f3 = vld1q_f32(sums + 12);

		vals = gather_lane0(f0, f1, f2, f3);

		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale), half));

		out[i]   = (unsigned char)vgetq_lane_s32(idx, 0);
		out[i+1] = (unsigned char)vgetq_lane_s32(idx, 1);
		out[i+2] = (unsigned char)vgetq_lane_s32(idx, 2);
		out[i+3] = (unsigned char)vgetq_lane_s32(idx, 3);

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(f3));

		sums += 16;
	}

	for (; i<len; i++) {
		float v = *sums;
		if (v > 1.0f) v = 1.0f;
		else if (v < 0.0f) v = 0.0f;
		out[i] = (int)(v * 255.0f + 0.5f);
		oil_shift_left_f_neon(sums);
		sums += 4;
	}
}

void oil_yscale_out_linear_neon(float *sums, int len, unsigned char *out)
{
	int i;
	float32x4_t scale_v, vals, f0, f1, f2, f3;
	int32x4_t idx;
	unsigned char *lut;

	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));

	for (i=0; i+3<len; i+=4) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		f3 = vld1q_f32(sums + 12);

		vals = gather_lane0(f0, f1, f2, f3);

		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = lut[vgetq_lane_s32(idx, 3)];

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(f3));

		sums += 16;
	}

	for (; i<len; i++) {
		out[i] = lut[(int)(*sums * (l2s_len - 1))];
		oil_shift_left_f_neon(sums);
		sums += 4;
	}
}

void oil_yscale_out_ga_neon(float *sums, int width, unsigned char *out)
{
	int i;
	float32x4_t f0, f1, f2, f3;
	float32x4_t vals, alpha_spread, safe_alpha, divided, gray_clamped, result;
	float32x4_t scale_v, half, zero, one;
	uint32x4_t blend_mask, nz_mask;
	int32x4_t idx;
	float gray, alpha;

	scale_v = vdupq_n_f32(255.0f);
	half = vdupq_n_f32(0.5f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);
	/* mask: 0 for gray positions (0,2), all-ones for alpha positions (1,3) */
	{
		uint32_t mask_vals[4] = {0, 0xFFFFFFFF, 0, 0xFFFFFFFF};
		blend_mask = vld1q_u32(mask_vals);
	}

	for (i=0; i+1<width; i+=2) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		f3 = vld1q_f32(sums + 12);

		/* vals = [gray0, alpha0, gray1, alpha1] */
		vals = gather_lane0(f0, f1, f2, f3);

		/* spread alpha: [alpha0, alpha0, alpha1, alpha1] */
		{
			float a0 = vgetq_lane_f32(vals, 1);
			float a1 = vgetq_lane_f32(vals, 3);
			float tmp[4] = {a0, a0, a1, a1};
			alpha_spread = vld1q_f32(tmp);
		}
		alpha_spread = vminq_f32(vmaxq_f32(alpha_spread, zero), one);
		nz_mask = vmvnq_u32(vceqq_f32(alpha_spread, zero));
		safe_alpha = vbslq_f32(nz_mask, alpha_spread, one);
		divided = vdivq_f32(vals, safe_alpha);
		gray_clamped = vminq_f32(vmaxq_f32(divided, zero), one);
		result = vbslq_f32(blend_mask, alpha_spread, gray_clamped);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(result, scale_v), half));

		/* Pack 4 ints -> 4 bytes */
		{
			int16x4_t n16 = vqmovn_s32(idx);
			uint8x8_t n8 = vqmovun_s16(vcombine_s16(n16, n16));
			vst1_lane_u32((uint32_t *)out, vreinterpret_u32_u8(n8), 0);
		}

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(f3));

		sums += 16;
		out += 4;
	}

	for (; i<width; i++) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);

		alpha = vgetq_lane_f32(f1, 0);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;

		gray = vgetq_lane_f32(f0, 0);
		if (alpha != 0) {
			gray /= alpha;
		}
		if (gray > 1.0f) gray = 1.0f;
		else if (gray < 0.0f) gray = 0.0f;

		out[0] = (int)(gray * 255.0f + 0.5f);
		out[1] = (int)(alpha * 255.0f + 0.5f);

		vst1q_f32(sums,     shift_right_1(f0));
		vst1q_f32(sums + 4, shift_right_1(f1));

		sums += 8;
		out += 2;
	}
}

void oil_yscale_out_rgbx_neon(float *sums, int width, unsigned char *out)
{
	int i;
	float32x4_t scale_v, vals, vals2, f0, f1, f2, g0, g1, g2;
	int32x4_t idx, idx2;
	unsigned char *lut;

	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));

	for (i=0; i+1<width; i+=2) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		g0 = vld1q_f32(sums + 12);
		g1 = vld1q_f32(sums + 16);
		g2 = vld1q_f32(sums + 20);

		vals = vsetq_lane_f32(vgetq_lane_f32(f0, 0), vdupq_n_f32(0), 0);
		vals = vsetq_lane_f32(vgetq_lane_f32(f1, 0), vals, 1);
		vals = vsetq_lane_f32(vgetq_lane_f32(f2, 0), vals, 2);

		vals2 = vsetq_lane_f32(vgetq_lane_f32(g0, 0), vdupq_n_f32(0), 0);
		vals2 = vsetq_lane_f32(vgetq_lane_f32(g1, 0), vals2, 1);
		vals2 = vsetq_lane_f32(vgetq_lane_f32(g2, 0), vals2, 2);

		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));
		idx2 = vcvtq_s32_f32(vmulq_f32(vals2, scale_v));

		out[0] = lut[vgetq_lane_s32(idx, 0)];
		out[1] = lut[vgetq_lane_s32(idx, 1)];
		out[2] = lut[vgetq_lane_s32(idx, 2)];
		out[3] = 255;
		out[4] = lut[vgetq_lane_s32(idx2, 0)];
		out[5] = lut[vgetq_lane_s32(idx2, 1)];
		out[6] = lut[vgetq_lane_s32(idx2, 2)];
		out[7] = 255;

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(g0));
		vst1q_f32(sums + 16, shift_right_1(g1));
		vst1q_f32(sums + 20, shift_right_1(g2));

		sums += 24;
		out += 8;
	}

	for (; i<width; i++) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);

		vals = vsetq_lane_f32(vgetq_lane_f32(f0, 0), vdupq_n_f32(0), 0);
		vals = vsetq_lane_f32(vgetq_lane_f32(f1, 0), vals, 1);
		vals = vsetq_lane_f32(vgetq_lane_f32(f2, 0), vals, 2);

		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));

		out[0] = lut[vgetq_lane_s32(idx, 0)];
		out[1] = lut[vgetq_lane_s32(idx, 1)];
		out[2] = lut[vgetq_lane_s32(idx, 2)];
		out[3] = 255;

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));

		sums += 12;
		out += 4;
	}
}

void oil_yscale_out_rgba_neon(float *sums, int width, unsigned char *out)
{
	int i;
	float32x4_t scale_v, one, zero;
	float32x4_t f0, f1, f2, f3, vals, alpha_v;
	int32x4_t idx;
	float alpha;
	unsigned char *lut;

	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		f3 = vld1q_f32(sums + 12);

		/* Gather first element of each accumulator: {R, G, B, A} */
		vals = gather_lane0(f0, f1, f2, f3);

		/* Clamp alpha to [0, 1] */
		alpha = vgetq_lane_f32(vals, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		alpha_v = vdupq_n_f32(alpha);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			vals = vdivq_f32(vals, alpha_v);
		}

		/* Clamp RGB to [0, 1] and compute l2s_map indices */
		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));

		out[0] = lut[vgetq_lane_s32(idx, 0)];
		out[1] = lut[vgetq_lane_s32(idx, 1)];
		out[2] = lut[vgetq_lane_s32(idx, 2)];
		out[3] = (int)(alpha * 255.0f + 0.5f);

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(f3));

		sums += 16;
		out += 4;
	}
}

void oil_yscale_out_cmyk_neon(float *sums, int len, unsigned char *out)
{
	int i;
	float32x4_t scale_v, vals, vals2, f0, f1, f2, f3, g0, g1, g2, g3;
	float32x4_t zero, one, half;
	int32x4_t idx, idx2;
	int16x4_t narrowed16, narrowed16b;
	uint8x8_t narrowed8;

	scale_v = vdupq_n_f32(255.0f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);
	half = vdupq_n_f32(0.5f);

	for (i=0; i+7<len; i+=8) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		f3 = vld1q_f32(sums + 12);
		g0 = vld1q_f32(sums + 16);
		g1 = vld1q_f32(sums + 20);
		g2 = vld1q_f32(sums + 24);
		g3 = vld1q_f32(sums + 28);

		vals = gather_lane0(f0, f1, f2, f3);
		vals2 = gather_lane0(g0, g1, g2, g3);

		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		vals2 = vminq_f32(vmaxq_f32(vals2, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale_v), half));
		idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals2, scale_v), half));

		narrowed16 = vqmovn_s32(idx);
		narrowed16b = vqmovn_s32(idx2);
		narrowed8 = vqmovun_s16(vcombine_s16(narrowed16, narrowed16b));

		/* Store 8 bytes */
		vst1_u8(&out[i], narrowed8);

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(f3));
		vst1q_f32(sums + 16, shift_right_1(g0));
		vst1q_f32(sums + 20, shift_right_1(g1));
		vst1q_f32(sums + 24, shift_right_1(g2));
		vst1q_f32(sums + 28, shift_right_1(g3));

		sums += 32;
	}

	for (; i+3<len; i+=4) {
		f0 = vld1q_f32(sums);
		f1 = vld1q_f32(sums + 4);
		f2 = vld1q_f32(sums + 8);
		f3 = vld1q_f32(sums + 12);

		vals = gather_lane0(f0, f1, f2, f3);

		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale_v), half));

		narrowed16 = vqmovn_s32(idx);
		narrowed8 = vqmovun_s16(vcombine_s16(narrowed16, narrowed16));

		vst1_lane_u32((uint32_t *)&out[i],
			vreinterpret_u32_u8(narrowed8), 0);

		vst1q_f32(sums,      shift_right_1(f0));
		vst1q_f32(sums + 4,  shift_right_1(f1));
		vst1q_f32(sums + 8,  shift_right_1(f2));
		vst1q_f32(sums + 12, shift_right_1(f3));

		sums += 16;
	}

	for (; i<len; i++) {
		float v = *sums;
		if (v < 0.0f) v = 0.0f;
		if (v > 1.0f) v = 1.0f;
		out[i] = (int)(v * 255.0f + 0.5f);
		oil_shift_left_f_neon(sums);
		sums += 4;
	}
}

void oil_yscale_up_g_cmyk_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum;
	float32x4_t scale, half, zero, one;
	int32x4_t idx;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	scale = vdupq_n_f32(255.0f);
	half = vdupq_n_f32(0.5f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);

	for (i=0; i+15<len; i+=16) {
		int32x4_t idx2, idx3, idx4;
		float32x4_t sum2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum2 = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum2, scale), half));

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum = vminq_f32(vmaxq_f32(sum, zero), one);
		idx3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		sum2 = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum2 = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx4 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum2, scale), half));

		/* Pack 4x4 int32 -> 2x8 int16 -> 16 uint8 */
		{
			int16x8_t n16a = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
			int16x8_t n16b = vcombine_s16(vqmovn_s32(idx3), vqmovn_s32(idx4));
			uint8x16_t n8 = vcombine_u8(vqmovun_s16(n16a), vqmovun_s16(n16b));
			vst1q_u8(out + i, n8);
		}
	}

	for (; i+7<len; i+=8) {
		int32x4_t idx2;
		float32x4_t sum2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum2 = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum2, scale), half));

		{
			int16x8_t n16 = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
			uint8x8_t n8 = vqmovun_s16(n16);
			vst1_u8(out + i, n8);
		}
	}

	for (; i+3<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		sum = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(sum, scale), half));
		{
			int16x4_t n16 = vqmovn_s32(idx);
			uint8x8_t n8 = vqmovun_s16(vcombine_s16(n16, n16));
			vst1_lane_u32((uint32_t *)(out + i),
				vreinterpret_u32_u8(n8), 0);
		}
	}

	for (; i<len; i++) {
		float s = coeffs[0] * in[0][i] + coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] + coeffs[3] * in[3][i];
		if (s > 1.0f) s = 1.0f;
		else if (s < 0.0f) s = 0.0f;
		out[i] = (int)(s * 255.0f + 0.5f);
	}
}

void oil_yscale_up_ga_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum, sum2;
	float32x4_t scale, half, zero, one;
	float32x4_t alpha_spread, safe_alpha, divided, gray_clamped, result;
	uint32x4_t blend_mask, nz_mask;
	int32x4_t idx;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	scale = vdupq_n_f32(255.0f);
	half = vdupq_n_f32(0.5f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);
	/* mask: 0 for gray positions (0,2), all-ones for alpha positions (1,3) */
	{
		uint32_t mask_vals[4] = {0, 0xFFFFFFFF, 0, 0xFFFFFFFF};
		blend_mask = vld1q_u32(mask_vals);
	}

	/* Process 4 GA pixels (8 floats) at a time */
	for (i=0; i+7<len; i+=8) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));

		/* sum = [g0, a0, g1, a1], sum2 = [g2, a2, g3, a3] */

		/* Process first pair: spread alpha to both lanes */
		/* shuffle(sum, sum, 3,3,1,1) -> [a0, a0, a1, a1] */
		{
			float a0 = vgetq_lane_f32(sum, 1);
			float a1 = vgetq_lane_f32(sum, 3);
			float tmp[4] = {a0, a0, a1, a1};
			alpha_spread = vld1q_f32(tmp);
		}
		alpha_spread = vminq_f32(vmaxq_f32(alpha_spread, zero), one);
		nz_mask = vmvnq_u32(vceqq_f32(alpha_spread, zero));
		safe_alpha = vbslq_f32(nz_mask, alpha_spread, one);
		divided = vdivq_f32(sum, safe_alpha);
		gray_clamped = vminq_f32(vmaxq_f32(divided, zero), one);
		result = vbslq_f32(blend_mask, alpha_spread, gray_clamped);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(result, scale), half));

		/* Process second pair */
		{
			float a2 = vgetq_lane_f32(sum2, 1);
			float a3 = vgetq_lane_f32(sum2, 3);
			float tmp[4] = {a2, a2, a3, a3};
			alpha_spread = vld1q_f32(tmp);
		}
		alpha_spread = vminq_f32(vmaxq_f32(alpha_spread, zero), one);
		nz_mask = vmvnq_u32(vceqq_f32(alpha_spread, zero));
		safe_alpha = vbslq_f32(nz_mask, alpha_spread, one);
		divided = vdivq_f32(sum2, safe_alpha);
		gray_clamped = vminq_f32(vmaxq_f32(divided, zero), one);
		result = vbslq_f32(blend_mask, alpha_spread, gray_clamped);
		{
			int32x4_t idx2 = vcvtq_s32_f32(
				vaddq_f32(vmulq_f32(result, scale), half));

			/* Pack 8 ints -> 8 bytes */
			int16x8_t n16 = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
			uint8x8_t n8 = vqmovun_s16(n16);
			vst1_u8(out + i, n8);
		}
	}

	/* Process 2 GA pixels (4 floats) at a time */
	for (; i+3<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));

		{
			float a0 = vgetq_lane_f32(sum, 1);
			float a1 = vgetq_lane_f32(sum, 3);
			float tmp[4] = {a0, a0, a1, a1};
			alpha_spread = vld1q_f32(tmp);
		}
		alpha_spread = vminq_f32(vmaxq_f32(alpha_spread, zero), one);
		nz_mask = vmvnq_u32(vceqq_f32(alpha_spread, zero));
		safe_alpha = vbslq_f32(nz_mask, alpha_spread, one);
		divided = vdivq_f32(sum, safe_alpha);
		gray_clamped = vminq_f32(vmaxq_f32(divided, zero), one);
		result = vbslq_f32(blend_mask, alpha_spread, gray_clamped);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(result, scale), half));
		{
			int16x4_t n16 = vqmovn_s32(idx);
			uint8x8_t n8 = vqmovun_s16(vcombine_s16(n16, n16));
			vst1_lane_u32((uint32_t *)(out + i),
				vreinterpret_u32_u8(n8), 0);
		}
	}

	/* Scalar tail for remaining pixel */
	for (; i<len; i+=2) {
		float gray, alpha_f;
		gray = coeffs[0] * in[0][i] + coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] + coeffs[3] * in[3][i];
		alpha_f = coeffs[0] * in[0][i+1] + coeffs[1] * in[1][i+1] +
			coeffs[2] * in[2][i+1] + coeffs[3] * in[3][i+1];
		if (alpha_f > 1.0f) alpha_f = 1.0f;
		else if (alpha_f < 0.0f) alpha_f = 0.0f;
		if (alpha_f != 0) gray /= alpha_f;
		if (gray > 1.0f) gray = 1.0f;
		else if (gray < 0.0f) gray = 0.0f;
		out[i] = (int)(gray * 255.0f + 0.5f);
		out[i+1] = (int)(alpha_f * 255.0f + 0.5f);
	}
}

void oil_yscale_up_rgb_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum;
	float32x4_t scale_v;
	int32x4_t idx;
	unsigned char *lut;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));

	for (i=0; i+7<len; i+=8) {
		int32x4_t idx2;
		float32x4_t sum2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		idx2 = vcvtq_s32_f32(vmulq_f32(sum2, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = lut[vgetq_lane_s32(idx, 3)];
		out[i+4] = lut[vgetq_lane_s32(idx2, 0)];
		out[i+5] = lut[vgetq_lane_s32(idx2, 1)];
		out[i+6] = lut[vgetq_lane_s32(idx2, 2)];
		out[i+7] = lut[vgetq_lane_s32(idx2, 3)];
	}

	for (; i+3<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));
		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = lut[vgetq_lane_s32(idx, 3)];
	}

	for (; i<len; i++) {
		out[i] = lut[(int)(
			(coeffs[0] * in[0][i] + coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] + coeffs[3] * in[3][i]) * (l2s_len - 1))];
	}
}

void oil_yscale_up_rgbx_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum;
	float32x4_t scale_v;
	int32x4_t idx;
	unsigned char *lut;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));

	for (i=0; i+7<len; i+=8) {
		/* Pixel 0: 4 floats [R, G, B, X] */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = 255;

		/* Pixel 1: next 4 floats */
		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		out[i+4] = lut[vgetq_lane_s32(idx, 0)];
		out[i+5] = lut[vgetq_lane_s32(idx, 1)];
		out[i+6] = lut[vgetq_lane_s32(idx, 2)];
		out[i+7] = 255;
	}

	for (; i+3<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vaddq_f32(
			vaddq_f32(vmulq_f32(c0, v0), vmulq_f32(c1, v1)),
			vaddq_f32(vmulq_f32(c2, v2), vmulq_f32(c3, v3)));
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = 255;
	}
}

void oil_yscale_up_rgba_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum;
	float32x4_t scale_v, one, zero;
	float32x4_t alpha_v, clamped;
	int32x4_t idx;
	unsigned char *lut;
	float alpha;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);

	for (i=0; i<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		/* Clamp alpha to [0, 1] */
		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		alpha_v = vdupq_n_f32(alpha);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			sum = vdivq_f32(sum, alpha_v);
		}

		/* Clamp to [0, 1] and compute l2s_map indices */
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = (int)(alpha * 255.0f + 0.5f);
	}
}

void oil_xscale_up_g_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp, coeffs, prod;

	smp = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		/* push_f: shift right by 1 lane, insert new value at position 3 */
		smp = push_f(smp, i2f_map[in[i]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);
			float32x4_t t2 = dot2(smp, c0, c1);
			out[0] = vgetq_lane_f32(t2, 0);
			out[1] = vgetq_lane_f32(t2, 1);
			out += 2;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			coeffs = vld1q_f32(coeff_buf);
			prod = vmulq_f32(smp, coeffs);
			out[0] = hsum_f32(prod);
			out += 1;
			coeff_buf += 4;
		}
	}
}

void oil_xscale_up_ga_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp_g, smp_a;

	smp_g = vdupq_n_f32(0.0f);
	smp_a = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float alpha_new = in[1] / 255.0f;
		smp_a = push_f(smp_a, alpha_new);
		smp_g = push_f(smp_g, alpha_new * i2f_map[in[0]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);

			/* gray dot products for 2 outputs */
			float32x4_t t2_g = dot2(smp_g, c0, c1);

			/* alpha dot products for 2 outputs */
			float32x4_t t2_a = dot2(smp_a, c0, c1);

			/* interleave: [gray0, alpha0, gray1, alpha1] */
			vst1q_f32(out, vzip1q_f32(t2_g, t2_a));
			out += 4;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t coeffs = vld1q_f32(coeff_buf);

			out[0] = hsum_f32(vmulq_f32(smp_g, coeffs));
			out[1] = hsum_f32(vmulq_f32(smp_a, coeffs));

			out += 2;
			coeff_buf += 4;
		}

		in += 2;
	}
}

void oil_xscale_up_rgb_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp_r, smp_g, smp_b;

	smp_r = vdupq_n_f32(0.0f);
	smp_g = vdupq_n_f32(0.0f);
	smp_b = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		smp_r = push_f(smp_r, s2l_map[in[0]]);
		smp_g = push_f(smp_g, s2l_map[in[1]]);
		smp_b = push_f(smp_b, s2l_map[in[2]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);

			float32x4_t t2_r = dot2(smp_r, c0, c1);
			float32x4_t t2_g = dot2(smp_g, c0, c1);
			float32x4_t t2_b = dot2(smp_b, c0, c1);

			/* Store interleaved: [R0, G0, B0, R1, G1, B1] */
			out[0] = vgetq_lane_f32(t2_r, 0);
			out[1] = vgetq_lane_f32(t2_g, 0);
			out[2] = vgetq_lane_f32(t2_b, 0);
			out[3] = vgetq_lane_f32(t2_r, 1);
			out[4] = vgetq_lane_f32(t2_g, 1);
			out[5] = vgetq_lane_f32(t2_b, 1);

			out += 6;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t coeffs = vld1q_f32(coeff_buf);

			out[0] = hsum_f32(vmulq_f32(smp_r, coeffs));
			out[1] = hsum_f32(vmulq_f32(smp_g, coeffs));
			out[2] = hsum_f32(vmulq_f32(smp_b, coeffs));

			out += 3;
			coeff_buf += 4;
		}

		in += 3;
	}
}

void oil_xscale_up_rgbx_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp_r, smp_g, smp_b, smp_x;

	smp_r = vdupq_n_f32(0.0f);
	smp_g = vdupq_n_f32(0.0f);
	smp_b = vdupq_n_f32(0.0f);
	smp_x = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		smp_r = push_f(smp_r, s2l_map[in[0]]);
		smp_g = push_f(smp_g, s2l_map[in[1]]);
		smp_b = push_f(smp_b, s2l_map[in[2]]);
		smp_x = push_f(smp_x, 1.0f);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);

			float32x4_t t2_r = dot2(smp_r, c0, c1);
			float32x4_t t2_g = dot2(smp_g, c0, c1);
			float32x4_t t2_b = dot2(smp_b, c0, c1);
			float32x4_t t2_x = dot2(smp_x, c0, c1);

			/* Store interleaved: [R0, G0, B0, X0, R1, G1, B1, X1] */
			out[0] = vgetq_lane_f32(t2_r, 0);
			out[1] = vgetq_lane_f32(t2_g, 0);
			out[2] = vgetq_lane_f32(t2_b, 0);
			out[3] = vgetq_lane_f32(t2_x, 0);
			out[4] = vgetq_lane_f32(t2_r, 1);
			out[5] = vgetq_lane_f32(t2_g, 1);
			out[6] = vgetq_lane_f32(t2_b, 1);
			out[7] = vgetq_lane_f32(t2_x, 1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t coeffs = vld1q_f32(coeff_buf);

			out[0] = hsum_f32(vmulq_f32(smp_r, coeffs));
			out[1] = hsum_f32(vmulq_f32(smp_g, coeffs));
			out[2] = hsum_f32(vmulq_f32(smp_b, coeffs));
			out[3] = hsum_f32(vmulq_f32(smp_x, coeffs));

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

void oil_xscale_up_rgba_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp_r, smp_g, smp_b, smp_a;
	float *sl;

	sl = s2l_map;
	smp_r = vdupq_n_f32(0.0f);
	smp_g = vdupq_n_f32(0.0f);
	smp_b = vdupq_n_f32(0.0f);
	smp_a = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float alpha_new = in[3] / 255.0f;

		smp_a = push_f(smp_a, alpha_new);
		smp_r = push_f(smp_r, alpha_new * sl[in[0]]);
		smp_g = push_f(smp_g, alpha_new * sl[in[1]]);
		smp_b = push_f(smp_b, alpha_new * sl[in[2]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);

			float32x4_t t2_r = dot2(smp_r, c0, c1);
			float32x4_t t2_g = dot2(smp_g, c0, c1);
			float32x4_t t2_b = dot2(smp_b, c0, c1);
			float32x4_t t2_a = dot2(smp_a, c0, c1);

			/* Store interleaved: [R0, G0, B0, A0, R1, G1, B1, A1] */
			out[0] = vgetq_lane_f32(t2_r, 0);
			out[1] = vgetq_lane_f32(t2_g, 0);
			out[2] = vgetq_lane_f32(t2_b, 0);
			out[3] = vgetq_lane_f32(t2_a, 0);
			out[4] = vgetq_lane_f32(t2_r, 1);
			out[5] = vgetq_lane_f32(t2_g, 1);
			out[6] = vgetq_lane_f32(t2_b, 1);
			out[7] = vgetq_lane_f32(t2_a, 1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t coeffs = vld1q_f32(coeff_buf);

			out[0] = hsum_f32(vmulq_f32(smp_r, coeffs));
			out[1] = hsum_f32(vmulq_f32(smp_g, coeffs));
			out[2] = hsum_f32(vmulq_f32(smp_b, coeffs));
			out[3] = hsum_f32(vmulq_f32(smp_a, coeffs));

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

void oil_xscale_up_cmyk_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3, inv255;

	/* Interleaved layout: each smpN = [C, M, Y, K] for one tap position */
	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);
	inv255 = vdupq_n_f32(1.0f / 255.0f);

	for (i=0; i<width_in; i++) {
		/* Push new pixel: load 4 bytes [C,M,Y,K], convert to floats */
		float32x4_t px_f;
		{
			uint8x8_t px8 = vreinterpret_u8_u32(vld1_dup_u32(
				(const uint32_t *)in));
			uint16x8_t px16 = vmovl_u8(px8);
			uint32x4_t px32 = vmovl_u16(vget_low_u16(px16));
			px_f = vcvtq_f32_u32(px32);
		}
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;
		smp3 = vmulq_f32(px_f, inv255);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t coeffs0 = vld1q_f32(coeff_buf);
			float32x4_t coeffs1 = vld1q_f32(coeff_buf + 4);

			/* First output: broadcast each coeff and multiply */
			float32x4_t result0 = vaddq_f32(
				vaddq_f32(
					vmulq_f32(smp0, vdupq_laneq_f32(coeffs0, 0)),
					vmulq_f32(smp1, vdupq_laneq_f32(coeffs0, 1))),
				vaddq_f32(
					vmulq_f32(smp2, vdupq_laneq_f32(coeffs0, 2)),
					vmulq_f32(smp3, vdupq_laneq_f32(coeffs0, 3))));

			/* Second output */
			float32x4_t result1 = vaddq_f32(
				vaddq_f32(
					vmulq_f32(smp0, vdupq_laneq_f32(coeffs1, 0)),
					vmulq_f32(smp1, vdupq_laneq_f32(coeffs1, 1))),
				vaddq_f32(
					vmulq_f32(smp2, vdupq_laneq_f32(coeffs1, 2)),
					vmulq_f32(smp3, vdupq_laneq_f32(coeffs1, 3))));

			vst1q_f32(out, result0);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t coeffs = vld1q_f32(coeff_buf);

			float32x4_t result = vaddq_f32(
				vaddq_f32(
					vmulq_f32(smp0, vdupq_laneq_f32(coeffs, 0)),
					vmulq_f32(smp1, vdupq_laneq_f32(coeffs, 1))),
				vaddq_f32(
					vmulq_f32(smp2, vdupq_laneq_f32(coeffs, 2)),
					vmulq_f32(smp3, vdupq_laneq_f32(coeffs, 3))));

			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

void oil_scale_down_g_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float32x4_t coeffs_x, coeffs_x2, coeffs_x3, coeffs_x4;
	float32x4_t sample_x, sum, sum2, sum3, sum4;
	float32x4_t coeffs_y, sums_y, sample_y;

	coeffs_y = vld1q_f32(coeffs_y_f);
	sum = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 8) {
			sum2 = vdupq_n_f32(0.0f);
			sum3 = vdupq_n_f32(0.0f);
			sum4 = vdupq_n_f32(0.0f);

			for (j=0; j+3<border_buf[i]; j+=4) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);
				coeffs_x3 = vld1q_f32(coeffs_x_f + 8);
				coeffs_x4 = vld1q_f32(coeffs_x_f + 12);

				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum);

				sample_x = vdupq_n_f32(i2f_map[in[1]]);
				sum2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum2);

				sample_x = vdupq_n_f32(i2f_map[in[2]]);
				sum3 = vaddq_f32(vmulq_f32(coeffs_x3, sample_x), sum3);

				sample_x = vdupq_n_f32(i2f_map[in[3]]);
				sum4 = vaddq_f32(vmulq_f32(coeffs_x4, sample_x), sum4);

				in += 4;
				coeffs_x_f += 16;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum);
				in += 1;
				coeffs_x_f += 4;
			}

			sum = vaddq_f32(vaddq_f32(sum, sum2),
				vaddq_f32(sum3, sum4));
		} else if (border_buf[i] >= 4) {
			sum2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum);

				sample_x = vdupq_n_f32(i2f_map[in[1]]);
				sum2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum2);

				in += 2;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum);
				in += 1;
				coeffs_x_f += 4;
			}

			sum = vaddq_f32(sum, sum2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum);
				in += 1;
				coeffs_x_f += 4;
			}
		}

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sum = vextq_f32(sum, vdupq_n_f32(0), 1);
	}
}

void oil_scale_down_ga_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float alpha;
	float32x4_t coeffs_x, coeffs_x2, coeffs_x3, coeffs_x4;
	float32x4_t sample_x, sum_g, sum_a;
	float32x4_t sum_g2, sum_a2, sum_g3, sum_a3, sum_g4, sum_a4;
	float32x4_t coeffs_y, sums_y, sample_y;

	coeffs_y = vld1q_f32(coeffs_y_f);

	sum_g = vdupq_n_f32(0.0f);
	sum_a = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 8) {
			sum_g2 = vdupq_n_f32(0.0f);
			sum_a2 = vdupq_n_f32(0.0f);
			sum_g3 = vdupq_n_f32(0.0f);
			sum_a3 = vdupq_n_f32(0.0f);
			sum_g4 = vdupq_n_f32(0.0f);
			sum_a4 = vdupq_n_f32(0.0f);

			for (j=0; j+3<border_buf[i]; j+=4) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);
				coeffs_x3 = vld1q_f32(coeffs_x_f + 8);
				coeffs_x4 = vld1q_f32(coeffs_x_f + 12);

				alpha = i2f_map[in[1]];
				sample_x = vdupq_n_f32(i2f_map[in[0]] * alpha);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);
				sample_x = vdupq_n_f32(alpha);
				sum_a = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_a);

				alpha = i2f_map[in[3]];
				sample_x = vdupq_n_f32(i2f_map[in[2]] * alpha);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);
				sample_x = vdupq_n_f32(alpha);
				sum_a2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_a2);

				alpha = i2f_map[in[5]];
				sample_x = vdupq_n_f32(i2f_map[in[4]] * alpha);
				sum_g3 = vaddq_f32(vmulq_f32(coeffs_x3, sample_x), sum_g3);
				sample_x = vdupq_n_f32(alpha);
				sum_a3 = vaddq_f32(vmulq_f32(coeffs_x3, sample_x), sum_a3);

				alpha = i2f_map[in[7]];
				sample_x = vdupq_n_f32(i2f_map[in[6]] * alpha);
				sum_g4 = vaddq_f32(vmulq_f32(coeffs_x4, sample_x), sum_g4);
				sample_x = vdupq_n_f32(alpha);
				sum_a4 = vaddq_f32(vmulq_f32(coeffs_x4, sample_x), sum_a4);

				in += 8;
				coeffs_x_f += 16;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				alpha = i2f_map[in[1]];
				sample_x = vdupq_n_f32(i2f_map[in[0]] * alpha);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);
				sample_x = vdupq_n_f32(alpha);
				sum_a = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_a);
				in += 2;
				coeffs_x_f += 4;
			}

			sum_g = vaddq_f32(vaddq_f32(sum_g, sum_g2),
				vaddq_f32(sum_g3, sum_g4));
			sum_a = vaddq_f32(vaddq_f32(sum_a, sum_a2),
				vaddq_f32(sum_a3, sum_a4));
		} else if (border_buf[i] >= 4) {
			sum_g2 = vdupq_n_f32(0.0f);
			sum_a2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				alpha = i2f_map[in[1]];
				sample_x = vdupq_n_f32(i2f_map[in[0]] * alpha);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);
				sample_x = vdupq_n_f32(alpha);
				sum_a = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_a);

				alpha = i2f_map[in[3]];
				sample_x = vdupq_n_f32(i2f_map[in[2]] * alpha);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);
				sample_x = vdupq_n_f32(alpha);
				sum_a2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_a2);

				in += 4;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				alpha = i2f_map[in[1]];
				sample_x = vdupq_n_f32(i2f_map[in[0]] * alpha);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);
				sample_x = vdupq_n_f32(alpha);
				sum_a = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_a);
				in += 2;
				coeffs_x_f += 4;
			}

			sum_g = vaddq_f32(sum_g, sum_g2);
			sum_a = vaddq_f32(sum_a, sum_a2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				alpha = i2f_map[in[1]];
				sample_x = vdupq_n_f32(i2f_map[in[0]] * alpha);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);
				sample_x = vdupq_n_f32(alpha);
				sum_a = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_a);
				in += 2;
				coeffs_x_f += 4;
			}
		}

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_g, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_a, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_a = vextq_f32(sum_a, vdupq_n_f32(0), 1);
	}
}

void oil_scale_down_rgb_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float32x4_t coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	float32x4_t sum_r2, sum_g2, sum_b2;
	float32x4_t coeffs_y, sums_y, sample_y;

	coeffs_y = vld1q_f32(coeffs_y_f);

	sum_r = vdupq_n_f32(0.0f);
	sum_g = vdupq_n_f32(0.0f);
	sum_b = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_r2 = vdupq_n_f32(0.0f);
			sum_g2 = vdupq_n_f32(0.0f);
			sum_b2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				sample_x = vdupq_n_f32(s2l_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sample_x = vdupq_n_f32(s2l_map[in[3]]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(s2l_map[in[4]]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(s2l_map[in[5]]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				in += 6;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(s2l_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				in += 3;
				coeffs_x_f += 4;
			}

			sum_r = vaddq_f32(sum_r, sum_r2);
			sum_g = vaddq_f32(sum_g, sum_g2);
			sum_b = vaddq_f32(sum_b, sum_b2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(s2l_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				in += 3;
				coeffs_x_f += 4;
			}
		}

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_r, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_g, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_b, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
	}
}

void oil_scale_down_rgba_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float32x4_t coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	float32x4_t sum_r, sum_g, sum_b, sum_a;
	float32x4_t sum_r2, sum_g2, sum_b2, sum_a2;
	float32x4_t coeffs_y, sums_y, sample_y;
	float *sl;

	sl = s2l_map;
	coeffs_y = vld1q_f32(coeffs_y_f);

	sum_r = vdupq_n_f32(0.0f);
	sum_g = vdupq_n_f32(0.0f);
	sum_b = vdupq_n_f32(0.0f);
	sum_a = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_r2 = vdupq_n_f32(0.0f);
			sum_g2 = vdupq_n_f32(0.0f);
			sum_b2 = vdupq_n_f32(0.0f);
			sum_a2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(in[3] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(sl[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(sl[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				coeffs_x2_a = vmulq_f32(coeffs_x2,
					vdupq_n_f32(in[7] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[4]]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_r2);

				sample_x = vdupq_n_f32(sl[in[5]]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_g2);

				sample_x = vdupq_n_f32(sl[in[6]]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = vaddq_f32(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(in[3] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(sl[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(sl[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_r = vaddq_f32(sum_r, sum_r2);
			sum_g = vaddq_f32(sum_g, sum_g2);
			sum_b = vaddq_f32(sum_b, sum_b2);
			sum_a = vaddq_f32(sum_a, sum_a2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(in[3] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(sl[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(sl[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_r, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_g, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_b, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_a, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
		sum_a = vextq_f32(sum_a, vdupq_n_f32(0), 1);
	}
}

void oil_scale_down_rgbx_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float32x4_t coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	float32x4_t sum_r2, sum_g2, sum_b2;
	float32x4_t coeffs_y, sums_y, sample_y;

	coeffs_y = vld1q_f32(coeffs_y_f);

	sum_r = vdupq_n_f32(0.0f);
	sum_g = vdupq_n_f32(0.0f);
	sum_b = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 2) {
			sum_r2 = vdupq_n_f32(0.0f);
			sum_g2 = vdupq_n_f32(0.0f);
			sum_b2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				sample_x = vdupq_n_f32(s2l_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sample_x = vdupq_n_f32(s2l_map[in[4]]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(s2l_map[in[5]]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(s2l_map[in[6]]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(s2l_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_r = vaddq_f32(sum_r, sum_r2);
			sum_g = vaddq_f32(sum_g, sum_g2);
			sum_b = vaddq_f32(sum_b, sum_b2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(s2l_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_r, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_g, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = vld1q_f32(sums_y_out);
		sample_y = vdupq_n_f32(vgetq_lane_f32(sum_b, 0));
		sums_y = vaddq_f32(vmulq_f32(coeffs_y, sample_y), sums_y);
		vst1q_f32(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
	}
}

void oil_scale_down_cmyk_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float32x4_t coeffs_x, coeffs_x2, sum_c, sum_m, sum_y, sum_k;
	float32x4_t sum_c2, sum_m2, sum_y2, sum_k2;
	float32x4_t coeffs_y, sums_yv;
	float32x4_t pix1, pix2;
	float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);

	coeffs_y = vld1q_f32(coeffs_y_f);

	sum_c = vdupq_n_f32(0.0f);
	sum_m = vdupq_n_f32(0.0f);
	sum_y = vdupq_n_f32(0.0f);
	sum_k = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 2) {
			sum_c2 = vdupq_n_f32(0.0f);
			sum_m2 = vdupq_n_f32(0.0f);
			sum_y2 = vdupq_n_f32(0.0f);
			sum_k2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				uint8x8_t bytes;
				uint16x8_t u16;

				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				bytes = vld1_u8(in);
				u16 = vmovl_u8(bytes);
				pix1 = vmulq_f32(vcvtq_f32_u32(
					vmovl_u16(vget_low_u16(u16))), inv255);
				pix2 = vmulq_f32(vcvtq_f32_u32(
					vmovl_u16(vget_high_u16(u16))), inv255);

				sum_c = vfmaq_laneq_f32(sum_c, coeffs_x, pix1, 0);
				sum_m = vfmaq_laneq_f32(sum_m, coeffs_x, pix1, 1);
				sum_y = vfmaq_laneq_f32(sum_y, coeffs_x, pix1, 2);
				sum_k = vfmaq_laneq_f32(sum_k, coeffs_x, pix1, 3);

				sum_c2 = vfmaq_laneq_f32(sum_c2, coeffs_x2, pix2, 0);
				sum_m2 = vfmaq_laneq_f32(sum_m2, coeffs_x2, pix2, 1);
				sum_y2 = vfmaq_laneq_f32(sum_y2, coeffs_x2, pix2, 2);
				sum_k2 = vfmaq_laneq_f32(sum_k2, coeffs_x2, pix2, 3);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				pix1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(
					vget_low_u16(vmovl_u8(
					vreinterpret_u8_u32(vld1_dup_u32(
					(const uint32_t *)in)))))), inv255);

				sum_c = vfmaq_laneq_f32(sum_c, coeffs_x, pix1, 0);
				sum_m = vfmaq_laneq_f32(sum_m, coeffs_x, pix1, 1);
				sum_y = vfmaq_laneq_f32(sum_y, coeffs_x, pix1, 2);
				sum_k = vfmaq_laneq_f32(sum_k, coeffs_x, pix1, 3);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_c = vaddq_f32(sum_c, sum_c2);
			sum_m = vaddq_f32(sum_m, sum_m2);
			sum_y = vaddq_f32(sum_y, sum_y2);
			sum_k = vaddq_f32(sum_k, sum_k2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				pix1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(
					vget_low_u16(vmovl_u8(
					vreinterpret_u8_u32(vld1_dup_u32(
					(const uint32_t *)in)))))), inv255);

				sum_c = vfmaq_laneq_f32(sum_c, coeffs_x, pix1, 0);
				sum_m = vfmaq_laneq_f32(sum_m, coeffs_x, pix1, 1);
				sum_y = vfmaq_laneq_f32(sum_y, coeffs_x, pix1, 2);
				sum_k = vfmaq_laneq_f32(sum_k, coeffs_x, pix1, 3);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		sums_yv = vld1q_f32(sums_y_out);
		sums_yv = vfmaq_laneq_f32(sums_yv, coeffs_y, sum_c, 0);
		vst1q_f32(sums_y_out, sums_yv);
		sums_y_out += 4;

		sums_yv = vld1q_f32(sums_y_out);
		sums_yv = vfmaq_laneq_f32(sums_yv, coeffs_y, sum_m, 0);
		vst1q_f32(sums_y_out, sums_yv);
		sums_y_out += 4;

		sums_yv = vld1q_f32(sums_y_out);
		sums_yv = vfmaq_laneq_f32(sums_yv, coeffs_y, sum_y, 0);
		vst1q_f32(sums_y_out, sums_yv);
		sums_y_out += 4;

		sums_yv = vld1q_f32(sums_y_out);
		sums_yv = vfmaq_laneq_f32(sums_yv, coeffs_y, sum_k, 0);
		vst1q_f32(sums_y_out, sums_yv);
		sums_y_out += 4;

		sum_c = vextq_f32(sum_c, vdupq_n_f32(0), 1);
		sum_m = vextq_f32(sum_m, vdupq_n_f32(0), 1);
		sum_y = vextq_f32(sum_y, vdupq_n_f32(0), 1);
		sum_k = vextq_f32(sum_k, vdupq_n_f32(0), 1);
	}
}

#endif /* OIL_USE_NEON */
