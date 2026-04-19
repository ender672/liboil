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

#include "oil_resample.h"
#include "oil_resample_internal.h"

#include <string.h>
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
	float32x4_t s1 = vpaddq_f32(p0, p1);
	float32x4_t s2 = vpaddq_f32(s1, s1);
	return s2;
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

static void oil_shift_left_f_neon(float *f)
{
	float32x4_t v = vld1q_f32(f);
	v = vextq_f32(v, vdupq_n_f32(0), 1);
	vst1q_f32(f, v);
}

static void oil_yscale_out_nonlinear_neon(float *sums, int len, unsigned char *out)
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

static void oil_yscale_out_linear_neon(float *sums, int len, unsigned char *out)
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

static void oil_yscale_out_ga_neon(float *sums, int width, unsigned char *out)
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

static void oil_yscale_out_rgbx_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, vals;
	int32x4_t idx;
	float32x4_t z;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		vals = vld1q_f32(sums + tap_off);

		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));

		out[0] = lut[vgetq_lane_s32(idx, 0)];
		out[1] = lut[vgetq_lane_s32(idx, 1)];
		out[2] = lut[vgetq_lane_s32(idx, 2)];
		out[3] = 255;

		/* Zero consumed tap */
		vst1q_f32(sums + tap_off, z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_out_rgba_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, one, zero;
	float32x4_t vals, alpha_v;
	int32x4_t idx;
	float32x4_t z;
	float alpha;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		/* Read [R, G, B, A] from current tap slot */
		vals = vld1q_f32(sums + tap_off);

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

		/* Zero consumed tap */
		vst1q_f32(sums + tap_off, z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_out_argb_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, one, zero;
	float32x4_t vals, alpha_v;
	int32x4_t idx;
	float alpha;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		/* Read [R, G, B, A] from current tap slot */
		vals = vld1q_f32(sums + tap_off);

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

		out[0] = (int)(alpha * 255.0f + 0.5f);
		out[1] = lut[vgetq_lane_s32(idx, 0)];
		out[2] = lut[vgetq_lane_s32(idx, 1)];
		out[3] = lut[vgetq_lane_s32(idx, 2)];

		/* Zero consumed tap */
		vst1q_f32(sums + tap_off, vdupq_n_f32(0.0f));

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_out_cmyk_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, vals, zero, one, half, z;
	int32x4_t idx;
	int16x4_t narrowed16;
	uint8x8_t narrowed8;

	tap_off = tap * 4;
	scale_v = vdupq_n_f32(255.0f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);
	half = vdupq_n_f32(0.5f);
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		vals = vld1q_f32(sums + tap_off);
		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale_v), half));

		narrowed16 = vqmovn_s32(idx);
		narrowed8 = vqmovun_s16(vcombine_s16(narrowed16, narrowed16));

		vst1_lane_u32((uint32_t *)out,
			vreinterpret_u32_u8(narrowed8), 0);

		vst1q_f32(sums + tap_off, z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_up_g_cmyk_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum;
	float32x4_t scale;
	int32x4_t idx;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	scale = vdupq_n_f32(255.0f);

	for (i=0; i+31<len; i+=32) {
		int32x4_t idx2, idx3, idx4, idx5, idx6, idx7, idx8;
		float32x4_t sum2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx2 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx3 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx4 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		v0 = vld1q_f32(in[0] + i + 16);
		v1 = vld1q_f32(in[1] + i + 16);
		v2 = vld1q_f32(in[2] + i + 16);
		v3 = vld1q_f32(in[3] + i + 16);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx5 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 20);
		v1 = vld1q_f32(in[1] + i + 20);
		v2 = vld1q_f32(in[2] + i + 20);
		v3 = vld1q_f32(in[3] + i + 20);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx6 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		v0 = vld1q_f32(in[0] + i + 24);
		v1 = vld1q_f32(in[1] + i + 24);
		v2 = vld1q_f32(in[2] + i + 24);
		v3 = vld1q_f32(in[3] + i + 24);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx7 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 28);
		v1 = vld1q_f32(in[1] + i + 28);
		v2 = vld1q_f32(in[2] + i + 28);
		v3 = vld1q_f32(in[3] + i + 28);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx8 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		/* Pack 8x4 int32 -> 4x8 int16 -> 2x16 uint8 */
		{
			int16x8_t n16a = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
			int16x8_t n16b = vcombine_s16(vqmovn_s32(idx3), vqmovn_s32(idx4));
			uint8x16_t n8a = vcombine_u8(vqmovun_s16(n16a), vqmovun_s16(n16b));
			vst1q_u8(out + i, n8a);

			int16x8_t n16c = vcombine_s16(vqmovn_s32(idx5), vqmovn_s32(idx6));
			int16x8_t n16d = vcombine_s16(vqmovn_s32(idx7), vqmovn_s32(idx8));
			uint8x16_t n8b = vcombine_u8(vqmovun_s16(n16c), vqmovun_s16(n16d));
			vst1q_u8(out + i + 16, n8b);
		}
	}

	for (; i+15<len; i+=16) {
		int32x4_t idx2, idx3, idx4;
		float32x4_t sum2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx2 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx3 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx4 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

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
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx2 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

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
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));
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

static void oil_yscale_up_ga_neon(float **in, int len, float *coeffs,
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
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);

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
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

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

static void oil_yscale_up_rgb_neon(float **in, int len, float *coeffs,
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

	for (i=0; i+15<len; i+=16) {
		int32x4_t idx2, idx3, idx4;
		float32x4_t sum2, sum3, sum4;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx2 = vcvtq_s32_f32(vmulq_f32(sum2, scale_v));

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		sum3 = vmulq_f32(c0, v0);
		sum3 = vfmaq_f32(sum3, c1, v1);
		sum3 = vfmaq_f32(sum3, c2, v2);
		sum3 = vfmaq_f32(sum3, c3, v3);
		idx3 = vcvtq_s32_f32(vmulq_f32(sum3, scale_v));

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		sum4 = vmulq_f32(c0, v0);
		sum4 = vfmaq_f32(sum4, c1, v1);
		sum4 = vfmaq_f32(sum4, c2, v2);
		sum4 = vfmaq_f32(sum4, c3, v3);
		idx4 = vcvtq_s32_f32(vmulq_f32(sum4, scale_v));

		{
			uint8x16_t bytes;
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 0)], vdupq_n_u8(0), 0);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 1)], bytes, 1);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 2)], bytes, 2);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 3)], bytes, 3);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 0)], bytes, 4);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 1)], bytes, 5);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 2)], bytes, 6);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 3)], bytes, 7);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 0)], bytes, 8);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 1)], bytes, 9);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 2)], bytes, 10);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 3)], bytes, 11);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 0)], bytes, 12);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 1)], bytes, 13);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 2)], bytes, 14);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 3)], bytes, 15);
			vst1q_u8(out + i, bytes);
		}
	}

	for (; i+7<len; i+=8) {
		int32x4_t idx2;
		float32x4_t sum2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
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
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
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

static void oil_yscale_up_rgbx_neon(float **in, int len, float *coeffs,
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

	for (i=0; i+15<len; i+=16) {
		float32x4_t sum2, sum3, sum4;
		int32x4_t idx2, idx3, idx4;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx2 = vcvtq_s32_f32(vmulq_f32(sum2, scale_v));

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		sum3 = vmulq_f32(c0, v0);
		sum3 = vfmaq_f32(sum3, c1, v1);
		sum3 = vfmaq_f32(sum3, c2, v2);
		sum3 = vfmaq_f32(sum3, c3, v3);
		idx3 = vcvtq_s32_f32(vmulq_f32(sum3, scale_v));

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		sum4 = vmulq_f32(c0, v0);
		sum4 = vfmaq_f32(sum4, c1, v1);
		sum4 = vfmaq_f32(sum4, c2, v2);
		sum4 = vfmaq_f32(sum4, c3, v3);
		idx4 = vcvtq_s32_f32(vmulq_f32(sum4, scale_v));

		{
			uint8x16_t bytes;
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 0)], vdupq_n_u8(255), 0);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 1)], bytes, 1);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx, 2)], bytes, 2);
			/* lane 3 = 255 (from vdupq_n_u8) */
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 0)], bytes, 4);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 1)], bytes, 5);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx2, 2)], bytes, 6);
			/* lane 7 = 255 */
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 0)], bytes, 8);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 1)], bytes, 9);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx3, 2)], bytes, 10);
			/* lane 11 = 255 */
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 0)], bytes, 12);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 1)], bytes, 13);
			bytes = vsetq_lane_u8(lut[vgetq_lane_s32(idx4, 2)], bytes, 14);
			/* lane 15 = 255 */
			vst1q_u8(out + i, bytes);
		}
	}

	for (; i+7<len; i+=8) {
		float32x4_t sum2;
		int32x4_t idx2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);
		idx2 = vcvtq_s32_f32(vmulq_f32(sum2, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = 255;
		out[i+4] = lut[vgetq_lane_s32(idx2, 0)];
		out[i+5] = lut[vgetq_lane_s32(idx2, 1)];
		out[i+6] = lut[vgetq_lane_s32(idx2, 2)];
		out[i+7] = 255;
	}

	for (; i+3<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = 255;
	}
}

static void oil_yscale_up_rgba_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum, sum2;
	float32x4_t scale_v, one, zero;
	float32x4_t alpha_v, clamped;
	int32x4_t idx;
	unsigned char *lut;
	float alpha, alpha2;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);

	for (i=0; i+15<len; i+=16) {
		float32x4_t s0, s1, s2, s3;
		float a0, a1, a2, a3;

		/* Compute weighted sums for 4 pixels */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		s0 = vmulq_f32(c0, v0);
		s0 = vfmaq_f32(s0, c1, v1);
		s0 = vfmaq_f32(s0, c2, v2);
		s0 = vfmaq_f32(s0, c3, v3);

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		s1 = vmulq_f32(c0, v0);
		s1 = vfmaq_f32(s1, c1, v1);
		s1 = vfmaq_f32(s1, c2, v2);
		s1 = vfmaq_f32(s1, c3, v3);

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		s2 = vmulq_f32(c0, v0);
		s2 = vfmaq_f32(s2, c1, v1);
		s2 = vfmaq_f32(s2, c2, v2);
		s2 = vfmaq_f32(s2, c3, v3);

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		s3 = vmulq_f32(c0, v0);
		s3 = vfmaq_f32(s3, c1, v1);
		s3 = vfmaq_f32(s3, c2, v2);
		s3 = vfmaq_f32(s3, c3, v3);

		/* Alpha handling for pixel 0 */
		a0 = vgetq_lane_f32(s0, 3);
		if (a0 > 1.0f) a0 = 1.0f;
		else if (a0 < 0.0f) a0 = 0.0f;
		if (a0 != 0) {
			alpha_v = vdupq_n_f32(a0);
			s0 = vdivq_f32(s0, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s0, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = (int)(a0 * 255.0f + 0.5f);

		/* Alpha handling for pixel 1 */
		a1 = vgetq_lane_f32(s1, 3);
		if (a1 > 1.0f) a1 = 1.0f;
		else if (a1 < 0.0f) a1 = 0.0f;
		if (a1 != 0) {
			alpha_v = vdupq_n_f32(a1);
			s1 = vdivq_f32(s1, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s1, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+4] = lut[vgetq_lane_s32(idx, 0)];
		out[i+5] = lut[vgetq_lane_s32(idx, 1)];
		out[i+6] = lut[vgetq_lane_s32(idx, 2)];
		out[i+7] = (int)(a1 * 255.0f + 0.5f);

		/* Alpha handling for pixel 2 */
		a2 = vgetq_lane_f32(s2, 3);
		if (a2 > 1.0f) a2 = 1.0f;
		else if (a2 < 0.0f) a2 = 0.0f;
		if (a2 != 0) {
			alpha_v = vdupq_n_f32(a2);
			s2 = vdivq_f32(s2, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s2, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+8]  = lut[vgetq_lane_s32(idx, 0)];
		out[i+9]  = lut[vgetq_lane_s32(idx, 1)];
		out[i+10] = lut[vgetq_lane_s32(idx, 2)];
		out[i+11] = (int)(a2 * 255.0f + 0.5f);

		/* Alpha handling for pixel 3 */
		a3 = vgetq_lane_f32(s3, 3);
		if (a3 > 1.0f) a3 = 1.0f;
		else if (a3 < 0.0f) a3 = 0.0f;
		if (a3 != 0) {
			alpha_v = vdupq_n_f32(a3);
			s3 = vdivq_f32(s3, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s3, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+12] = lut[vgetq_lane_s32(idx, 0)];
		out[i+13] = lut[vgetq_lane_s32(idx, 1)];
		out[i+14] = lut[vgetq_lane_s32(idx, 2)];
		out[i+15] = (int)(a3 * 255.0f + 0.5f);
	}

	for (; i+7<len; i+=8) {
		/* Pixel 0 */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		/* Pixel 1 */
		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);

		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		if (alpha != 0) {
			alpha_v = vdupq_n_f32(alpha);
			sum = vdivq_f32(sum, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = (int)(alpha * 255.0f + 0.5f);

		alpha2 = vgetq_lane_f32(sum2, 3);
		if (alpha2 > 1.0f) alpha2 = 1.0f;
		else if (alpha2 < 0.0f) alpha2 = 0.0f;
		if (alpha2 != 0) {
			alpha_v = vdupq_n_f32(alpha2);
			sum2 = vdivq_f32(sum2, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+4] = lut[vgetq_lane_s32(idx, 0)];
		out[i+5] = lut[vgetq_lane_s32(idx, 1)];
		out[i+6] = lut[vgetq_lane_s32(idx, 2)];
		out[i+7] = (int)(alpha2 * 255.0f + 0.5f);
	}

	for (; i<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		if (alpha != 0) {
			alpha_v = vdupq_n_f32(alpha);
			sum = vdivq_f32(sum, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = (int)(alpha * 255.0f + 0.5f);
	}
}

static void oil_yscale_up_argb_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum, sum2;
	float32x4_t scale_v, one, zero;
	float32x4_t alpha_v, clamped;
	int32x4_t idx;
	unsigned char *lut;
	float alpha, alpha2;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);

	for (i=0; i+15<len; i+=16) {
		float32x4_t s0, s1, s2, s3;
		float a0, a1, a2, a3;

		/* Compute weighted sums for 4 pixels */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		s0 = vmulq_f32(c0, v0);
		s0 = vfmaq_f32(s0, c1, v1);
		s0 = vfmaq_f32(s0, c2, v2);
		s0 = vfmaq_f32(s0, c3, v3);

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		s1 = vmulq_f32(c0, v0);
		s1 = vfmaq_f32(s1, c1, v1);
		s1 = vfmaq_f32(s1, c2, v2);
		s1 = vfmaq_f32(s1, c3, v3);

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		s2 = vmulq_f32(c0, v0);
		s2 = vfmaq_f32(s2, c1, v1);
		s2 = vfmaq_f32(s2, c2, v2);
		s2 = vfmaq_f32(s2, c3, v3);

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		s3 = vmulq_f32(c0, v0);
		s3 = vfmaq_f32(s3, c1, v1);
		s3 = vfmaq_f32(s3, c2, v2);
		s3 = vfmaq_f32(s3, c3, v3);

		/* Alpha handling for pixel 0 */
		a0 = vgetq_lane_f32(s0, 3);
		if (a0 > 1.0f) a0 = 1.0f;
		else if (a0 < 0.0f) a0 = 0.0f;
		if (a0 != 0) {
			alpha_v = vdupq_n_f32(a0);
			s0 = vdivq_f32(s0, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s0, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i]   = (int)(a0 * 255.0f + 0.5f);
		out[i+1] = lut[vgetq_lane_s32(idx, 0)];
		out[i+2] = lut[vgetq_lane_s32(idx, 1)];
		out[i+3] = lut[vgetq_lane_s32(idx, 2)];

		/* Alpha handling for pixel 1 */
		a1 = vgetq_lane_f32(s1, 3);
		if (a1 > 1.0f) a1 = 1.0f;
		else if (a1 < 0.0f) a1 = 0.0f;
		if (a1 != 0) {
			alpha_v = vdupq_n_f32(a1);
			s1 = vdivq_f32(s1, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s1, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+4] = (int)(a1 * 255.0f + 0.5f);
		out[i+5] = lut[vgetq_lane_s32(idx, 0)];
		out[i+6] = lut[vgetq_lane_s32(idx, 1)];
		out[i+7] = lut[vgetq_lane_s32(idx, 2)];

		/* Alpha handling for pixel 2 */
		a2 = vgetq_lane_f32(s2, 3);
		if (a2 > 1.0f) a2 = 1.0f;
		else if (a2 < 0.0f) a2 = 0.0f;
		if (a2 != 0) {
			alpha_v = vdupq_n_f32(a2);
			s2 = vdivq_f32(s2, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s2, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+8]  = (int)(a2 * 255.0f + 0.5f);
		out[i+9]  = lut[vgetq_lane_s32(idx, 0)];
		out[i+10] = lut[vgetq_lane_s32(idx, 1)];
		out[i+11] = lut[vgetq_lane_s32(idx, 2)];

		/* Alpha handling for pixel 3 */
		a3 = vgetq_lane_f32(s3, 3);
		if (a3 > 1.0f) a3 = 1.0f;
		else if (a3 < 0.0f) a3 = 0.0f;
		if (a3 != 0) {
			alpha_v = vdupq_n_f32(a3);
			s3 = vdivq_f32(s3, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s3, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+12] = (int)(a3 * 255.0f + 0.5f);
		out[i+13] = lut[vgetq_lane_s32(idx, 0)];
		out[i+14] = lut[vgetq_lane_s32(idx, 1)];
		out[i+15] = lut[vgetq_lane_s32(idx, 2)];
	}

	for (; i+7<len; i+=8) {
		/* Pixel 0 */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		/* Pixel 1 */
		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);

		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		if (alpha != 0) {
			alpha_v = vdupq_n_f32(alpha);
			sum = vdivq_f32(sum, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i]   = (int)(alpha * 255.0f + 0.5f);
		out[i+1] = lut[vgetq_lane_s32(idx, 0)];
		out[i+2] = lut[vgetq_lane_s32(idx, 1)];
		out[i+3] = lut[vgetq_lane_s32(idx, 2)];

		alpha2 = vgetq_lane_f32(sum2, 3);
		if (alpha2 > 1.0f) alpha2 = 1.0f;
		else if (alpha2 < 0.0f) alpha2 = 0.0f;
		if (alpha2 != 0) {
			alpha_v = vdupq_n_f32(alpha2);
			sum2 = vdivq_f32(sum2, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i+4] = (int)(alpha2 * 255.0f + 0.5f);
		out[i+5] = lut[vgetq_lane_s32(idx, 0)];
		out[i+6] = lut[vgetq_lane_s32(idx, 1)];
		out[i+7] = lut[vgetq_lane_s32(idx, 2)];
	}

	for (; i<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		if (alpha != 0) {
			alpha_v = vdupq_n_f32(alpha);
			sum = vdivq_f32(sum, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vmulq_f32(clamped, scale_v));
		out[i]   = (int)(alpha * 255.0f + 0.5f);
		out[i+1] = lut[vgetq_lane_s32(idx, 0)];
		out[i+2] = lut[vgetq_lane_s32(idx, 1)];
		out[i+3] = lut[vgetq_lane_s32(idx, 2)];
	}
}

static void oil_xscale_up_g_neon(unsigned char *in, int width_in, float *out,
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

static void oil_xscale_up_ga_neon(unsigned char *in, int width_in, float *out,
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

static void oil_xscale_up_rgb_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;
	float *sl;

	sl = s2l_map;
	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: [R_linear, G_linear, B_linear, 0] */
		pixel = vsetq_lane_f32(sl[in[0]], vdupq_n_f32(0), 0);
		pixel = vsetq_lane_f32(sl[in[1]], pixel, 1);
		pixel = vsetq_lane_f32(sl[in[2]], pixel, 2);
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);

			/* Store interleaved: [R0, G0, B0, R1, G1, B1] */
			out[0] = vgetq_lane_f32(result0, 0);
			out[1] = vgetq_lane_f32(result0, 1);
			out[2] = vgetq_lane_f32(result0, 2);
			out[3] = vgetq_lane_f32(result1, 0);
			out[4] = vgetq_lane_f32(result1, 1);
			out[5] = vgetq_lane_f32(result1, 2);

			out += 6;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);

			out[0] = vgetq_lane_f32(result, 0);
			out[1] = vgetq_lane_f32(result, 1);
			out[2] = vgetq_lane_f32(result, 2);

			out += 3;
			coeff_buf += 4;
		}

		in += 3;
	}
}

static void oil_xscale_up_rgbx_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;
	float *sl;

	sl = s2l_map;
	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: load 4 bytes as uint32, extract via bitshift */
		{
			unsigned int px;
			memcpy(&px, in, 4);
			pixel = vsetq_lane_f32(sl[px & 0xFF], vdupq_n_f32(1.0f), 0);
			pixel = vsetq_lane_f32(sl[(px >> 8) & 0xFF], pixel, 1);
			pixel = vsetq_lane_f32(sl[(px >> 16) & 0xFF], pixel, 2);
		}
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);
			vst1q_f32(out, result0);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);
			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_xscale_up_rgba_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;
	float *sl;

	sl = s2l_map;
	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float alpha_new = in[3] / 255.0f;
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: [alpha*R_linear, alpha*G_linear, alpha*B_linear, alpha] */
		pixel = vsetq_lane_f32(alpha_new * sl[in[0]], vdupq_n_f32(0), 0);
		pixel = vsetq_lane_f32(alpha_new * sl[in[1]], pixel, 1);
		pixel = vsetq_lane_f32(alpha_new * sl[in[2]], pixel, 2);
		pixel = vsetq_lane_f32(alpha_new, pixel, 3);
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			/* dot product: c[0]*smp0 + c[1]*smp1 + c[2]*smp2 + c[3]*smp3 */
			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);
			vst1q_f32(out, result0);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);
			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_xscale_up_argb_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;
	float *sl;

	sl = s2l_map;
	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float alpha_new = in[0] / 255.0f;
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: [alpha*R_linear, alpha*G_linear, alpha*B_linear, alpha] */
		pixel = vsetq_lane_f32(alpha_new * sl[in[1]], vdupq_n_f32(0), 0);
		pixel = vsetq_lane_f32(alpha_new * sl[in[2]], pixel, 1);
		pixel = vsetq_lane_f32(alpha_new * sl[in[3]], pixel, 2);
		pixel = vsetq_lane_f32(alpha_new, pixel, 3);
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			/* dot product: c[0]*smp0 + c[1]*smp1 + c[2]*smp2 + c[3]*smp3 */
			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);
			vst1q_f32(out, result0);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);
			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_xscale_up_cmyk_neon(unsigned char *in, int width_in, float *out,
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
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);
			vst1q_f32(out, result0);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);
			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_scale_down_g_neon(unsigned char *in, float *sums_y_out,
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

static void oil_scale_down_ga_neon(unsigned char *in, float *sums_y_out,
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

static void oil_scale_down_rgb_neon(unsigned char *in, float *sums_y_out,
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

static void oil_scale_down_rgba_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	float32x4_t coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	float32x4_t sum_r, sum_g, sum_b, sum_a;
	float32x4_t sum_r2, sum_g2, sum_b2, sum_a2;
	float32x4_t cy0, cy1, cy2, cy3;
	float *sl;

	sl = s2l_map;
	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = vdupq_n_f32(coeffs_y_f[0]);
	cy1 = vdupq_n_f32(coeffs_y_f[1]);
	cy2 = vdupq_n_f32(coeffs_y_f[2]);
	cy3 = vdupq_n_f32(coeffs_y_f[3]);

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

		/* Vertical accumulation using ring buffer offsets */
		{
			float32x4_t rgba, sy;

			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_r, 0), vdupq_n_f32(0), 0);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_g, 0), rgba, 1);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_b, 0), rgba, 2);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_a, 0), rgba, 3);

			sy = vld1q_f32(sums_y_out + off0);
			sy = vfmaq_f32(sy, cy0, rgba);
			vst1q_f32(sums_y_out + off0, sy);

			sy = vld1q_f32(sums_y_out + off1);
			sy = vfmaq_f32(sy, cy1, rgba);
			vst1q_f32(sums_y_out + off1, sy);

			sy = vld1q_f32(sums_y_out + off2);
			sy = vfmaq_f32(sy, cy2, rgba);
			vst1q_f32(sums_y_out + off2, sy);

			sy = vld1q_f32(sums_y_out + off3);
			sy = vfmaq_f32(sy, cy3, rgba);
			vst1q_f32(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
		sum_a = vextq_f32(sum_a, vdupq_n_f32(0), 1);
	}
}

static void oil_scale_down_argb_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	float32x4_t coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	float32x4_t sum_r, sum_g, sum_b, sum_a;
	float32x4_t sum_r2, sum_g2, sum_b2, sum_a2;
	float32x4_t cy0, cy1, cy2, cy3;
	float *sl;

	sl = s2l_map;
	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = vdupq_n_f32(coeffs_y_f[0]);
	cy1 = vdupq_n_f32(coeffs_y_f[1]);
	cy2 = vdupq_n_f32(coeffs_y_f[2]);
	cy3 = vdupq_n_f32(coeffs_y_f[3]);

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
					vdupq_n_f32(in[0] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[1]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(sl[in[2]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(sl[in[3]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				coeffs_x2_a = vmulq_f32(coeffs_x2,
					vdupq_n_f32(in[4] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[5]]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_r2);

				sample_x = vdupq_n_f32(sl[in[6]]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_g2);

				sample_x = vdupq_n_f32(sl[in[7]]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = vaddq_f32(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(in[0] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[1]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(sl[in[2]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(sl[in[3]]);
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
					vdupq_n_f32(in[0] * (1.0f / 255.0f)));

				sample_x = vdupq_n_f32(sl[in[1]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(sl[in[2]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(sl[in[3]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			float32x4_t rgba, sy;

			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_r, 0), vdupq_n_f32(0), 0);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_g, 0), rgba, 1);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_b, 0), rgba, 2);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_a, 0), rgba, 3);

			sy = vld1q_f32(sums_y_out + off0);
			sy = vfmaq_f32(sy, cy0, rgba);
			vst1q_f32(sums_y_out + off0, sy);

			sy = vld1q_f32(sums_y_out + off1);
			sy = vfmaq_f32(sy, cy1, rgba);
			vst1q_f32(sums_y_out + off1, sy);

			sy = vld1q_f32(sums_y_out + off2);
			sy = vfmaq_f32(sy, cy2, rgba);
			vst1q_f32(sums_y_out + off2, sy);

			sy = vld1q_f32(sums_y_out + off3);
			sy = vfmaq_f32(sy, cy3, rgba);
			vst1q_f32(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
		sum_a = vextq_f32(sum_a, vdupq_n_f32(0), 1);
	}
}

static void oil_scale_down_rgbx_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	float32x4_t coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	float32x4_t sum_r2, sum_g2, sum_b2;
	float32x4_t cy0, cy1, cy2, cy3;

	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = vdupq_n_f32(coeffs_y_f[0]);
	cy1 = vdupq_n_f32(coeffs_y_f[1]);
	cy2 = vdupq_n_f32(coeffs_y_f[2]);
	cy3 = vdupq_n_f32(coeffs_y_f[3]);

	sum_r = vdupq_n_f32(0.0f);
	sum_g = vdupq_n_f32(0.0f);
	sum_b = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 2) {
			sum_r2 = vdupq_n_f32(0.0f);
			sum_g2 = vdupq_n_f32(0.0f);
			sum_b2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				unsigned int px0, px1;
				memcpy(&px0, in, 4);
				memcpy(&px1, in + 4, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				sample_x = vdupq_n_f32(s2l_map[px0 & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[(px0 >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[(px0 >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sample_x = vdupq_n_f32(s2l_map[px1 & 0xFF]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(s2l_map[(px1 >> 8) & 0xFF]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(s2l_map[(px1 >> 16) & 0xFF]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(s2l_map[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[(px >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_r = vaddq_f32(sum_r, sum_r2);
			sum_g = vaddq_f32(sum_g, sum_g2);
			sum_b = vaddq_f32(sum_b, sum_b2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(s2l_map[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(s2l_map[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(s2l_map[(px >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			float32x4_t rgbx, sy;

			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_r, 0), vdupq_n_f32(0), 0);
			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_g, 0), rgbx, 1);
			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_b, 0), rgbx, 2);

			sy = vld1q_f32(sums_y_out + off0);
			sy = vfmaq_f32(sy, cy0, rgbx);
			vst1q_f32(sums_y_out + off0, sy);

			sy = vld1q_f32(sums_y_out + off1);
			sy = vfmaq_f32(sy, cy1, rgbx);
			vst1q_f32(sums_y_out + off1, sy);

			sy = vld1q_f32(sums_y_out + off2);
			sy = vfmaq_f32(sy, cy2, rgbx);
			vst1q_f32(sums_y_out + off2, sy);

			sy = vld1q_f32(sums_y_out + off3);
			sy = vfmaq_f32(sy, cy3, rgbx);
			vst1q_f32(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
	}
}

static void oil_scale_down_cmyk_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	float32x4_t coeffs_x, coeffs_x2, sum_c, sum_m, sum_y, sum_k;
	float32x4_t sum_c2, sum_m2, sum_y2, sum_k2;
	float32x4_t cy0, cy1, cy2, cy3;
	float32x4_t pix1, pix2;
	float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);

	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = vdupq_n_f32(coeffs_y_f[0]);
	cy1 = vdupq_n_f32(coeffs_y_f[1]);
	cy2 = vdupq_n_f32(coeffs_y_f[2]);
	cy3 = vdupq_n_f32(coeffs_y_f[3]);

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

		{
			float32x4_t cmyk, sy;

			cmyk = vsetq_lane_f32(vgetq_lane_f32(sum_c, 0), vdupq_n_f32(0), 0);
			cmyk = vsetq_lane_f32(vgetq_lane_f32(sum_m, 0), cmyk, 1);
			cmyk = vsetq_lane_f32(vgetq_lane_f32(sum_y, 0), cmyk, 2);
			cmyk = vsetq_lane_f32(vgetq_lane_f32(sum_k, 0), cmyk, 3);

			sy = vld1q_f32(sums_y_out + off0);
			sy = vfmaq_f32(sy, cy0, cmyk);
			vst1q_f32(sums_y_out + off0, sy);

			sy = vld1q_f32(sums_y_out + off1);
			sy = vfmaq_f32(sy, cy1, cmyk);
			vst1q_f32(sums_y_out + off1, sy);

			sy = vld1q_f32(sums_y_out + off2);
			sy = vfmaq_f32(sy, cy2, cmyk);
			vst1q_f32(sums_y_out + off2, sy);

			sy = vld1q_f32(sums_y_out + off3);
			sy = vfmaq_f32(sy, cy3, cmyk);
			vst1q_f32(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_c = vextq_f32(sum_c, vdupq_n_f32(0), 1);
		sum_m = vextq_f32(sum_m, vdupq_n_f32(0), 1);
		sum_y = vextq_f32(sum_y, vdupq_n_f32(0), 1);
		sum_k = vextq_f32(sum_k, vdupq_n_f32(0), 1);
	}
}

static void oil_xscale_up_rgb_nogamma_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;

	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: [R, G, B, 0] using i2f_map (no gamma) */
		pixel = vsetq_lane_f32(i2f_map[in[0]], vdupq_n_f32(0), 0);
		pixel = vsetq_lane_f32(i2f_map[in[1]], pixel, 1);
		pixel = vsetq_lane_f32(i2f_map[in[2]], pixel, 2);
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);

			/* Store interleaved: [R0, G0, B0, R1, G1, B1] */
			out[0] = vgetq_lane_f32(result0, 0);
			out[1] = vgetq_lane_f32(result0, 1);
			out[2] = vgetq_lane_f32(result0, 2);
			out[3] = vgetq_lane_f32(result1, 0);
			out[4] = vgetq_lane_f32(result1, 1);
			out[5] = vgetq_lane_f32(result1, 2);

			out += 6;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);

			out[0] = vgetq_lane_f32(result, 0);
			out[1] = vgetq_lane_f32(result, 1);
			out[2] = vgetq_lane_f32(result, 2);

			out += 3;
			coeff_buf += 4;
		}

		in += 3;
	}
}

static void oil_scale_down_rgb_nogamma_neon(unsigned char *in, float *sums_y_out,
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

				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sample_x = vdupq_n_f32(i2f_map[in[3]]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(i2f_map[in[4]]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(i2f_map[in[5]]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				in += 6;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[in[2]]);
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

				sample_x = vdupq_n_f32(i2f_map[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[in[2]]);
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

static void oil_yscale_out_rgba_nogamma_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, one, zero, half;
	float32x4_t vals, alpha_v;
	int32x4_t idx;
	float32x4_t z;
	float alpha;

	tap_off = tap * 4;
	scale_v = vdupq_n_f32(255.0f);
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	half = vdupq_n_f32(0.5f);
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		/* Read [R, G, B, A] from current tap slot */
		vals = vld1q_f32(sums + tap_off);

		/* Clamp alpha to [0, 1] */
		alpha = vgetq_lane_f32(vals, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		alpha_v = vdupq_n_f32(alpha);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			vals = vdivq_f32(vals, alpha_v);
		}

		/* Clamp RGB to [0, 1], scale to [0, 255], round */
		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale_v), half));

		out[0] = vgetq_lane_s32(idx, 0);
		out[1] = vgetq_lane_s32(idx, 1);
		out[2] = vgetq_lane_s32(idx, 2);
		out[3] = (int)(alpha * 255.0f + 0.5f);

		/* Zero consumed tap */
		vst1q_f32(sums + tap_off, z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_up_rgba_nogamma_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum, sum2;
	float32x4_t scale_v, one, zero, half;
	float32x4_t alpha_v, clamped;
	int32x4_t idx;
	float alpha, alpha2;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	scale_v = vdupq_n_f32(255.0f);
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	half = vdupq_n_f32(0.5f);

	for (i=0; i+15<len; i+=16) {
		float32x4_t s0, s1, s2, s3;
		float a0, a1, a2, a3;

		/* Compute weighted sums for 4 pixels */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		s0 = vmulq_f32(c0, v0);
		s0 = vfmaq_f32(s0, c1, v1);
		s0 = vfmaq_f32(s0, c2, v2);
		s0 = vfmaq_f32(s0, c3, v3);

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		s1 = vmulq_f32(c0, v0);
		s1 = vfmaq_f32(s1, c1, v1);
		s1 = vfmaq_f32(s1, c2, v2);
		s1 = vfmaq_f32(s1, c3, v3);

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		s2 = vmulq_f32(c0, v0);
		s2 = vfmaq_f32(s2, c1, v1);
		s2 = vfmaq_f32(s2, c2, v2);
		s2 = vfmaq_f32(s2, c3, v3);

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		s3 = vmulq_f32(c0, v0);
		s3 = vfmaq_f32(s3, c1, v1);
		s3 = vfmaq_f32(s3, c2, v2);
		s3 = vfmaq_f32(s3, c3, v3);

		/* Alpha handling for pixel 0 */
		a0 = vgetq_lane_f32(s0, 3);
		if (a0 > 1.0f) a0 = 1.0f;
		else if (a0 < 0.0f) a0 = 0.0f;
		if (a0 != 0) {
			alpha_v = vdupq_n_f32(a0);
			s0 = vdivq_f32(s0, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s0, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i]   = vgetq_lane_s32(idx, 0);
		out[i+1] = vgetq_lane_s32(idx, 1);
		out[i+2] = vgetq_lane_s32(idx, 2);
		out[i+3] = (int)(a0 * 255.0f + 0.5f);

		/* Alpha handling for pixel 1 */
		a1 = vgetq_lane_f32(s1, 3);
		if (a1 > 1.0f) a1 = 1.0f;
		else if (a1 < 0.0f) a1 = 0.0f;
		if (a1 != 0) {
			alpha_v = vdupq_n_f32(a1);
			s1 = vdivq_f32(s1, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s1, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i+4] = vgetq_lane_s32(idx, 0);
		out[i+5] = vgetq_lane_s32(idx, 1);
		out[i+6] = vgetq_lane_s32(idx, 2);
		out[i+7] = (int)(a1 * 255.0f + 0.5f);

		/* Alpha handling for pixel 2 */
		a2 = vgetq_lane_f32(s2, 3);
		if (a2 > 1.0f) a2 = 1.0f;
		else if (a2 < 0.0f) a2 = 0.0f;
		if (a2 != 0) {
			alpha_v = vdupq_n_f32(a2);
			s2 = vdivq_f32(s2, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s2, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i+8]  = vgetq_lane_s32(idx, 0);
		out[i+9]  = vgetq_lane_s32(idx, 1);
		out[i+10] = vgetq_lane_s32(idx, 2);
		out[i+11] = (int)(a2 * 255.0f + 0.5f);

		/* Alpha handling for pixel 3 */
		a3 = vgetq_lane_f32(s3, 3);
		if (a3 > 1.0f) a3 = 1.0f;
		else if (a3 < 0.0f) a3 = 0.0f;
		if (a3 != 0) {
			alpha_v = vdupq_n_f32(a3);
			s3 = vdivq_f32(s3, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(s3, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i+12] = vgetq_lane_s32(idx, 0);
		out[i+13] = vgetq_lane_s32(idx, 1);
		out[i+14] = vgetq_lane_s32(idx, 2);
		out[i+15] = (int)(a3 * 255.0f + 0.5f);
	}

	for (; i+7<len; i+=8) {
		/* Pixel 0 */
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		/* Pixel 1 */
		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);

		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		if (alpha != 0) {
			alpha_v = vdupq_n_f32(alpha);
			sum = vdivq_f32(sum, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i]   = vgetq_lane_s32(idx, 0);
		out[i+1] = vgetq_lane_s32(idx, 1);
		out[i+2] = vgetq_lane_s32(idx, 2);
		out[i+3] = (int)(alpha * 255.0f + 0.5f);

		alpha2 = vgetq_lane_f32(sum2, 3);
		if (alpha2 > 1.0f) alpha2 = 1.0f;
		else if (alpha2 < 0.0f) alpha2 = 0.0f;
		if (alpha2 != 0) {
			alpha_v = vdupq_n_f32(alpha2);
			sum2 = vdivq_f32(sum2, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i+4] = vgetq_lane_s32(idx, 0);
		out[i+5] = vgetq_lane_s32(idx, 1);
		out[i+6] = vgetq_lane_s32(idx, 2);
		out[i+7] = (int)(alpha2 * 255.0f + 0.5f);
	}

	for (; i<len; i+=4) {
		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		alpha = vgetq_lane_f32(sum, 3);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;
		if (alpha != 0) {
			alpha_v = vdupq_n_f32(alpha);
			sum = vdivq_f32(sum, alpha_v);
		}
		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));
		out[i]   = vgetq_lane_s32(idx, 0);
		out[i+1] = vgetq_lane_s32(idx, 1);
		out[i+2] = vgetq_lane_s32(idx, 2);
		out[i+3] = (int)(alpha * 255.0f + 0.5f);
	}
}

static void oil_xscale_up_rgba_nogamma_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;

	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float32x4_t pixel, pxf;
		float alpha_new;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: load 4 bytes, widen to float, premultiply by alpha */
		{
			uint8x8_t px8 = vreinterpret_u8_u32(vld1_dup_u32(
				(const uint32_t *)in));
			uint16x8_t px16 = vmovl_u8(px8);
			uint32x4_t px32 = vmovl_u16(vget_low_u16(px16));
			pxf = vmulq_f32(vcvtq_f32_u32(px32), vdupq_n_f32(1.0f / 255.0f));
		}
		alpha_new = vgetq_lane_f32(pxf, 3);
		pixel = vmulq_n_f32(pxf, alpha_new);
		pixel = vsetq_lane_f32(alpha_new, pixel, 3);
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			/* dot product: c[0]*smp0 + c[1]*smp1 + c[2]*smp2 + c[3]*smp3 */
			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);
			vst1q_f32(out, result0);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);
			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_scale_down_rgba_nogamma_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	float32x4_t coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	float32x4_t sum_r, sum_g, sum_b, sum_a;
	float32x4_t sum_r2, sum_g2, sum_b2, sum_a2;
	float32x4_t cy0, cy1, cy2, cy3;

	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = vdupq_n_f32(coeffs_y_f[0]);
	cy1 = vdupq_n_f32(coeffs_y_f[1]);
	cy2 = vdupq_n_f32(coeffs_y_f[2]);
	cy3 = vdupq_n_f32(coeffs_y_f[3]);

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
				unsigned int px0, px1;
				memcpy(&px0, in, 4);
				memcpy(&px1, in + 4, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(i2f_map[px0 >> 24]));

				sample_x = vdupq_n_f32(i2f_map[px0 & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[(px0 >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[(px0 >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				coeffs_x2_a = vmulq_f32(coeffs_x2,
					vdupq_n_f32(i2f_map[px1 >> 24]));

				sample_x = vdupq_n_f32(i2f_map[px1 & 0xFF]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_r2);

				sample_x = vdupq_n_f32(i2f_map[(px1 >> 8) & 0xFF]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_g2);

				sample_x = vdupq_n_f32(i2f_map[(px1 >> 16) & 0xFF]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = vaddq_f32(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(i2f_map[px >> 24]));

				sample_x = vdupq_n_f32(i2f_map[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[(px >> 16) & 0xFF]);
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
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				coeffs_x_a = vmulq_f32(coeffs_x,
					vdupq_n_f32(i2f_map[px >> 24]));

				sample_x = vdupq_n_f32(i2f_map[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[(px >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			float32x4_t rgba, sy;

			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_r, 0), vdupq_n_f32(0), 0);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_g, 0), rgba, 1);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_b, 0), rgba, 2);
			rgba = vsetq_lane_f32(vgetq_lane_f32(sum_a, 0), rgba, 3);

			sy = vld1q_f32(sums_y_out + off0);
			sy = vfmaq_f32(sy, cy0, rgba);
			vst1q_f32(sums_y_out + off0, sy);

			sy = vld1q_f32(sums_y_out + off1);
			sy = vfmaq_f32(sy, cy1, rgba);
			vst1q_f32(sums_y_out + off1, sy);

			sy = vld1q_f32(sums_y_out + off2);
			sy = vfmaq_f32(sy, cy2, rgba);
			vst1q_f32(sums_y_out + off2, sy);

			sy = vld1q_f32(sums_y_out + off3);
			sy = vfmaq_f32(sy, cy3, rgba);
			vst1q_f32(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
		sum_a = vextq_f32(sum_a, vdupq_n_f32(0), 1);
	}
}

static void oil_yscale_out_rgbx_nogamma_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, one, zero, half;
	float32x4_t z;
	uint8x16_t alpha_mask;

	tap_off = tap * 4;
	scale_v = vdupq_n_f32(255.0f);
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	half = vdupq_n_f32(0.5f);
	z = vdupq_n_f32(0.0f);

	{
		static const uint8_t amask[16] = {0,0,0,255, 0,0,0,255,
			0,0,0,255, 0,0,0,255};
		alpha_mask = vld1q_u8(amask);
	}

	for (i=0; i+3<width; i+=4) {
		float32x4_t v0, v1, v2, v3;
		int32x4_t i0, i1, i2, i3;
		int16x4_t h0, h1, h2, h3;
		int16x8_t h01, h23;
		uint8x8_t b01, b23;
		uint8x16_t result;

		v0 = vld1q_f32(sums + tap_off);
		v1 = vld1q_f32(sums + 16 + tap_off);
		v2 = vld1q_f32(sums + 32 + tap_off);
		v3 = vld1q_f32(sums + 48 + tap_off);

		v0 = vminq_f32(vmaxq_f32(v0, zero), one);
		v1 = vminq_f32(vmaxq_f32(v1, zero), one);
		v2 = vminq_f32(vmaxq_f32(v2, zero), one);
		v3 = vminq_f32(vmaxq_f32(v3, zero), one);

		i0 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v0, scale_v), half));
		i1 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v1, scale_v), half));
		i2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v2, scale_v), half));
		i3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v3, scale_v), half));

		h0 = vqmovn_s32(i0);
		h1 = vqmovn_s32(i1);
		h01 = vcombine_s16(h0, h1);
		h2 = vqmovn_s32(i2);
		h3 = vqmovn_s32(i3);
		h23 = vcombine_s16(h2, h3);

		b01 = vqmovun_s16(h01);
		b23 = vqmovun_s16(h23);
		result = vcombine_u8(b01, b23);

		/* Set alpha lanes to 255 */
		result = vbslq_u8(alpha_mask, vdupq_n_u8(255), result);

		vst1q_u8(out, result);

		/* Zero consumed taps */
		vst1q_f32(sums + tap_off, z);
		vst1q_f32(sums + 16 + tap_off, z);
		vst1q_f32(sums + 32 + tap_off, z);
		vst1q_f32(sums + 48 + tap_off, z);

		sums += 64;
		out += 16;
	}

	for (; i<width; i++) {
		float32x4_t vals;
		int32x4_t idx;

		vals = vld1q_f32(sums + tap_off);
		vals = vminq_f32(vmaxq_f32(vals, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scale_v), half));

		out[0] = vgetq_lane_s32(idx, 0);
		out[1] = vgetq_lane_s32(idx, 1);
		out[2] = vgetq_lane_s32(idx, 2);
		out[3] = 255;

		vst1q_f32(sums + tap_off, z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_up_rgbx_nogamma_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t v0, v1, v2, v3, sum;
	float32x4_t scale_v, one, zero, half;
	int32x4_t idx;

	c0 = vdupq_n_f32(coeffs[0]);
	c1 = vdupq_n_f32(coeffs[1]);
	c2 = vdupq_n_f32(coeffs[2]);
	c3 = vdupq_n_f32(coeffs[3]);
	scale_v = vdupq_n_f32(255.0f);
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	half = vdupq_n_f32(0.5f);

	for (i=0; i+15<len; i+=16) {
		float32x4_t sum2, sum3, sum4;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);

		v0 = vld1q_f32(in[0] + i + 8);
		v1 = vld1q_f32(in[1] + i + 8);
		v2 = vld1q_f32(in[2] + i + 8);
		v3 = vld1q_f32(in[3] + i + 8);
		sum3 = vmulq_f32(c0, v0);
		sum3 = vfmaq_f32(sum3, c1, v1);
		sum3 = vfmaq_f32(sum3, c2, v2);
		sum3 = vfmaq_f32(sum3, c3, v3);

		v0 = vld1q_f32(in[0] + i + 12);
		v1 = vld1q_f32(in[1] + i + 12);
		v2 = vld1q_f32(in[2] + i + 12);
		v3 = vld1q_f32(in[3] + i + 12);
		sum4 = vmulq_f32(c0, v0);
		sum4 = vfmaq_f32(sum4, c1, v1);
		sum4 = vfmaq_f32(sum4, c2, v2);
		sum4 = vfmaq_f32(sum4, c3, v3);

		{
			float32x4_t clamped;
			int32x4_t idx2, idx3, idx4;

			clamped = vminq_f32(vmaxq_f32(sum, zero), one);
			idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

			clamped = vminq_f32(vmaxq_f32(sum2, zero), one);
			idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

			clamped = vminq_f32(vmaxq_f32(sum3, zero), one);
			idx3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

			clamped = vminq_f32(vmaxq_f32(sum4, zero), one);
			idx4 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

			{
				uint8x16_t bytes;
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx, 0), vdupq_n_u8(255), 0);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx, 1), bytes, 1);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx, 2), bytes, 2);
				/* lane 3 = 255 */
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx2, 0), bytes, 4);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx2, 1), bytes, 5);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx2, 2), bytes, 6);
				/* lane 7 = 255 */
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx3, 0), bytes, 8);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx3, 1), bytes, 9);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx3, 2), bytes, 10);
				/* lane 11 = 255 */
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx4, 0), bytes, 12);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx4, 1), bytes, 13);
				bytes = vsetq_lane_u8(vgetq_lane_s32(idx4, 2), bytes, 14);
				/* lane 15 = 255 */
				vst1q_u8(out + i, bytes);
			}
		}
	}

	for (; i+7<len; i+=8) {
		float32x4_t sum2, clamped;
		int32x4_t idx2;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		v0 = vld1q_f32(in[0] + i + 4);
		v1 = vld1q_f32(in[1] + i + 4);
		v2 = vld1q_f32(in[2] + i + 4);
		v3 = vld1q_f32(in[3] + i + 4);
		sum2 = vmulq_f32(c0, v0);
		sum2 = vfmaq_f32(sum2, c1, v1);
		sum2 = vfmaq_f32(sum2, c2, v2);
		sum2 = vfmaq_f32(sum2, c3, v3);

		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

		clamped = vminq_f32(vmaxq_f32(sum2, zero), one);
		idx2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

		out[i]   = vgetq_lane_s32(idx, 0);
		out[i+1] = vgetq_lane_s32(idx, 1);
		out[i+2] = vgetq_lane_s32(idx, 2);
		out[i+3] = 255;
		out[i+4] = vgetq_lane_s32(idx2, 0);
		out[i+5] = vgetq_lane_s32(idx2, 1);
		out[i+6] = vgetq_lane_s32(idx2, 2);
		out[i+7] = 255;
	}

	for (; i+3<len; i+=4) {
		float32x4_t clamped;

		v0 = vld1q_f32(in[0] + i);
		v1 = vld1q_f32(in[1] + i);
		v2 = vld1q_f32(in[2] + i);
		v3 = vld1q_f32(in[3] + i);
		sum = vmulq_f32(c0, v0);
		sum = vfmaq_f32(sum, c1, v1);
		sum = vfmaq_f32(sum, c2, v2);
		sum = vfmaq_f32(sum, c3, v3);

		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

		out[i]   = vgetq_lane_s32(idx, 0);
		out[i+1] = vgetq_lane_s32(idx, 1);
		out[i+2] = vgetq_lane_s32(idx, 2);
		out[i+3] = 255;
	}
}

static void oil_xscale_up_rgbx_nogamma_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3, inv255;

	inv255 = vdupq_n_f32(1.0f / 255.0f);
	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: load 4 bytes, widen to float, scale by 1/255 */
		{
			uint8x8_t px8 = vreinterpret_u8_u32(vld1_dup_u32(
				(const uint32_t *)in));
			uint16x8_t px16 = vmovl_u8(px8);
			uint32x4_t px32 = vmovl_u16(vget_low_u16(px16));
			pixel = vmulq_f32(vcvtq_f32_u32(px32), inv255);
			pixel = vsetq_lane_f32(1.0f, pixel, 3);
		}
		smp3 = pixel;

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result0, result1;

			result0 = vmulq_laneq_f32(smp0, co, 0);
			result0 = vfmaq_laneq_f32(result0, smp1, co, 1);
			result0 = vfmaq_laneq_f32(result0, smp2, co, 2);
			result0 = vfmaq_laneq_f32(result0, smp3, co, 3);
			vst1q_f32(out, result0);

			co = vld1q_f32(coeff_buf + 4);
			result1 = vmulq_laneq_f32(smp0, co, 0);
			result1 = vfmaq_laneq_f32(result1, smp1, co, 1);
			result1 = vfmaq_laneq_f32(result1, smp2, co, 2);
			result1 = vfmaq_laneq_f32(result1, smp3, co, 3);
			vst1q_f32(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t co = vld1q_f32(coeff_buf);
			float32x4_t result;

			result = vmulq_laneq_f32(smp0, co, 0);
			result = vfmaq_laneq_f32(result, smp1, co, 1);
			result = vfmaq_laneq_f32(result, smp2, co, 2);
			result = vfmaq_laneq_f32(result, smp3, co, 3);
			vst1q_f32(out, result);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_scale_down_rgbx_nogamma_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	float32x4_t coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b, sum_x;
	float32x4_t sum_r2, sum_g2, sum_b2, sum_x2;
	float32x4_t one_v;
	float32x4_t cy0, cy1, cy2, cy3;

	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = vdupq_n_f32(coeffs_y_f[0]);
	cy1 = vdupq_n_f32(coeffs_y_f[1]);
	cy2 = vdupq_n_f32(coeffs_y_f[2]);
	cy3 = vdupq_n_f32(coeffs_y_f[3]);
	one_v = vdupq_n_f32(1.0f);

	sum_r = vdupq_n_f32(0.0f);
	sum_g = vdupq_n_f32(0.0f);
	sum_b = vdupq_n_f32(0.0f);
	sum_x = vdupq_n_f32(0.0f);

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_r2 = vdupq_n_f32(0.0f);
			sum_g2 = vdupq_n_f32(0.0f);
			sum_b2 = vdupq_n_f32(0.0f);
			sum_x2 = vdupq_n_f32(0.0f);

			for (j=0; j+1<border_buf[i]; j+=2) {
				unsigned int px0, px1;
				memcpy(&px0, in, 4);
				memcpy(&px1, in + 4, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);
				coeffs_x2 = vld1q_f32(coeffs_x_f + 4);

				sample_x = vdupq_n_f32(i2f_map[px0 & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[(px0 >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[(px0 >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sum_x = vaddq_f32(vmulq_f32(coeffs_x, one_v), sum_x);

				sample_x = vdupq_n_f32(i2f_map[px1 & 0xFF]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(i2f_map[(px1 >> 8) & 0xFF]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(i2f_map[(px1 >> 16) & 0xFF]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				sum_x2 = vaddq_f32(vmulq_f32(coeffs_x2, one_v), sum_x2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(i2f_map[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[(px >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sum_x = vaddq_f32(vmulq_f32(coeffs_x, one_v), sum_x);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_r = vaddq_f32(sum_r, sum_r2);
			sum_g = vaddq_f32(sum_g, sum_g2);
			sum_b = vaddq_f32(sum_b, sum_b2);
			sum_x = vaddq_f32(sum_x, sum_x2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(i2f_map[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(i2f_map[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(i2f_map[(px >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sum_x = vaddq_f32(vmulq_f32(coeffs_x, one_v), sum_x);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			float32x4_t rgbx, sy;

			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_r, 0), vdupq_n_f32(0), 0);
			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_g, 0), rgbx, 1);
			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_b, 0), rgbx, 2);
			rgbx = vsetq_lane_f32(vgetq_lane_f32(sum_x, 0), rgbx, 3);

			sy = vld1q_f32(sums_y_out + off0);
			sy = vfmaq_f32(sy, cy0, rgbx);
			vst1q_f32(sums_y_out + off0, sy);

			sy = vld1q_f32(sums_y_out + off1);
			sy = vfmaq_f32(sy, cy1, rgbx);
			vst1q_f32(sums_y_out + off1, sy);

			sy = vld1q_f32(sums_y_out + off2);
			sy = vfmaq_f32(sy, cy2, rgbx);
			vst1q_f32(sums_y_out + off2, sy);

			sy = vld1q_f32(sums_y_out + off3);
			sy = vfmaq_f32(sy, cy3, rgbx);
			vst1q_f32(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = vextq_f32(sum_r, vdupq_n_f32(0), 1);
		sum_g = vextq_f32(sum_g, vdupq_n_f32(0), 1);
		sum_b = vextq_f32(sum_b, vdupq_n_f32(0), 1);
		sum_x = vextq_f32(sum_x, vdupq_n_f32(0), 1);
	}
}

/* NEON dispatch functions */

static float *get_rb_line(struct oil_scale *os, int line)
{
	int sl_len;
	sl_len = OIL_CMP(os->cs) * os->out_width;
	return os->rb + line * sl_len;
}

static void yscale_out_neon(float *sums, int width, unsigned char *out,
	enum oil_colorspace cs, int tap)
{
	int sl_len;

	sl_len = width * OIL_CMP(cs);

	switch(cs) {
	case OIL_CS_G:
		oil_yscale_out_nonlinear_neon(sums, sl_len, out);
		break;
	case OIL_CS_CMYK:
		oil_yscale_out_cmyk_neon(sums, width, out, tap);
		break;
	case OIL_CS_GA:
		oil_yscale_out_ga_neon(sums, width, out);
		break;
	case OIL_CS_RGB:
		oil_yscale_out_linear_neon(sums, sl_len, out);
		break;
	case OIL_CS_RGBA:
		oil_yscale_out_rgba_neon(sums, width, out, tap);
		break;
	case OIL_CS_ARGB:
		oil_yscale_out_argb_neon(sums, width, out, tap);
		break;
	case OIL_CS_RGBX:
		oil_yscale_out_rgbx_neon(sums, width, out, tap);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_yscale_out_nonlinear_neon(sums, sl_len, out);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_yscale_out_rgba_nogamma_neon(sums, width, out, tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_yscale_out_rgbx_nogamma_neon(sums, width, out, tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void yscale_up_neon(float **in, int len, float *coeffs,
	unsigned char *out, enum oil_colorspace cs)
{
	switch(cs) {
	case OIL_CS_G:
	case OIL_CS_CMYK:
		oil_yscale_up_g_cmyk_neon(in, len, coeffs, out);
		break;
	case OIL_CS_GA:
		oil_yscale_up_ga_neon(in, len, coeffs, out);
		break;
	case OIL_CS_RGB:
		oil_yscale_up_rgb_neon(in, len, coeffs, out);
		break;
	case OIL_CS_RGBA:
		oil_yscale_up_rgba_neon(in, len, coeffs, out);
		break;
	case OIL_CS_ARGB:
		oil_yscale_up_argb_neon(in, len, coeffs, out);
		break;
	case OIL_CS_RGBX:
		oil_yscale_up_rgbx_neon(in, len, coeffs, out);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_yscale_up_g_cmyk_neon(in, len, coeffs, out);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_yscale_up_rgba_nogamma_neon(in, len, coeffs, out);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_yscale_up_rgbx_nogamma_neon(in, len, coeffs, out);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void xscale_up_neon(unsigned char *in, int width_in, float *out,
	enum oil_colorspace cs_in, float *coeff_buf, int *border_buf)
{
	switch(cs_in) {
	case OIL_CS_RGB:
		oil_xscale_up_rgb_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_G:
		oil_xscale_up_g_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_CMYK:
		oil_xscale_up_cmyk_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA:
		oil_xscale_up_rgba_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_GA:
		oil_xscale_up_ga_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_ARGB:
		oil_xscale_up_argb_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBX:
		oil_xscale_up_rgbx_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_xscale_up_rgb_nogamma_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_xscale_up_rgba_nogamma_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_xscale_up_rgbx_nogamma_neon(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void down_scale_in_neon(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	switch(os->cs) {
	case OIL_CS_RGB:
		oil_scale_down_rgb_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_G:
		oil_scale_down_g_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_CMYK:
		oil_scale_down_cmyk_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBA:
		oil_scale_down_rgba_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_GA:
		oil_scale_down_ga_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_ARGB:
		oil_scale_down_argb_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX:
		oil_scale_down_rgbx_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_scale_down_rgb_nogamma_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_scale_down_rgba_nogamma_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_scale_down_rgbx_nogamma_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}

	os->borders_y[os->out_pos] -= 1;
	os->in_pos++;
}

static void up_scale_in_neon(struct oil_scale *os, unsigned char *in)
{
	float *tmp;

	tmp = get_rb_line(os, os->in_pos % 4);
	xscale_up_neon(in, os->in_width, tmp, os->cs, os->coeffs_x, os->borders_x);

	os->in_pos++;
}

int oil_scale_in_neon(struct oil_scale *os, unsigned char *in)
{
	if (oil_scale_slots(os) == 0) {
		return -1;
	}
	if (os->out_width > os->in_width) {
		up_scale_in_neon(os, in);
	} else {
		down_scale_in_neon(os, in);
	}
	return 0;
}

int oil_scale_out_neon(struct oil_scale *os, unsigned char *out)
{
	int i, sl_len;
	float *in[4];

	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	if (os->out_height <= os->in_height) {
		yscale_out_neon(os->sums_y, os->out_width, out, os->cs, os->sums_y_tap);
		os->sums_y_tap = (os->sums_y_tap + 1) & 3;
	} else {
		sl_len = OIL_CMP(os->cs) * os->out_width;
		for (i=0; i<4; i++) {
			in[i] = get_rb_line(os, (os->in_pos + i) % 4);
		}
		yscale_up_neon(in, sl_len, os->coeffs_y + os->out_pos * 4, out,
			os->cs);
		os->borders_y[os->in_pos - 1] -= 1;
	}

	os->out_pos++;
	return 0;
}
