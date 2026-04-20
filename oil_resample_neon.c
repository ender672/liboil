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

/* Shift smp left by one float lane (lane 0 drops, lane 3 gets zero). */
static inline float32x4_t oil_shift_f_left_neon(float32x4_t v)
{
	return vextq_f32(v, vdupq_n_f32(0), 1);
}

static void oil_shift_left_f_neon(float *f)
{
	vst1q_f32(f, oil_shift_f_left_neon(vld1q_f32(f)));
}

/* Shift smp left by one lane and place v into the now-empty top lane. */
static inline float32x4_t oil_push_f_neon(float32x4_t smp, float v)
{
	smp = oil_shift_f_left_neon(smp);
	return vsetq_lane_f32(v, smp, 3);
}

/* Horizontal dot products dot(smp, c0) and dot(smp, c1) into lanes [0, 1]. */
static inline float32x4_t oil_dot2_f_neon(float32x4_t smp,
	float32x4_t c0, float32x4_t c1)
{
	float32x4_t p0 = vmulq_f32(smp, c0);
	float32x4_t p1 = vmulq_f32(smp, c1);
	float32x4_t s1 = vpaddq_f32(p0, p1);
	return vpaddq_f32(s1, s1);
}

/* Horizontal dot product dot(smp, coeffs) as a scalar float. */
static inline float oil_dot1_f_neon(float32x4_t smp, float32x4_t coeffs)
{
	return vaddvq_f32(vmulq_f32(smp, coeffs));
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

/* Consume one output pixel across 4 stride-4 channel ring-buffer slots:
 * gather lane 0 from each of sums[0..3], sums[4..7], sums[8..11], sums[12..15]
 * into a single packed vector, then shift each slot left (discarding the
 * consumed tap).
 */
static inline float32x4_t oil_consume_ch0_x4_neon(float *sums)
{
	float32x4_t f0, f1, f2, f3, vals;

	f0 = vld1q_f32(sums);
	f1 = vld1q_f32(sums + 4);
	f2 = vld1q_f32(sums + 8);
	f3 = vld1q_f32(sums + 12);

	vals = gather_lane0(f0, f1, f2, f3);

	vst1q_f32(sums,      oil_shift_f_left_neon(f0));
	vst1q_f32(sums + 4,  oil_shift_f_left_neon(f1));
	vst1q_f32(sums + 8,  oil_shift_f_left_neon(f2));
	vst1q_f32(sums + 12, oil_shift_f_left_neon(f3));

	return vals;
}

/* 4-tap y-axis dot product: loads 4 floats from each of in[0..3] at offset
 * `off` and returns c0*in[0] + c1*in[1] + c2*in[2] + c3*in[3].
 */
static inline float32x4_t oil_ydot4_load_neon(float **in, int off,
	float32x4_t c0, float32x4_t c1, float32x4_t c2, float32x4_t c3)
{
	float32x4_t v0 = vld1q_f32(in[0] + off);
	float32x4_t v1 = vld1q_f32(in[1] + off);
	float32x4_t v2 = vld1q_f32(in[2] + off);
	float32x4_t v3 = vld1q_f32(in[3] + off);
	float32x4_t sum = vmulq_f32(c0, v0);
	sum = vfmaq_f32(sum, c1, v1);
	sum = vfmaq_f32(sum, c2, v2);
	sum = vfmaq_f32(sum, c3, v3);
	return sum;
}

/* Clamp v to [0,1], multiply by `scale`, round to nearest, and truncate to
 * int32. Produces the byte-range index used by sRGB byte packing and LUTs.
 */
static inline int32x4_t oil_clamp_round_idx_neon(float32x4_t v,
	float32x4_t zero, float32x4_t one, float32x4_t scale, float32x4_t half)
{
	v = vminq_f32(vmaxq_f32(v, zero), one);
	return vcvtq_s32_f32(vfmaq_f32(half, v, scale));
}

/* Unpremultiply a premultiplied RGBA sum (alpha in lane 3), clamp RGB and
 * alpha to [0,1], then scale to 0..255 (with rounding) for byte packing.
 * Lane 3 of the result contains the clamped alpha byte, not the reciprocal.
 */
static inline int32x4_t oil_unpremul_rgba_idx_neon(float32x4_t vals,
	float32x4_t zero, float32x4_t one, float32x4_t scale, float32x4_t half)
{
	float32x4_t alpha_v;
	float alpha;

	alpha_v = vdupq_n_f32(vgetq_lane_f32(vals, 3));
	alpha_v = vminq_f32(vmaxq_f32(alpha_v, zero), one);
	alpha = vgetq_lane_f32(alpha_v, 0);
	if (alpha != 0) {
		vals = vdivq_f32(vals, alpha_v);
	}
	vals = vminq_f32(vmaxq_f32(vals, zero), one);
	vals = vsetq_lane_f32(alpha, vals, 3);
	return vcvtq_s32_f32(vfmaq_f32(half, vals, scale));
}

/* Write 3 bytes to out[0..2] by indexing lut with the low three int32 lanes
 * of idx.
 */
static inline void oil_lut_store3_neon(unsigned char *out, int32x4_t idx,
	unsigned char *lut)
{
	out[0] = lut[vgetq_lane_s32(idx, 0)];
	out[1] = lut[vgetq_lane_s32(idx, 1)];
	out[2] = lut[vgetq_lane_s32(idx, 2)];
}

/* Write 4 bytes to out[0..3] by indexing lut with all four int32 lanes. */
static inline void oil_lut_store4_neon(unsigned char *out, int32x4_t idx,
	unsigned char *lut)
{
	oil_lut_store3_neon(out, idx, lut);
	out[3] = lut[vgetq_lane_s32(idx, 3)];
}

/* Unpremultiply a premultiplied RGBA sum (alpha in lane 3) and emit one
 * output pixel: alpha as a rounded byte, RGB via the linear-to-sRGB LUT.
 * a_off/rgb_off select RGBA vs ARGB output layout.
 */
static inline void oil_unpremul_rgba_lut_neon(float32x4_t vals,
	float32x4_t zero, float32x4_t one, float32x4_t scale,
	unsigned char *lut, unsigned char *out, int a_off, int rgb_off)
{
	float32x4_t alpha_v;
	int32x4_t idx;
	float alpha;

	alpha = vgetq_lane_f32(vals, 3);
	if (alpha > 1.0f) alpha = 1.0f;
	else if (alpha < 0.0f) alpha = 0.0f;
	alpha_v = vdupq_n_f32(alpha);

	if (alpha != 0) {
		vals = vdivq_f32(vals, alpha_v);
	}

	vals = vminq_f32(vmaxq_f32(vals, zero), one);
	idx = vcvtq_s32_f32(vmulq_f32(vals, scale));

	out[a_off] = (int)(alpha * 255.0f + 0.5f);
	oil_lut_store3_neon(out + rgb_off, idx, lut);
}

/* Unpremultiply and clamp a GA vector [g0, a0, g1, a1].
 * Spreads alpha to gray positions, divides gray by alpha (safe when alpha==0),
 * clamps both to [0,1], and blends the result back to GA layout.
 */
static inline float32x4_t unpremul_clamp_ga_neon(float32x4_t sum,
	float32x4_t zero, float32x4_t one, uint32x4_t blend_mask)
{
	float32x4_t alpha_spread, safe_alpha, divided, gray_clamped;
	uint32x4_t nz_mask;

	/* [s0,s1,s2,s3] -> [s1,s1,s3,s3] */
	alpha_spread = vtrn2q_f32(sum, sum);
	alpha_spread = vminq_f32(vmaxq_f32(alpha_spread, zero), one);
	nz_mask = vmvnq_u32(vceqq_f32(alpha_spread, zero));
	safe_alpha = vbslq_f32(nz_mask, alpha_spread, one);
	divided = vdivq_f32(sum, safe_alpha);
	gray_clamped = vminq_f32(vmaxq_f32(divided, zero), one);
	return vbslq_f32(blend_mask, alpha_spread, gray_clamped);
}

/* Dot product of four [C,M,Y,K] sample vectors with per-tap coefficients
 * packed as lanes [c0,c1,c2,c3] of `coeffs`: c0*smp0 + c1*smp1 + c2*smp2 +
 * c3*smp3.
 */
static inline float32x4_t oil_cmyk_dot4_neon(float32x4_t smp0, float32x4_t smp1,
	float32x4_t smp2, float32x4_t smp3, float32x4_t coeffs)
{
	float32x4_t result;
	result = vmulq_laneq_f32(smp0, coeffs, 0);
	result = vfmaq_laneq_f32(result, smp1, coeffs, 1);
	result = vfmaq_laneq_f32(result, smp2, coeffs, 2);
	result = vfmaq_laneq_f32(result, smp3, coeffs, 3);
	return result;
}

static void oil_yscale_out_nonlinear_neon(float *sums, int len, unsigned char *out)
{
	int i;
	float32x4_t vals;
	float32x4_t scale, half, zero, one;
	int32x4_t idx;

	scale = vdupq_n_f32(255.0f);
	half = vdupq_n_f32(0.5f);
	zero = vdupq_n_f32(0.0f);
	one = vdupq_n_f32(1.0f);

	for (i=0; i+3<len; i+=4) {
		vals = oil_consume_ch0_x4_neon(sums);
		idx = oil_clamp_round_idx_neon(vals, zero, one, scale, half);

		out[i]   = (unsigned char)vgetq_lane_s32(idx, 0);
		out[i+1] = (unsigned char)vgetq_lane_s32(idx, 1);
		out[i+2] = (unsigned char)vgetq_lane_s32(idx, 2);
		out[i+3] = (unsigned char)vgetq_lane_s32(idx, 3);

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
	float32x4_t scale_v, vals;
	int32x4_t idx;
	unsigned char *lut;

	lut = l2s_map;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));

	for (i=0; i+3<len; i+=4) {
		vals = oil_consume_ch0_x4_neon(sums);
		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));
		oil_lut_store4_neon(out + i, idx, lut);

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
	float32x4_t f0, f1;
	float32x4_t vals, result;
	float32x4_t scale_v, half, zero, one;
	uint32x4_t blend_mask;
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
		/* vals = [gray0, alpha0, gray1, alpha1] */
		vals = oil_consume_ch0_x4_neon(sums);
		result = unpremul_clamp_ga_neon(vals, zero, one, blend_mask);
		idx = vcvtq_s32_f32(vfmaq_f32(half, result, scale_v));

		/* Pack 4 ints -> 4 bytes */
		{
			int16x4_t n16 = vqmovn_s32(idx);
			uint8x8_t n8 = vqmovun_s16(vcombine_s16(n16, n16));
			vst1_lane_u32((uint32_t *)out, vreinterpret_u32_u8(n8), 0);
		}

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

		vst1q_f32(sums,     oil_shift_f_left_neon(f0));
		vst1q_f32(sums + 4, oil_shift_f_left_neon(f1));

		sums += 8;
		out += 2;
	}
}

static void oil_yscale_out_rgbx_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, vals, z;
	int32x4_t idx;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		vals = vld1q_f32(sums + tap_off);
		idx = vcvtq_s32_f32(vmulq_f32(vals, scale_v));
		oil_lut_store3_neon(out, idx, lut);
		out[3] = 255;

		/* Zero consumed tap */
		vst1q_f32(sums + tap_off, z);

		sums += 16;
		out += 4;
	}
}

static inline __attribute__((always_inline))
void yscale_out_alpha_neon_impl(float *sums, int width, unsigned char *out,
	int tap, int a_off, int rgb_off)
{
	int i, tap_off;
	float32x4_t scale_v, one, zero, z;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale_v = vdupq_n_f32((float)(l2s_len - 1));
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		oil_unpremul_rgba_lut_neon(vld1q_f32(sums + tap_off),
			zero, one, scale_v, lut, out, a_off, rgb_off);
		vst1q_f32(sums + tap_off, z);
		sums += 16;
		out += 4;
	}
}

static void oil_yscale_out_rgba_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	yscale_out_alpha_neon_impl(sums, width, out, tap, 3, 0);
}

static void oil_yscale_out_argb_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	yscale_out_alpha_neon_impl(sums, width, out, tap, 0, 1);
}

static void oil_yscale_out_cmyk_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, zero, one, half, z;
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
		idx = oil_clamp_round_idx_neon(vld1q_f32(sums + tap_off),
			zero, one, scale_v, half);

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
	float32x4_t sum;
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
		idx2 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		sum = oil_ydot4_load_neon(in, i + 8, c0, c1, c2, c3);
		idx3 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);
		idx4 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		sum = oil_ydot4_load_neon(in, i + 16, c0, c1, c2, c3);
		idx5 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 20, c0, c1, c2, c3);
		idx6 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		sum = oil_ydot4_load_neon(in, i + 24, c0, c1, c2, c3);
		idx7 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 28, c0, c1, c2, c3);
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
		idx2 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		sum = oil_ydot4_load_neon(in, i + 8, c0, c1, c2, c3);
		idx3 = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtnq_s32_f32(vmulq_f32(sum, scale));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
		idx2 = vcvtnq_s32_f32(vmulq_f32(sum2, scale));

		{
			int16x8_t n16 = vcombine_s16(vqmovn_s32(idx), vqmovn_s32(idx2));
			uint8x8_t n8 = vqmovun_s16(n16);
			vst1_u8(out + i, n8);
		}
	}

	for (; i+3<len; i+=4) {
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
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
	float32x4_t sum, sum2;
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
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);

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
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

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
	float32x4_t sum;
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
		idx2 = vcvtq_s32_f32(vmulq_f32(sum2, scale_v));

		sum3 = oil_ydot4_load_neon(in, i + 8, c0, c1, c2, c3);
		idx3 = vcvtq_s32_f32(vmulq_f32(sum3, scale_v));

		sum4 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
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
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
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
	float32x4_t sum;
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
		idx2 = vcvtq_s32_f32(vmulq_f32(sum2, scale_v));

		sum3 = oil_ydot4_load_neon(in, i + 8, c0, c1, c2, c3);
		idx3 = vcvtq_s32_f32(vmulq_f32(sum3, scale_v));

		sum4 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
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
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		idx = vcvtq_s32_f32(vmulq_f32(sum, scale_v));

		out[i]   = lut[vgetq_lane_s32(idx, 0)];
		out[i+1] = lut[vgetq_lane_s32(idx, 1)];
		out[i+2] = lut[vgetq_lane_s32(idx, 2)];
		out[i+3] = 255;
	}
}

static inline __attribute__((always_inline))
void yscale_up_alpha_neon_impl(float **in, int len, float *coeffs,
	unsigned char *out, int a_off, int rgb_off)
{
	int i;
	float32x4_t c0, c1, c2, c3;
	float32x4_t sum, sum2;
	float32x4_t scale_v, one, zero;
	unsigned char *lut;

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

		s0 = oil_ydot4_load_neon(in, i,      c0, c1, c2, c3);
		s1 = oil_ydot4_load_neon(in, i + 4,  c0, c1, c2, c3);
		s2 = oil_ydot4_load_neon(in, i + 8,  c0, c1, c2, c3);
		s3 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);

		oil_unpremul_rgba_lut_neon(s0, zero, one, scale_v, lut,
			out + i,      a_off, rgb_off);
		oil_unpremul_rgba_lut_neon(s1, zero, one, scale_v, lut,
			out + i + 4,  a_off, rgb_off);
		oil_unpremul_rgba_lut_neon(s2, zero, one, scale_v, lut,
			out + i + 8,  a_off, rgb_off);
		oil_unpremul_rgba_lut_neon(s3, zero, one, scale_v, lut,
			out + i + 12, a_off, rgb_off);
	}

	for (; i+7<len; i+=8) {
		sum  = oil_ydot4_load_neon(in, i,     c0, c1, c2, c3);
		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);
		oil_unpremul_rgba_lut_neon(sum, zero, one, scale_v, lut,
			out + i,     a_off, rgb_off);
		oil_unpremul_rgba_lut_neon(sum2, zero, one, scale_v, lut,
			out + i + 4, a_off, rgb_off);
	}

	for (; i<len; i+=4) {
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);
		oil_unpremul_rgba_lut_neon(sum, zero, one, scale_v, lut,
			out + i, a_off, rgb_off);
	}
}

static void oil_yscale_up_rgba_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	yscale_up_alpha_neon_impl(in, len, coeffs, out, 3, 0);
}

static void oil_yscale_up_argb_neon(float **in, int len, float *coeffs,
	unsigned char *out)
{
	yscale_up_alpha_neon_impl(in, len, coeffs, out, 0, 1);
}

static void oil_xscale_up_g_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	float32x4_t smp, coeffs;

	smp = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		/* push_f: shift right by 1 lane, insert new value at position 3 */
		smp = oil_push_f_neon(smp, i2f_map[in[i]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);
			float32x4_t t2 = oil_dot2_f_neon(smp, c0, c1);
			out[0] = vgetq_lane_f32(t2, 0);
			out[1] = vgetq_lane_f32(t2, 1);
			out += 2;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			coeffs = vld1q_f32(coeff_buf);
			out[0] = oil_dot1_f_neon(smp, coeffs);
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
		smp_a = oil_push_f_neon(smp_a, alpha_new);
		smp_g = oil_push_f_neon(smp_g, alpha_new * i2f_map[in[0]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			float32x4_t c0 = vld1q_f32(coeff_buf);
			float32x4_t c1 = vld1q_f32(coeff_buf + 4);

			/* gray dot products for 2 outputs */
			float32x4_t t2_g = oil_dot2_f_neon(smp_g, c0, c1);

			/* alpha dot products for 2 outputs */
			float32x4_t t2_a = oil_dot2_f_neon(smp_a, c0, c1);

			/* interleave: [gray0, alpha0, gray1, alpha1] */
			vst1q_f32(out, vzip1q_f32(t2_g, t2_a));
			out += 4;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			float32x4_t coeffs = vld1q_f32(coeff_buf);

			out[0] = oil_dot1_f_neon(smp_g, coeffs);
			out[1] = oil_dot1_f_neon(smp_a, coeffs);

			out += 2;
			coeff_buf += 4;
		}

		in += 2;
	}
}

static inline __attribute__((always_inline))
void oil_xscale_up_rgb_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf, float *lut)
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

		/* New pixel: [R, G, B, 0] */
		pixel = vsetq_lane_f32(lut[in[0]], vdupq_n_f32(0), 0);
		pixel = vsetq_lane_f32(lut[in[1]], pixel, 1);
		pixel = vsetq_lane_f32(lut[in[2]], pixel, 2);
		smp3 = pixel;

		j = border_buf[i];

		while (j >= 2) {
			float32x4_t result0, result1;

			result0 = oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf));
			result1 = oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf + 4));

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

		if (j) {
			float32x4_t result = oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf));
			out[0] = vgetq_lane_f32(result, 0);
			out[1] = vgetq_lane_f32(result, 1);
			out[2] = vgetq_lane_f32(result, 2);
			out += 3;
			coeff_buf += 4;
		}

		in += 3;
	}
}

static inline __attribute__((always_inline))
void oil_xscale_up_rgbx_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf, float *lut)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;

	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float32x4_t pixel;
		unsigned int px;

		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		memcpy(&px, in, 4);
		pixel = vsetq_lane_f32(lut[px & 0xFF], vdupq_n_f32(1.0f), 0);
		pixel = vsetq_lane_f32(lut[(px >> 8) & 0xFF], pixel, 1);
		pixel = vsetq_lane_f32(lut[(px >> 16) & 0xFF], pixel, 2);
		smp3 = pixel;

		j = border_buf[i];

		while (j >= 2) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			vst1q_f32(out + 4, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf + 4)));
			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		if (j) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static inline __attribute__((always_inline))
void xscale_up_alpha_neon_impl(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf, int a_off, int rgb_off, float *rgb_lut)
{
	int i, j;
	float32x4_t smp0, smp1, smp2, smp3;

	smp0 = vdupq_n_f32(0.0f);
	smp1 = vdupq_n_f32(0.0f);
	smp2 = vdupq_n_f32(0.0f);
	smp3 = vdupq_n_f32(0.0f);

	for (i=0; i<width_in; i++) {
		float alpha_new = i2f_map[in[a_off]];
		float32x4_t pixel;

		/* Shift tap window: oldest tap falls off */
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		/* New pixel: [alpha*R, alpha*G, alpha*B, alpha] */
		pixel = vsetq_lane_f32(alpha_new * rgb_lut[in[rgb_off]],     vdupq_n_f32(0), 0);
		pixel = vsetq_lane_f32(alpha_new * rgb_lut[in[rgb_off + 1]], pixel, 1);
		pixel = vsetq_lane_f32(alpha_new * rgb_lut[in[rgb_off + 2]], pixel, 2);
		pixel = vsetq_lane_f32(alpha_new, pixel, 3);
		smp3 = pixel;

		j = border_buf[i];

		while (j >= 2) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			vst1q_f32(out + 4, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf + 4)));
			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		if (j) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_xscale_up_rgba_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	xscale_up_alpha_neon_impl(in, width_in, out, coeff_buf, border_buf,
		3, 0, s2l_map);
}

static void oil_xscale_up_argb_neon(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	xscale_up_alpha_neon_impl(in, width_in, out, coeff_buf, border_buf,
		0, 1, s2l_map);
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
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			vst1q_f32(out + 4, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf + 4)));
			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
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

		sum = oil_shift_f_left_neon(sum);
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

		sum_g = oil_shift_f_left_neon(sum_g);
		sum_a = oil_shift_f_left_neon(sum_a);
	}
}

static inline __attribute__((always_inline))
void oil_scale_down_rgb_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	float *lut)
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

				sample_x = vdupq_n_f32(lut[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(lut[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(lut[in[2]]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sample_x = vdupq_n_f32(lut[in[3]]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(lut[in[4]]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(lut[in[5]]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				in += 6;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(lut[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(lut[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(lut[in[2]]);
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

				sample_x = vdupq_n_f32(lut[in[0]]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(lut[in[1]]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(lut[in[2]]);
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

		sum_r = oil_shift_f_left_neon(sum_r);
		sum_g = oil_shift_f_left_neon(sum_g);
		sum_b = oil_shift_f_left_neon(sum_b);
	}
}

#define PX_BYTE(px, idx) (((px) >> ((idx) * 8)) & 0xFF)

static inline __attribute__((always_inline))
void scale_down_alpha_neon_impl(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap, int a_off, int rgb_off, float *rgb_lut)
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
					vdupq_n_f32(i2f_map[PX_BYTE(px0, a_off)]));

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px0, rgb_off)]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px0, rgb_off + 1)]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px0, rgb_off + 2)]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_b);

				sum_a = vaddq_f32(coeffs_x_a, sum_a);

				coeffs_x2_a = vmulq_f32(coeffs_x2,
					vdupq_n_f32(i2f_map[PX_BYTE(px1, a_off)]));

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px1, rgb_off)]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_r2);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px1, rgb_off + 1)]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2_a, sample_x), sum_g2);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px1, rgb_off + 2)]);
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
					vdupq_n_f32(i2f_map[PX_BYTE(px, a_off)]));

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px, rgb_off)]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px, rgb_off + 1)]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px, rgb_off + 2)]);
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
					vdupq_n_f32(i2f_map[PX_BYTE(px, a_off)]));

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px, rgb_off)]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_r);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px, rgb_off + 1)]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x_a, sample_x), sum_g);

				sample_x = vdupq_n_f32(rgb_lut[PX_BYTE(px, rgb_off + 2)]);
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

		sum_r = oil_shift_f_left_neon(sum_r);
		sum_g = oil_shift_f_left_neon(sum_g);
		sum_b = oil_shift_f_left_neon(sum_b);
		sum_a = oil_shift_f_left_neon(sum_a);
	}
}

static void oil_scale_down_rgba_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	scale_down_alpha_neon_impl(in, sums_y_out, out_width, coeffs_x_f,
		border_buf, coeffs_y_f, tap, 3, 0, s2l_map);
}

static void oil_scale_down_argb_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	scale_down_alpha_neon_impl(in, sums_y_out, out_width, coeffs_x_f,
		border_buf, coeffs_y_f, tap, 0, 1, s2l_map);
}

static void oil_scale_down_rgba_nogamma_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	scale_down_alpha_neon_impl(in, sums_y_out, out_width, coeffs_x_f,
		border_buf, coeffs_y_f, tap, 3, 0, i2f_map);
}

static inline __attribute__((always_inline))
void oil_scale_down_rgbx_neon(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap, float *lut)
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

				sample_x = vdupq_n_f32(lut[px0 & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(lut[(px0 >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(lut[(px0 >> 16) & 0xFF]);
				sum_b = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_b);

				sample_x = vdupq_n_f32(lut[px1 & 0xFF]);
				sum_r2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_r2);

				sample_x = vdupq_n_f32(lut[(px1 >> 8) & 0xFF]);
				sum_g2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_g2);

				sample_x = vdupq_n_f32(lut[(px1 >> 16) & 0xFF]);
				sum_b2 = vaddq_f32(vmulq_f32(coeffs_x2, sample_x), sum_b2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = vld1q_f32(coeffs_x_f);

				sample_x = vdupq_n_f32(lut[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(lut[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(lut[(px >> 16) & 0xFF]);
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

				sample_x = vdupq_n_f32(lut[px & 0xFF]);
				sum_r = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_r);

				sample_x = vdupq_n_f32(lut[(px >> 8) & 0xFF]);
				sum_g = vaddq_f32(vmulq_f32(coeffs_x, sample_x), sum_g);

				sample_x = vdupq_n_f32(lut[(px >> 16) & 0xFF]);
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

		sum_r = oil_shift_f_left_neon(sum_r);
		sum_g = oil_shift_f_left_neon(sum_g);
		sum_b = oil_shift_f_left_neon(sum_b);
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

		sum_c = oil_shift_f_left_neon(sum_c);
		sum_m = oil_shift_f_left_neon(sum_m);
		sum_y = oil_shift_f_left_neon(sum_y);
		sum_k = oil_shift_f_left_neon(sum_k);
	}
}

static void oil_yscale_out_rgba_nogamma_neon(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	float32x4_t scale_v, one, zero, half, z;
	int32x4_t idx;

	tap_off = tap * 4;
	scale_v = vdupq_n_f32(255.0f);
	one = vdupq_n_f32(1.0f);
	zero = vdupq_n_f32(0.0f);
	half = vdupq_n_f32(0.5f);
	z = vdupq_n_f32(0.0f);

	for (i=0; i<width; i++) {
		idx = oil_unpremul_rgba_idx_neon(vld1q_f32(sums + tap_off),
			zero, one, scale_v, half);
		out[0] = vgetq_lane_s32(idx, 0);
		out[1] = vgetq_lane_s32(idx, 1);
		out[2] = vgetq_lane_s32(idx, 2);
		out[3] = vgetq_lane_s32(idx, 3);

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
	float32x4_t sum, sum2;
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
		s0 = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

		s1 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);

		s2 = oil_ydot4_load_neon(in, i + 8, c0, c1, c2, c3);

		s3 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);

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
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

		/* Pixel 1 */
		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);

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
		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

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
		uint8x8_t px8 = vreinterpret_u8_u32(vld1_dup_u32(
			(const uint32_t *)in));
		uint16x8_t px16 = vmovl_u8(px8);
		uint32x4_t px32 = vmovl_u16(vget_low_u16(px16));

		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		pixel = vmulq_f32(vcvtq_f32_u32(px32), inv255);
		pixel = vsetq_lane_f32(1.0f, pixel, 3);
		smp3 = pixel;

		j = border_buf[i];

		while (j >= 2) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			vst1q_f32(out + 4, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf + 4)));
			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		if (j) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_xscale_up_rgba_nogamma_neon(unsigned char *in, int width_in, float *out,
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
		float32x4_t pixel, pxf;
		float alpha_new;
		uint8x8_t px8 = vreinterpret_u8_u32(vld1_dup_u32(
			(const uint32_t *)in));
		uint16x8_t px16 = vmovl_u8(px8);
		uint32x4_t px32 = vmovl_u16(vget_low_u16(px16));

		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;

		pxf = vmulq_f32(vcvtq_f32_u32(px32), inv255);
		alpha_new = vgetq_lane_f32(pxf, 3);
		pixel = vmulq_n_f32(pxf, alpha_new);
		pixel = vsetq_lane_f32(alpha_new, pixel, 3);
		smp3 = pixel;

		j = border_buf[i];

		while (j >= 2) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			vst1q_f32(out + 4, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf + 4)));
			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		if (j) {
			vst1q_f32(out, oil_cmyk_dot4_neon(smp0, smp1, smp2, smp3,
				vld1q_f32(coeff_buf)));
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
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
		int32x4_t idx = oil_clamp_round_idx_neon(vld1q_f32(sums + tap_off),
			zero, one, scale_v, half);

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
	float32x4_t sum;
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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);

		sum3 = oil_ydot4_load_neon(in, i + 8, c0, c1, c2, c3);

		sum4 = oil_ydot4_load_neon(in, i + 12, c0, c1, c2, c3);

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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

		sum2 = oil_ydot4_load_neon(in, i + 4, c0, c1, c2, c3);

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

		sum = oil_ydot4_load_neon(in, i, c0, c1, c2, c3);

		clamped = vminq_f32(vmaxq_f32(sum, zero), one);
		idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(clamped, scale_v), half));

		out[i]   = vgetq_lane_s32(idx, 0);
		out[i+1] = vgetq_lane_s32(idx, 1);
		out[i+2] = vgetq_lane_s32(idx, 2);
		out[i+3] = 255;
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
		oil_xscale_up_rgb_neon(in, width_in, out, coeff_buf, border_buf, s2l_map);
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
		oil_xscale_up_rgbx_neon(in, width_in, out, coeff_buf, border_buf, s2l_map);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_xscale_up_rgb_neon(in, width_in, out, coeff_buf, border_buf, i2f_map);
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
		oil_scale_down_rgb_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, s2l_map);
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
		oil_scale_down_rgbx_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, s2l_map);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_scale_down_rgb_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, i2f_map);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_scale_down_rgba_nogamma_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_scale_down_rgbx_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, i2f_map);
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
