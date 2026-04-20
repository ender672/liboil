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
#include <immintrin.h>
#include <string.h>

/* Shift smp left by one float lane, zero-filling the top lane. */
static inline __m128 oil_shift_f_left_avx2(__m128 v)
{
	return (__m128)_mm_srli_si128(_mm_castps_si128(v), 4);
}

static void oil_shift_left_f_avx2(float *f)
{
	_mm_store_ps(f, oil_shift_f_left_avx2(_mm_load_ps(f)));
}

/* Shift smp left by one lane and place v into the now-empty top lane. */
static inline __m128 oil_push_f_avx2(__m128 smp, float v)
{
	__m128 newval, hi;
	smp = (__m128)_mm_srli_si128((__m128i)smp, 4);
	newval = _mm_set_ss(v);
	hi = _mm_shuffle_ps(smp, newval, _MM_SHUFFLE(0, 0, 3, 2));
	return _mm_shuffle_ps(smp, hi, _MM_SHUFFLE(2, 0, 1, 0));
}

/* Horizontal dot products dot(smp, c0) and dot(smp, c1) into lanes [0, 1]. */
static inline __m128 oil_dot2_f_avx2(__m128 smp, __m128 c0, __m128 c1)
{
	__m128 p0 = _mm_mul_ps(smp, c0);
	__m128 p1 = _mm_mul_ps(smp, c1);
	__m128 lo = _mm_unpacklo_ps(p0, p1);
	__m128 hh = _mm_unpackhi_ps(p0, p1);
	__m128 sum = _mm_add_ps(lo, hh);
	return _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
}

/* Horizontal dot product dot(smp, coeffs) as a scalar float. */
static inline float oil_dot1_f_avx2(__m128 smp, __m128 coeffs)
{
	__m128 prod = _mm_mul_ps(smp, coeffs);
	__m128 t2 = _mm_add_ps(prod, _mm_movehl_ps(prod, prod));
	t2 = _mm_add_ss(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1)));
	return _mm_cvtss_f32(t2);
}

/* Write 3 bytes to out[0..2] by indexing lut with the low three int32 lanes
 * of idx. Used when the 4th lane is either discarded or handled separately.
 */
static inline __attribute__((always_inline))
void oil_lut_store3_avx2(unsigned char *out, __m128i idx, unsigned char *lut)
{
	out[0] = lut[_mm_cvtsi128_si32(idx)];
	out[1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
	out[2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
}

/* Write 4 bytes to out[0..3] by indexing lut with all four int32 lanes of idx. */
static inline __attribute__((always_inline))
void oil_lut_store4_avx2(unsigned char *out, __m128i idx, unsigned char *lut)
{
	oil_lut_store3_avx2(out, idx, lut);
	out[3] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 12))];
}

/* 4-tap y-axis dot product: loads 4 floats from each of in[0..3] at offset
 * `off` and returns c0*in[0] + c1*in[1] + c2*in[2] + c3*in[3].
 */
static inline __m128 oil_ydot4_load_avx2(float **in, int off,
	__m128 c0, __m128 c1, __m128 c2, __m128 c3)
{
	__m128 v0 = _mm_loadu_ps(in[0] + off);
	__m128 v1 = _mm_loadu_ps(in[1] + off);
	__m128 v2 = _mm_loadu_ps(in[2] + off);
	__m128 v3 = _mm_loadu_ps(in[3] + off);
	return _mm_add_ps(
		_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
		_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
}

/* Unpremultiply a pair of packed GA samples [g0, a0, g1, a1], clamping alpha
 * to [0,1] and handling alpha==0 as passthrough-with-zero-gray. Returns the
 * four components scaled to byte-rounded int32 lanes.
 */
static inline __m128i oil_unpremul_ga_pair_idx_avx2(__m128 sum,
	__m128 zero, __m128 one, __m128 scale, __m128 half)
{
	__m128 alpha_spread, nz_mask, safe_alpha, divided, gray_clamped, result;
	__m128 blend_mask;

	/* mask: 0 for gray lanes (0,2), all-ones for alpha lanes (1,3) */
	blend_mask = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
	alpha_spread = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 1, 1));
	alpha_spread = _mm_min_ps(_mm_max_ps(alpha_spread, zero), one);
	nz_mask = _mm_cmpneq_ps(alpha_spread, zero);
	safe_alpha = _mm_or_ps(
		_mm_and_ps(nz_mask, alpha_spread),
		_mm_andnot_ps(nz_mask, one));
	divided = _mm_div_ps(sum, safe_alpha);
	gray_clamped = _mm_min_ps(_mm_max_ps(divided, zero), one);
	result = _mm_or_ps(
		_mm_andnot_ps(blend_mask, gray_clamped),
		_mm_and_ps(blend_mask, alpha_spread));
	return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(result, scale), half));
}

/* Build the 256-bit y-coefficient vectors used by oil_yacc_fma4_avx2. The
 * y-coefficients are permuted so each physical ring-buffer slot (0..3) receives
 * the matching coefficient for the current tap phase. Returns two __m256s:
 * cy_lo broadcasts coeffs for slots 0,1; cy_hi broadcasts for slots 2,3.
 */
static inline void oil_yacc_build_coeffs_avx2(float *coeffs_y_f, int tap,
	__m256 *cy_lo, __m256 *cy_hi)
{
	float cy[4];
	cy[tap & 3] = coeffs_y_f[0];
	cy[(tap + 1) & 3] = coeffs_y_f[1];
	cy[(tap + 2) & 3] = coeffs_y_f[2];
	cy[(tap + 3) & 3] = coeffs_y_f[3];
	*cy_lo = _mm256_set_m128(_mm_set1_ps(cy[1]), _mm_set1_ps(cy[0]));
	*cy_hi = _mm256_set_m128(_mm_set1_ps(cy[3]), _mm_set1_ps(cy[2]));
}

/* Single-channel vertical FMA accumulate: load 4 tap floats from sums_y_out,
 * FMA with broadcast(sum[0]) × coeffs_y, store back. Channel-major layout.
 */
static inline __attribute__((always_inline))
void oil_yacc_fma1_avx2(float *sums_y_out, __m128 sum, __m128 coeffs_y)
{
	__m128 sample_y = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 sums_y = _mm_load_ps(sums_y_out);
	sums_y = _mm_fmadd_ps(coeffs_y, sample_y, sums_y);
	_mm_store_ps(sums_y_out, sums_y);
}

/* Two-channel vertical FMA accumulate using 256-bit: load 8 tap floats from
 * sums_y_out (s0's 4 taps followed by s1's 4 taps), FMA both channels at
 * once, store back. Channel-major layout.
 */
static inline __attribute__((always_inline))
void oil_yacc_fma2_avx2(float *sums_y_out, __m128 s0, __m128 s1,
	__m128 coeffs_y)
{
	__m128 b0 = _mm_shuffle_ps(s0, s0, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 b1 = _mm_shuffle_ps(s1, s1, _MM_SHUFFLE(0, 0, 0, 0));
	__m256 sample_y = _mm256_set_m128(b1, b0);
	__m256 cy256 = _mm256_set_m128(coeffs_y, coeffs_y);
	__m256 sy256 = _mm256_loadu_ps(sums_y_out);
	sy256 = _mm256_fmadd_ps(cy256, sample_y, sy256);
	_mm256_storeu_ps(sums_y_out, sy256);
}

/* Vertical FMA accumulate for a 4-channel output pixel. s0..s3 each hold the
 * horizontal sum for one channel in lane 0; the four lane-0 values are packed
 * into a single 4-float channel vector and FMA'd into the 16-float ring-buffer
 * slice at sums_y_out, using the precomputed per-slot y-coefficients.
 */
static inline __attribute__((always_inline))
void oil_yacc_fma4_avx2(float *sums_y_out, __m128 s0, __m128 s1, __m128 s2,
	__m128 s3, __m256 cy_lo, __m256 cy_hi)
{
	__m128 ab, cd, pix;
	__m256 pix256, sy_lo, sy_hi;
	ab = _mm_unpacklo_ps(s0, s1);
	cd = _mm_unpacklo_ps(s2, s3);
	pix = _mm_movelh_ps(ab, cd);
	pix256 = _mm256_set_m128(pix, pix);
	sy_lo = _mm256_loadu_ps(sums_y_out);
	sy_hi = _mm256_loadu_ps(sums_y_out + 8);
	sy_lo = _mm256_fmadd_ps(cy_lo, pix256, sy_lo);
	sy_hi = _mm256_fmadd_ps(cy_hi, pix256, sy_hi);
	_mm256_storeu_ps(sums_y_out, sy_lo);
	_mm256_storeu_ps(sums_y_out + 8, sy_hi);
}

/* Unpremultiply a premultiplied RGBA sum (alpha in lane 3), clamp RGB and
 * alpha to [0,1], then scale to 0..255 (with rounding) for byte packing.
 * Lane 3 of the result contains the clamped alpha byte, not the reciprocal.
 */
static inline __m128i oil_unpremul_rgba_idx_avx2(__m128 vals,
	__m128 zero, __m128 one, __m128 scale, __m128 half)
{
	__m128 alpha_v, hi;
	alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
	alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
	if (_mm_cvtss_f32(alpha_v) != 0)
		vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
	vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
	hi = _mm_shuffle_ps(vals, alpha_v, _MM_SHUFFLE(0, 0, 2, 2));
	vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
	return _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
}

static void oil_yscale_out_nonlinear_avx2(float *sums, int len, unsigned char *out)
{
	int i;
	__m128 vals, ab, cd, f0, f1, f2, f3;
	__m128 scale, half, zero, one;
	__m128i idx;

	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	zero = _mm_setzero_ps();
	one = _mm_set1_ps(1.0f);

	for (i=0; i+7<len; i+=8) {
		__m128i idx2;
		__m128 vals2, ab2, cd2, g0, g1, g2, g3;

		f0 = _mm_load_ps(sums);
		f1 = _mm_load_ps(sums + 4);
		f2 = _mm_load_ps(sums + 8);
		f3 = _mm_load_ps(sums + 12);

		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		g0 = _mm_load_ps(sums + 16);
		g1 = _mm_load_ps(sums + 20);
		g2 = _mm_load_ps(sums + 24);
		g3 = _mm_load_ps(sums + 28);

		ab2 = _mm_shuffle_ps(g0, g1, _MM_SHUFFLE(0, 0, 0, 0));
		cd2 = _mm_shuffle_ps(g2, g3, _MM_SHUFFLE(0, 0, 0, 0));
		vals2 = _mm_shuffle_ps(ab2, cd2, _MM_SHUFFLE(2, 0, 2, 0));

		vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));

		idx = _mm_packs_epi32(idx, idx2);
		idx = _mm_packus_epi16(idx, idx);
		_mm_storel_epi64((__m128i *)(out + i), idx);

		_mm_store_ps(sums,      oil_shift_f_left_avx2(f0));
		_mm_store_ps(sums + 4,  oil_shift_f_left_avx2(f1));
		_mm_store_ps(sums + 8,  oil_shift_f_left_avx2(f2));
		_mm_store_ps(sums + 12, oil_shift_f_left_avx2(f3));
		_mm_store_ps(sums + 16, oil_shift_f_left_avx2(g0));
		_mm_store_ps(sums + 20, oil_shift_f_left_avx2(g1));
		_mm_store_ps(sums + 24, oil_shift_f_left_avx2(g2));
		_mm_store_ps(sums + 28, oil_shift_f_left_avx2(g3));

		sums += 32;
	}

	for (; i+3<len; i+=4) {
		f0 = _mm_load_ps(sums);
		f1 = _mm_load_ps(sums + 4);
		f2 = _mm_load_ps(sums + 8);
		f3 = _mm_load_ps(sums + 12);

		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		idx = _mm_packs_epi32(idx, idx);
		idx = _mm_packus_epi16(idx, idx);
		*(int *)(out + i) = _mm_cvtsi128_si32(idx);

		_mm_store_ps(sums,      oil_shift_f_left_avx2(f0));
		_mm_store_ps(sums + 4,  oil_shift_f_left_avx2(f1));
		_mm_store_ps(sums + 8,  oil_shift_f_left_avx2(f2));
		_mm_store_ps(sums + 12, oil_shift_f_left_avx2(f3));

		sums += 16;
	}

	for (; i<len; i++) {
		float v = *sums;
		if (v > 1.0f) v = 1.0f;
		else if (v < 0.0f) v = 0.0f;
		out[i] = (int)(v * 255.0f + 0.5f);
		oil_shift_left_f_avx2(sums);
		sums += 4;
	}
}

static void oil_yscale_out_linear_avx2(float *sums, int len, unsigned char *out)
{
	int i;
	__m128 scale, vals, ab, cd, f0, f1, f2, f3;
	__m128i idx;
	unsigned char *lut;

	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));

	for (i=0; i+3<len; i+=4) {
		f0 = _mm_load_ps(sums);
		f1 = _mm_load_ps(sums + 4);
		f2 = _mm_load_ps(sums + 8);
		f3 = _mm_load_ps(sums + 12);

		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

		oil_lut_store4_avx2(out + i, idx, lut);

		_mm_store_ps(sums,      oil_shift_f_left_avx2(f0));
		_mm_store_ps(sums + 4,  oil_shift_f_left_avx2(f1));
		_mm_store_ps(sums + 8,  oil_shift_f_left_avx2(f2));
		_mm_store_ps(sums + 12, oil_shift_f_left_avx2(f3));

		sums += 16;
	}

	for (; i<len; i++) {
		out[i] = lut[(int)(*sums * (l2s_len - 1))];
		oil_shift_left_f_avx2(sums);
		sums += 4;
	}
}

static void oil_yscale_out_ga_avx2(float *sums, int width, unsigned char *out)
{
	int i;
	__m128 v0, v1;
	float gray, alpha;

	for (i=0; i<width; i++) {
		v0 = _mm_load_ps(sums);
		v1 = _mm_load_ps(sums + 4);

		alpha = _mm_cvtss_f32(v1);
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;

		gray = _mm_cvtss_f32(v0);
		if (alpha != 0) {
			gray /= alpha;
		}
		if (gray > 1.0f) gray = 1.0f;
		else if (gray < 0.0f) gray = 0.0f;

		out[0] = (int)(gray * 255.0f + 0.5f);
		out[1] = (int)(alpha * 255.0f + 0.5f);

		_mm_store_ps(sums,     oil_shift_f_left_avx2(v0));
		_mm_store_ps(sums + 4, oil_shift_f_left_avx2(v1));

		sums += 8;
		out += 2;
	}
}

static void oil_yscale_out_rgbx_avx2(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	__m128 scale, vals;
	__m128i idx;
	__m128i z;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	z = _mm_setzero_si128();

	for (i=0; i<width; i++) {
		vals = _mm_load_ps(sums + tap_off);

		idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

		oil_lut_store3_avx2(out, idx, lut);
		out[3] = 255;

		/* Zero consumed tap */
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		sums += 16;
		out += 4;
	}
}

static void oil_xscale_up_g_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp;

	smp = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		smp = oil_push_f_avx2(smp, i2f_map[in[i]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 t2 = oil_dot2_f_avx2(smp,
				_mm_load_ps(coeff_buf),
				_mm_load_ps(coeff_buf + 4));
			out[0] = _mm_cvtss_f32(t2);
			out[1] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1)));
			out += 2;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			out[0] = oil_dot1_f_avx2(smp, _mm_load_ps(coeff_buf));
			out += 1;
			coeff_buf += 4;
		}
	}
}

static void oil_xscale_up_ga_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_g, smp_a;

	smp_g = _mm_setzero_ps();
	smp_a = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		float alpha_new = in[1] / 255.0f;
		smp_a = oil_push_f_avx2(smp_a, alpha_new);
		smp_g = oil_push_f_avx2(smp_g, alpha_new * i2f_map[in[0]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);
			__m128 t2_g = oil_dot2_f_avx2(smp_g, c0, c1);
			__m128 t2_a = oil_dot2_f_avx2(smp_a, c0, c1);

			/* interleave: [gray0, alpha0, gray1, alpha1] */
			_mm_storeu_ps(out, _mm_unpacklo_ps(t2_g, t2_a));
			out += 4;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);
			out[0] = oil_dot1_f_avx2(smp_g, coeffs);
			out[1] = oil_dot1_f_avx2(smp_a, coeffs);
			out += 2;
			coeff_buf += 4;
		}

		in += 2;
	}
}

static void oil_yscale_up_ga_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum, sum2;
	__m128 scale, half, zero, one;
	__m128i idx, idx2;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	zero = _mm_setzero_ps();
	one = _mm_set1_ps(1.0f);

	/* Process 4 GA pixels (8 floats) at a time */
	for (i=0; i+7<len; i+=8) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		sum2 = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);

		idx = oil_unpremul_ga_pair_idx_avx2(sum, zero, one, scale, half);
		idx2 = oil_unpremul_ga_pair_idx_avx2(sum2, zero, one, scale, half);

		idx = _mm_packs_epi32(idx, idx2);
		idx = _mm_packus_epi16(idx, idx);
		_mm_storel_epi64((__m128i *)(out + i), idx);
	}

	/* Process 2 GA pixels (4 floats) at a time */
	for (; i+3<len; i+=4) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);

		idx = oil_unpremul_ga_pair_idx_avx2(sum, zero, one, scale, half);
		idx = _mm_packs_epi32(idx, idx);
		idx = _mm_packus_epi16(idx, idx);
		*(int *)(out + i) = _mm_cvtsi128_si32(idx);
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

static void oil_yscale_up_rgb_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum;
	__m128 scale;
	__m128i idx;
	unsigned char *lut;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));

	for (i=0; i+7<len; i+=8) {
		__m128i idx2;
		__m128 sum2;

		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));

		sum2 = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);
		idx2 = _mm_cvttps_epi32(_mm_mul_ps(sum2, scale));

		oil_lut_store4_avx2(out + i, idx, lut);
		oil_lut_store4_avx2(out + i + 4, idx2, lut);
	}

	for (; i+3<len; i+=4) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));
		oil_lut_store4_avx2(out + i, idx, lut);
	}

	for (; i<len; i++) {
		out[i] = lut[(int)(
			(coeffs[0] * in[0][i] + coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] + coeffs[3] * in[3][i]) * (l2s_len - 1))];
	}
}

static void oil_yscale_up_rgbx_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum;
	__m128 scale;
	__m128i idx;
	unsigned char *lut;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));

	for (i=0; i+7<len; i+=8) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));
		oil_lut_store3_avx2(out + i, idx, lut);
		out[i+3] = 255;

		sum = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));
		oil_lut_store3_avx2(out + i + 4, idx, lut);
		out[i+7] = 255;
	}

	for (; i+3<len; i+=4) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));
		oil_lut_store3_avx2(out + i, idx, lut);
		out[i+3] = 255;
	}
}

static inline __attribute__((always_inline))
void oil_xscale_up_rgb_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf, float *lut)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b;

	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		smp_r = oil_push_f_avx2(smp_r, lut[in[0]]);
		smp_g = oil_push_f_avx2(smp_g, lut[in[1]]);
		smp_b = oil_push_f_avx2(smp_b, lut[in[2]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);
			__m128 t2_r = oil_dot2_f_avx2(smp_r, c0, c1);
			__m128 t2_g = oil_dot2_f_avx2(smp_g, c0, c1);
			__m128 t2_b = oil_dot2_f_avx2(smp_b, c0, c1);

			/* Store interleaved: [R0, G0, B0, R1, G1, B1] */
			__m128 rg = _mm_unpacklo_ps(t2_r, t2_g);
			_mm_storel_pi((__m64 *)out, rg);
			_mm_store_ss(out + 2, t2_b);
			_mm_storeh_pi((__m64 *)(out + 3), rg);
			_mm_store_ss(out + 5,
				_mm_shuffle_ps(t2_b, t2_b, _MM_SHUFFLE(1,1,1,1)));

			out += 6;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);
			out[0] = oil_dot1_f_avx2(smp_r, coeffs);
			out[1] = oil_dot1_f_avx2(smp_g, coeffs);
			out[2] = oil_dot1_f_avx2(smp_b, coeffs);
			out += 3;
			coeff_buf += 4;
		}

		in += 3;
	}
}

/* The X slot emitted here is unused downstream: both y-path consumers
 * (oil_yscale_up_rgbx_avx2 and its nogamma variant) overwrite it with 255.
 * We pack t2_b into the X lane as a harmless finite filler and skip the
 * smp_x push + dot product entirely.
 */
static inline __attribute__((always_inline))
void oil_xscale_up_rgbx_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf, float *lut)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b;

	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		smp_r = oil_push_f_avx2(smp_r, lut[in[0]]);
		smp_g = oil_push_f_avx2(smp_g, lut[in[1]]);
		smp_b = oil_push_f_avx2(smp_b, lut[in[2]]);

		j = border_buf[i];

		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);
			__m128 t2_r = oil_dot2_f_avx2(smp_r, c0, c1);
			__m128 t2_g = oil_dot2_f_avx2(smp_g, c0, c1);
			__m128 t2_b = oil_dot2_f_avx2(smp_b, c0, c1);

			__m128 rg = _mm_unpacklo_ps(t2_r, t2_g);
			__m128 bx = _mm_unpacklo_ps(t2_b, t2_b);
			_mm_storeu_ps(out, _mm_movelh_ps(rg, bx));
			_mm_storeu_ps(out + 4, _mm_movehl_ps(bx, rg));

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);
			out[0] = oil_dot1_f_avx2(smp_r, coeffs);
			out[1] = oil_dot1_f_avx2(smp_g, coeffs);
			out[2] = oil_dot1_f_avx2(smp_b, coeffs);
			out[3] = 1.0f;
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

/* Dot product of four [C,M,Y,K] sample vectors with per-tap coefficients
 * packed as lanes [c0,c1,c2,c3] of `coeffs`: c0*smp0 + c1*smp1 + c2*smp2 +
 * c3*smp3. Uses a parallel mul + tree add to keep the dependency chain short.
 */
static inline __attribute__((always_inline))
__m128 oil_cmyk_dot4_avx2(__m128 smp0, __m128 smp1, __m128 smp2, __m128 smp3,
	__m128 coeffs)
{
	__m128 p0, p1, p2, p3;
	p0 = _mm_mul_ps(smp0, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0,0,0,0)));
	p1 = _mm_mul_ps(smp1, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1,1,1,1)));
	p2 = _mm_mul_ps(smp2, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2,2,2,2)));
	p3 = _mm_mul_ps(smp3, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3,3,3,3)));
	return _mm_add_ps(_mm_add_ps(p0, p1), _mm_add_ps(p2, p3));
}

static void oil_xscale_up_cmyk_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp0, smp1, smp2, smp3, inv255;
	__m128i zero_i;

	/* Interleaved layout: each smpN = [C, M, Y, K] for one tap position */
	smp0 = _mm_setzero_ps();
	smp1 = _mm_setzero_ps();
	smp2 = _mm_setzero_ps();
	smp3 = _mm_setzero_ps();
	inv255 = _mm_set1_ps(1.0f / 255.0f);
	zero_i = _mm_setzero_si128();

	for (i=0; i<width_in; i++) {
		/* Push new pixel: load 4 bytes [C,M,Y,K], convert to floats */
		__m128i px = _mm_cvtsi32_si128(*(int *)in);
		px = _mm_unpacklo_epi8(px, zero_i);
		px = _mm_unpacklo_epi16(px, zero_i);
		smp0 = smp1;
		smp1 = smp2;
		smp2 = smp3;
		smp3 = _mm_mul_ps(_mm_cvtepi32_ps(px), inv255);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			_mm_storeu_ps(out, oil_cmyk_dot4_avx2(smp0, smp1, smp2,
				smp3, _mm_load_ps(coeff_buf)));
			_mm_storeu_ps(out + 4, oil_cmyk_dot4_avx2(smp0, smp1, smp2,
				smp3, _mm_load_ps(coeff_buf + 4)));
			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			_mm_storeu_ps(out, oil_cmyk_dot4_avx2(smp0, smp1, smp2,
				smp3, _mm_load_ps(coeff_buf)));
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_yscale_up_g_cmyk_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum;
	__m128 scale, half, zero, one;
	__m128i idx;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	zero = _mm_setzero_ps();
	one = _mm_set1_ps(1.0f);

	for (i=0; i+15<len; i+=16) {
		__m128i idx2, idx3, idx4;
		__m128 sum2;

		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

		sum2 = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);
		sum2 = _mm_min_ps(_mm_max_ps(sum2, zero), one);
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

		sum = oil_ydot4_load_avx2(in, i + 8, c0, c1, c2, c3);
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

		sum2 = oil_ydot4_load_avx2(in, i + 12, c0, c1, c2, c3);
		sum2 = _mm_min_ps(_mm_max_ps(sum2, zero), one);
		idx4 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

		idx = _mm_packs_epi32(idx, idx2);
		idx3 = _mm_packs_epi32(idx3, idx4);
		idx = _mm_packus_epi16(idx, idx3);
		_mm_storeu_si128((__m128i *)(out + i), idx);
	}

	for (; i+7<len; i+=8) {
		__m128i idx2;
		__m128 sum2;

		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

		sum2 = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);
		sum2 = _mm_min_ps(_mm_max_ps(sum2, zero), one);
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

		idx = _mm_packs_epi32(idx, idx2);
		idx = _mm_packus_epi16(idx, idx);
		_mm_storel_epi64((__m128i *)(out + i), idx);
	}

	for (; i+3<len; i+=4) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));
		idx = _mm_packs_epi32(idx, idx);
		idx = _mm_packus_epi16(idx, idx);
		*(int *)(out + i) = _mm_cvtsi128_si32(idx);
	}

	for (; i<len; i++) {
		float s = coeffs[0] * in[0][i] + coeffs[1] * in[1][i] +
			coeffs[2] * in[2][i] + coeffs[3] * in[3][i];
		if (s > 1.0f) s = 1.0f;
		else if (s < 0.0f) s = 0.0f;
		out[i] = (int)(s * 255.0f + 0.5f);
	}
}

/* Accumulate `count` horizontal G samples into `sum`, using a 4-way unrolled
 * inner loop with a 1-way scalar tail. Advances *in_p and *coeffs_x_f_p past
 * the consumed samples/coefficients. `sum` carries the partial sum shifted in
 * from the previous output position; the four parallel accumulators are
 * reduced before returning.
 */
static inline __attribute__((always_inline))
__m128 oil_xacc_g_heavy_avx2(unsigned char **in_p, float **coeffs_x_f_p,
	int count, __m128 sum)
{
	int j;
	unsigned char *in = *in_p;
	float *coeffs_x_f = *coeffs_x_f_p;
	__m128 coeffs_x, sample_x, sum2, sum3, sum4;

	sum2 = _mm_setzero_ps();
	sum3 = _mm_setzero_ps();
	sum4 = _mm_setzero_ps();

	for (j=0; j+3<count; j+=4) {
		coeffs_x = _mm_load_ps(coeffs_x_f);
		sample_x = _mm_set1_ps(i2f_map[in[0]]);
		sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);

		coeffs_x = _mm_load_ps(coeffs_x_f + 4);
		sample_x = _mm_set1_ps(i2f_map[in[1]]);
		sum2 = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum2);

		coeffs_x = _mm_load_ps(coeffs_x_f + 8);
		sample_x = _mm_set1_ps(i2f_map[in[2]]);
		sum3 = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum3);

		coeffs_x = _mm_load_ps(coeffs_x_f + 12);
		sample_x = _mm_set1_ps(i2f_map[in[3]]);
		sum4 = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum4);

		in += 4;
		coeffs_x_f += 16;
	}
	for (; j<count; j++) {
		coeffs_x = _mm_load_ps(coeffs_x_f);
		sample_x = _mm_set1_ps(i2f_map[in[0]]);
		sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);
		in += 1;
		coeffs_x_f += 4;
	}

	*in_p = in;
	*coeffs_x_f_p = coeffs_x_f;
	return _mm_add_ps(_mm_add_ps(sum, sum2), _mm_add_ps(sum3, sum4));
}

static void __attribute__((noinline)) oil_scale_down_g_heavy_avx2(
	unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i;
	__m128 sum;
	__m256 coeffs_y256, sums_y256, sample_y256;
	__m128 result_lo, result_hi;

	coeffs_y256 = _mm256_broadcast_ps((__m128 const *)coeffs_y_f);
	sum = _mm_setzero_ps();

	for (i=0; i+1<out_width; i+=2) {
		sum = oil_xacc_g_heavy_avx2(&in, &coeffs_x_f, border_buf[i], sum);
		result_lo = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
		sum = oil_shift_f_left_avx2(sum);

		sum = oil_xacc_g_heavy_avx2(&in, &coeffs_x_f, border_buf[i+1], sum);
		result_hi = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
		sum = oil_shift_f_left_avx2(sum);

		sums_y256 = _mm256_loadu_ps(sums_y_out);
		sample_y256 = _mm256_set_m128(result_hi, result_lo);
		sums_y256 = _mm256_add_ps(_mm256_mul_ps(coeffs_y256, sample_y256), sums_y256);
		_mm256_storeu_ps(sums_y_out, sums_y256);
		sums_y_out += 8;
	}

	for (; i<out_width; i++) {
		__m128 coeffs_y = _mm256_castps256_ps128(coeffs_y256);
		sum = oil_xacc_g_heavy_avx2(&in, &coeffs_x_f, border_buf[i], sum);
		oil_yacc_fma1_avx2(sums_y_out, sum, coeffs_y);
		sums_y_out += 4;
		sum = oil_shift_f_left_avx2(sum);
	}
}

static void oil_scale_down_g_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, sample_x, sum;
	__m256 coeffs_y256, sums_y256, sample_y256;
	__m128 result_lo, result_hi;

	coeffs_y256 = _mm256_broadcast_ps((__m128 const *)coeffs_y_f);
	sum = _mm_setzero_ps();

	for (i=0; i+1<out_width; i+=2) {
		for (j=0; j<border_buf[i]; j++) {
			coeffs_x = _mm_load_ps(coeffs_x_f);
			sample_x = _mm_set1_ps(i2f_map[in[0]]);
			sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);
			in += 1;
			coeffs_x_f += 4;
		}
		result_lo = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
		sum = oil_shift_f_left_avx2(sum);

		for (j=0; j<border_buf[i+1]; j++) {
			coeffs_x = _mm_load_ps(coeffs_x_f);
			sample_x = _mm_set1_ps(i2f_map[in[0]]);
			sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);
			in += 1;
			coeffs_x_f += 4;
		}
		result_hi = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
		sum = oil_shift_f_left_avx2(sum);

		sums_y256 = _mm256_loadu_ps(sums_y_out);
		sample_y256 = _mm256_set_m128(result_hi, result_lo);
		sums_y256 = _mm256_add_ps(_mm256_mul_ps(coeffs_y256, sample_y256), sums_y256);
		_mm256_storeu_ps(sums_y_out, sums_y256);
		sums_y_out += 8;
	}

	for (; i<out_width; i++) {
		__m128 coeffs_y = _mm256_castps256_ps128(coeffs_y256);
		for (j=0; j<border_buf[i]; j++) {
			coeffs_x = _mm_load_ps(coeffs_x_f);
			sample_x = _mm_set1_ps(i2f_map[in[0]]);
			sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);
			in += 1;
			coeffs_x_f += 4;
		}
		oil_yacc_fma1_avx2(sums_y_out, sum, coeffs_y);
		sums_y_out += 4;
		sum = oil_shift_f_left_avx2(sum);
	}
}

static void oil_scale_down_ga_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float alpha;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_g, sum_a;
	__m128 sum_g2, sum_a2;
	__m128 coeffs_y;

	coeffs_y = _mm_load_ps(coeffs_y_f);

	sum_g = _mm_setzero_ps();
	sum_a = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_g2 = _mm_setzero_ps();
			sum_a2 = _mm_setzero_ps();

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				alpha = i2f_map[in[1]];
				sample_x = _mm_set1_ps(i2f_map[in[0]] * alpha);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);
				sample_x = _mm_set1_ps(alpha);
				sum_a = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_a);

				alpha = i2f_map[in[3]];
				sample_x = _mm_set1_ps(i2f_map[in[2]] * alpha);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_g2);
				sample_x = _mm_set1_ps(alpha);
				sum_a2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_a2);

				in += 4;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);
				alpha = i2f_map[in[1]];
				sample_x = _mm_set1_ps(i2f_map[in[0]] * alpha);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);
				sample_x = _mm_set1_ps(alpha);
				sum_a = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_a);
				in += 2;
				coeffs_x_f += 4;
			}

			sum_g = _mm_add_ps(sum_g, sum_g2);
			sum_a = _mm_add_ps(sum_a, sum_a2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);
				alpha = i2f_map[in[1]];
				sample_x = _mm_set1_ps(i2f_map[in[0]] * alpha);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);
				sample_x = _mm_set1_ps(alpha);
				sum_a = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_a);
				in += 2;
				coeffs_x_f += 4;
			}
		}

		oil_yacc_fma2_avx2(sums_y_out, sum_g, sum_a, coeffs_y);
		sums_y_out += 8;

		sum_g = oil_shift_f_left_avx2(sum_g);
		sum_a = oil_shift_f_left_avx2(sum_a);
	}
}

static void oil_scale_down_rgb_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	float *lut)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m128 coeffs_y;

	coeffs_y = _mm_load_ps(coeffs_y_f);

	sum_r = _mm_setzero_ps();
	sum_g = _mm_setzero_ps();
	sum_b = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_r2 = _mm_setzero_ps();
			sum_g2 = _mm_setzero_ps();
			sum_b2 = _mm_setzero_ps();

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				sample_x = _mm_set1_ps(lut[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				sample_x = _mm_set1_ps(lut[in[3]]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_r2);

				sample_x = _mm_set1_ps(lut[in[4]]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_g2);

				sample_x = _mm_set1_ps(lut[in[5]]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_b2);

				in += 6;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(lut[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				in += 3;
				coeffs_x_f += 4;
			}

			sum_r = _mm_add_ps(sum_r, sum_r2);
			sum_g = _mm_add_ps(sum_g, sum_g2);
			sum_b = _mm_add_ps(sum_b, sum_b2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(lut[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				in += 3;
				coeffs_x_f += 4;
			}
		}

		oil_yacc_fma2_avx2(sums_y_out, sum_r, sum_g, coeffs_y);
		oil_yacc_fma1_avx2(sums_y_out + 8, sum_b, coeffs_y);
		sums_y_out += 12;

		sum_r = oil_shift_f_left_avx2(sum_r);
		sum_g = oil_shift_f_left_avx2(sum_g);
		sum_b = oil_shift_f_left_avx2(sum_b);
	}
}

/* Unpremultiply a premultiplied RGBA sum (alpha in lane 3) and emit one
 * output pixel: alpha as a rounded byte, RGB via the linear-to-sRGB LUT.
 * a_off/rgb_off select RGBA vs ARGB output layout.
 */
static inline __attribute__((always_inline))
void oil_unpremul_rgba_lut_avx2(__m128 vals, __m128 zero, __m128 one,
	__m128 scale, unsigned char *lut, unsigned char *out,
	int a_off, int rgb_off)
{
	__m128 alpha_v;
	__m128i idx;
	float alpha;

	alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
	alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
	alpha = _mm_cvtss_f32(alpha_v);

	if (alpha != 0) {
		vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
	}

	vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
	idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

	out[a_off] = (int)(alpha * 255.0f + 0.5f);
	oil_lut_store3_avx2(out + rgb_off, idx, lut);
}

static inline __attribute__((always_inline))
void oil_yscale_out_rgba_avx2(float *sums, int width, unsigned char *out,
	int tap, int a_off, int rgb_off)
{
	int i, tap_off;
	__m128 scale, one, zero;
	__m128i z;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i<width; i++) {
		oil_unpremul_rgba_lut_avx2(_mm_load_ps(sums + tap_off),
			zero, one, scale, lut, out, a_off, rgb_off);
		_mm_store_si128((__m128i *)(sums + tap_off), z);
		sums += 16;
		out += 4;
	}
}

static inline __attribute__((always_inline))
void oil_yscale_up_rgba_avx2(float **in, int len, float *coeffs,
	unsigned char *out, int a_off, int rgb_off)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum;
	__m128 scale, one, zero;
	unsigned char *lut;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();

	for (i=0; i<len; i+=4) {
		sum = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		oil_unpremul_rgba_lut_avx2(sum, zero, one, scale, lut,
			out + i, a_off, rgb_off);
	}
}

static inline __attribute__((always_inline))
void oil_xscale_up_rgba_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf, int a_off, int rgb_off, float *rgb_lut)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, smp_a;

	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();
	smp_a = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		float alpha_new = i2f_map[in[a_off]];

		smp_a = oil_push_f_avx2(smp_a, alpha_new);
		smp_r = oil_push_f_avx2(smp_r, alpha_new * rgb_lut[in[rgb_off]]);
		smp_g = oil_push_f_avx2(smp_g, alpha_new * rgb_lut[in[rgb_off + 1]]);
		smp_b = oil_push_f_avx2(smp_b, alpha_new * rgb_lut[in[rgb_off + 2]]);

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);
			__m128 t2_r = oil_dot2_f_avx2(smp_r, c0, c1);
			__m128 t2_g = oil_dot2_f_avx2(smp_g, c0, c1);
			__m128 t2_b = oil_dot2_f_avx2(smp_b, c0, c1);
			__m128 t2_a = oil_dot2_f_avx2(smp_a, c0, c1);

			/* Store interleaved: [R0, G0, B0, A0, R1, G1, B1, A1] */
			{
				__m128 rg = _mm_unpacklo_ps(t2_r, t2_g);
				__m128 ba = _mm_unpacklo_ps(t2_b, t2_a);
				_mm_storeu_ps(out, _mm_movelh_ps(rg, ba));
				_mm_storeu_ps(out + 4, _mm_movehl_ps(ba, rg));
			}

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);
			out[0] = oil_dot1_f_avx2(smp_r, coeffs);
			out[1] = oil_dot1_f_avx2(smp_g, coeffs);
			out[2] = oil_dot1_f_avx2(smp_b, coeffs);
			out[3] = oil_dot1_f_avx2(smp_a, coeffs);
			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static inline __attribute__((always_inline))
void oil_scale_down_rgba_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap, int a_off, int rgb_off, float *rgb_lut)
{
	int i, j;
	int a_sh, r_sh, g_sh, b_sh;
	__m128 coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	__m128 sum_r, sum_g, sum_b, sum_a;
	__m128 sum_r2, sum_g2, sum_b2, sum_a2;
	__m256 cy_lo, cy_hi;

	a_sh = a_off * 8;
	r_sh = rgb_off * 8;
	g_sh = r_sh + 8;
	b_sh = r_sh + 16;

	oil_yacc_build_coeffs_avx2(coeffs_y_f, tap, &cy_lo, &cy_hi);

	sum_r = _mm_setzero_ps();
	sum_g = _mm_setzero_ps();
	sum_b = _mm_setzero_ps();
	sum_a = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_r2 = _mm_setzero_ps();
			sum_g2 = _mm_setzero_ps();
			sum_b2 = _mm_setzero_ps();
			sum_a2 = _mm_setzero_ps();

			for (j=0; j+1<border_buf[i]; j+=2) {
				unsigned int px0, px1;
				memcpy(&px0, in, 4);
				memcpy(&px1, in + 4, 4);

				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[(px0 >> a_sh) & 0xFF]));

				sample_x = _mm_set1_ps(rgb_lut[(px0 >> r_sh) & 0xFF]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(rgb_lut[(px0 >> g_sh) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(rgb_lut[(px0 >> b_sh) & 0xFF]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				coeffs_x2_a = _mm_mul_ps(coeffs_x2, _mm_set1_ps(i2f_map[(px1 >> a_sh) & 0xFF]));

				sample_x = _mm_set1_ps(rgb_lut[(px1 >> r_sh) & 0xFF]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_r2);

				sample_x = _mm_set1_ps(rgb_lut[(px1 >> g_sh) & 0xFF]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_g2);

				sample_x = _mm_set1_ps(rgb_lut[(px1 >> b_sh) & 0xFF]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = _mm_add_ps(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[a_off]]));

				sample_x = _mm_set1_ps(rgb_lut[in[rgb_off]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(rgb_lut[in[rgb_off + 1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(rgb_lut[in[rgb_off + 2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_r = _mm_add_ps(sum_r, sum_r2);
			sum_g = _mm_add_ps(sum_g, sum_g2);
			sum_b = _mm_add_ps(sum_b, sum_b2);
			sum_a = _mm_add_ps(sum_a, sum_a2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[a_off]]));

				sample_x = _mm_set1_ps(rgb_lut[in[rgb_off]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(rgb_lut[in[rgb_off + 1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(rgb_lut[in[rgb_off + 2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		oil_yacc_fma4_avx2(sums_y_out, sum_r, sum_g, sum_b, sum_a,
			cy_lo, cy_hi);
		sums_y_out += 16;

		sum_r = oil_shift_f_left_avx2(sum_r);
		sum_g = oil_shift_f_left_avx2(sum_g);
		sum_b = oil_shift_f_left_avx2(sum_b);
		sum_a = oil_shift_f_left_avx2(sum_a);
	}
}

static void oil_yscale_out_cmyk_avx2(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	__m128 scale, half, vals, one, zero;
	__m128i idx, clamped, z;

	tap_off = tap * 4;
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i<width; i++) {
		vals = _mm_load_ps(sums + tap_off);
		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		clamped = _mm_packs_epi32(idx, idx);
		clamped = _mm_packus_epi16(clamped, clamped);

		*(int *)out = _mm_cvtsi128_si32(clamped);

		_mm_store_si128((__m128i *)(sums + tap_off), z);

		sums += 16;
		out += 4;
	}
}

static void oil_scale_down_cmyk_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_c, sum_m, sum_y, sum_k;
	__m128 sum_c2, sum_m2, sum_y2, sum_k2;
	__m256 cy_lo, cy_hi;

	oil_yacc_build_coeffs_avx2(coeffs_y_f, tap, &cy_lo, &cy_hi);

	sum_c = _mm_setzero_ps();
	sum_m = _mm_setzero_ps();
	sum_y = _mm_setzero_ps();
	sum_k = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 2) {
			sum_c2 = _mm_setzero_ps();
			sum_m2 = _mm_setzero_ps();
			sum_y2 = _mm_setzero_ps();
			sum_k2 = _mm_setzero_ps();

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				sample_x = _mm_set1_ps(i2f_map[in[0]]);
				sum_c = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_c);

				sample_x = _mm_set1_ps(i2f_map[in[1]]);
				sum_m = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_m);

				sample_x = _mm_set1_ps(i2f_map[in[2]]);
				sum_y = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_y);

				sample_x = _mm_set1_ps(i2f_map[in[3]]);
				sum_k = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_k);

				sample_x = _mm_set1_ps(i2f_map[in[4]]);
				sum_c2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_c2);

				sample_x = _mm_set1_ps(i2f_map[in[5]]);
				sum_m2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_m2);

				sample_x = _mm_set1_ps(i2f_map[in[6]]);
				sum_y2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_y2);

				sample_x = _mm_set1_ps(i2f_map[in[7]]);
				sum_k2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_k2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(i2f_map[in[0]]);
				sum_c = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_c);

				sample_x = _mm_set1_ps(i2f_map[in[1]]);
				sum_m = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_m);

				sample_x = _mm_set1_ps(i2f_map[in[2]]);
				sum_y = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_y);

				sample_x = _mm_set1_ps(i2f_map[in[3]]);
				sum_k = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_k);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_c = _mm_add_ps(sum_c, sum_c2);
			sum_m = _mm_add_ps(sum_m, sum_m2);
			sum_y = _mm_add_ps(sum_y, sum_y2);
			sum_k = _mm_add_ps(sum_k, sum_k2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(i2f_map[in[0]]);
				sum_c = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_c);

				sample_x = _mm_set1_ps(i2f_map[in[1]]);
				sum_m = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_m);

				sample_x = _mm_set1_ps(i2f_map[in[2]]);
				sum_y = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_y);

				sample_x = _mm_set1_ps(i2f_map[in[3]]);
				sum_k = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_k);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		oil_yacc_fma4_avx2(sums_y_out, sum_c, sum_m, sum_y, sum_k,
			cy_lo, cy_hi);
		sums_y_out += 16;

		sum_c = oil_shift_f_left_avx2(sum_c);
		sum_m = oil_shift_f_left_avx2(sum_m);
		sum_y = oil_shift_f_left_avx2(sum_y);
		sum_k = oil_shift_f_left_avx2(sum_k);
	}
}

static void oil_scale_down_rgbx_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap, float *lut)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m256 cy_lo, cy_hi;

	oil_yacc_build_coeffs_avx2(coeffs_y_f, tap, &cy_lo, &cy_hi);

	sum_r = _mm_setzero_ps();
	sum_g = _mm_setzero_ps();
	sum_b = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		if (border_buf[i] >= 4) {
			sum_r2 = _mm_setzero_ps();
			sum_g2 = _mm_setzero_ps();
			sum_b2 = _mm_setzero_ps();

			for (j=0; j+1<border_buf[i]; j+=2) {
				unsigned int px0, px1;
				memcpy(&px0, in, 4);
				memcpy(&px1, in + 4, 4);

				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				sample_x = _mm_set1_ps(lut[px0 & 0xFF]);
				sum_r = _mm_fmadd_ps(coeffs_x, sample_x, sum_r);

				sample_x = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
				sum_g = _mm_fmadd_ps(coeffs_x, sample_x, sum_g);

				sample_x = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
				sum_b = _mm_fmadd_ps(coeffs_x, sample_x, sum_b);

				sample_x = _mm_set1_ps(lut[px1 & 0xFF]);
				sum_r2 = _mm_fmadd_ps(coeffs_x2, sample_x, sum_r2);

				sample_x = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
				sum_g2 = _mm_fmadd_ps(coeffs_x2, sample_x, sum_g2);

				sample_x = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
				sum_b2 = _mm_fmadd_ps(coeffs_x2, sample_x, sum_b2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(lut[px & 0xFF]);
				sum_r = _mm_fmadd_ps(coeffs_x, sample_x, sum_r);

				sample_x = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
				sum_g = _mm_fmadd_ps(coeffs_x, sample_x, sum_g);

				sample_x = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
				sum_b = _mm_fmadd_ps(coeffs_x, sample_x, sum_b);

				in += 4;
				coeffs_x_f += 4;
			}

			sum_r = _mm_add_ps(sum_r, sum_r2);
			sum_g = _mm_add_ps(sum_g, sum_g2);
			sum_b = _mm_add_ps(sum_b, sum_b2);
		} else {
			for (j=0; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(lut[in[0]]);
				sum_r = _mm_fmadd_ps(coeffs_x, sample_x, sum_r);

				sample_x = _mm_set1_ps(lut[in[1]]);
				sum_g = _mm_fmadd_ps(coeffs_x, sample_x, sum_g);

				sample_x = _mm_set1_ps(lut[in[2]]);
				sum_b = _mm_fmadd_ps(coeffs_x, sample_x, sum_b);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* X slot is unused downstream (overwritten to 255), so feed
		 * sum_b again rather than computing a distinct sum_x.
		 */
		oil_yacc_fma4_avx2(sums_y_out, sum_r, sum_g, sum_b, sum_b,
			cy_lo, cy_hi);
		sums_y_out += 16;

		sum_r = oil_shift_f_left_avx2(sum_r);
		sum_g = oil_shift_f_left_avx2(sum_g);
		sum_b = oil_shift_f_left_avx2(sum_b);
	}
}

static void oil_yscale_out_rgbx_nogamma_avx2(float *sums, int width,
	unsigned char *out, int tap)
{
	int i, tap_off;
	__m128 scale, half, one, zero;
	__m128 vals;
	__m128i idx, packed;
	__m128i z, mask, x_val;

	tap_off = tap * 4;
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();
	mask = _mm_set_epi32(0, -1, -1, -1);
	x_val = _mm_set_epi32(255, 0, 0, 0);

	for (i=0; i+3<width; i+=4) {
		__m128 v0, v1, v2, v3;
		__m128i i0, i1, i2, i3, p01, p23;

		v0 = _mm_load_ps(sums + tap_off);
		v1 = _mm_load_ps(sums + 16 + tap_off);
		v2 = _mm_load_ps(sums + 32 + tap_off);
		v3 = _mm_load_ps(sums + 48 + tap_off);

		v0 = _mm_min_ps(_mm_max_ps(v0, zero), one);
		v1 = _mm_min_ps(_mm_max_ps(v1, zero), one);
		v2 = _mm_min_ps(_mm_max_ps(v2, zero), one);
		v3 = _mm_min_ps(_mm_max_ps(v3, zero), one);

		i0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v0, scale), half));
		i1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v1, scale), half));
		i2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v2, scale), half));
		i3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v3, scale), half));

		i0 = _mm_or_si128(_mm_and_si128(i0, mask), x_val);
		i1 = _mm_or_si128(_mm_and_si128(i1, mask), x_val);
		i2 = _mm_or_si128(_mm_and_si128(i2, mask), x_val);
		i3 = _mm_or_si128(_mm_and_si128(i3, mask), x_val);

		p01 = _mm_packs_epi32(i0, i1);
		p23 = _mm_packs_epi32(i2, i3);
		packed = _mm_packus_epi16(p01, p23);
		_mm_storeu_si128((__m128i *)out, packed);

		_mm_store_si128((__m128i *)(sums + tap_off), z);
		_mm_store_si128((__m128i *)(sums + 16 + tap_off), z);
		_mm_store_si128((__m128i *)(sums + 32 + tap_off), z);
		_mm_store_si128((__m128i *)(sums + 48 + tap_off), z);

		sums += 64;
		out += 16;
	}

	for (; i<width; i++) {
		vals = _mm_load_ps(sums + tap_off);

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
		idx = _mm_or_si128(_mm_and_si128(idx, mask), x_val);
		packed = _mm_packs_epi32(idx, idx);
		packed = _mm_packus_epi16(packed, packed);
		*(int *)out = _mm_cvtsi128_si32(packed);

		_mm_store_si128((__m128i *)(sums + tap_off), z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_out_rgba_nogamma_avx2(float *sums, int width,
	unsigned char *out, int tap)
{
	int i, tap_off;
	__m128 scale, half, one, zero;
	__m128i idx, packed;
	__m128i z;

	tap_off = tap * 4;
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i+3<width; i+=4) {
		__m128i idx2, idx3, idx4, packed2;

		idx = oil_unpremul_rgba_idx_avx2(_mm_load_ps(sums + tap_off),
			zero, one, scale, half);
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		idx2 = oil_unpremul_rgba_idx_avx2(_mm_load_ps(sums + 16 + tap_off),
			zero, one, scale, half);
		_mm_store_si128((__m128i *)(sums + 16 + tap_off), z);

		packed = _mm_packs_epi32(idx, idx2);

		idx3 = oil_unpremul_rgba_idx_avx2(_mm_load_ps(sums + 32 + tap_off),
			zero, one, scale, half);
		_mm_store_si128((__m128i *)(sums + 32 + tap_off), z);

		idx4 = oil_unpremul_rgba_idx_avx2(_mm_load_ps(sums + 48 + tap_off),
			zero, one, scale, half);
		_mm_store_si128((__m128i *)(sums + 48 + tap_off), z);

		packed2 = _mm_packs_epi32(idx3, idx4);
		packed = _mm_packus_epi16(packed, packed2);
		_mm_storeu_si128((__m128i *)out, packed);

		sums += 64;
		out += 16;
	}

	for (; i<width; i++) {
		idx = oil_unpremul_rgba_idx_avx2(_mm_load_ps(sums + tap_off),
			zero, one, scale, half);
		packed = _mm_packs_epi32(idx, idx);
		packed = _mm_packus_epi16(packed, packed);
		*(int *)out = _mm_cvtsi128_si32(packed);

		_mm_store_si128((__m128i *)(sums + tap_off), z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_up_rgba_nogamma_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum_a, sum_b;
	__m128 scale, half, one, zero;
	__m128i idx_a, idx_b, packed;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();

	for (i=0; i+7<len; i+=8) {
		sum_a = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);
		sum_b = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);

		idx_a = oil_unpremul_rgba_idx_avx2(sum_a, zero, one, scale, half);
		idx_b = oil_unpremul_rgba_idx_avx2(sum_b, zero, one, scale, half);

		packed = _mm_packs_epi32(idx_a, idx_b);
		packed = _mm_packus_epi16(packed, packed);
		_mm_storel_epi64((__m128i *)(out + i), packed);
	}

	for (; i<len; i+=4) {
		sum_a = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);

		idx_a = oil_unpremul_rgba_idx_avx2(sum_a, zero, one, scale, half);
		packed = _mm_packs_epi32(idx_a, idx_a);
		packed = _mm_packus_epi16(packed, packed);
		*(int *)(out + i) = _mm_cvtsi128_si32(packed);
	}
}

static void oil_yscale_up_rgbx_nogamma_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 sum_a, sum_b;
	__m128 scale, half, one, zero;
	__m128i idx_a, idx_b, packed;
	__m128i mask, x_val;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	mask = _mm_set_epi32(0, -1, -1, -1);
	x_val = _mm_set_epi32(255, 0, 0, 0);

	for (i=0; i+7<len; i+=8) {
		/* Pixel 1: 4 floats [R, G, B, X] */
		sum_a = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);

		/* Pixel 2 */
		sum_b = oil_ydot4_load_avx2(in, i + 4, c0, c1, c2, c3);

		/* Clamp, scale, and force X=255 for pixel 1 */
		sum_a = _mm_min_ps(_mm_max_ps(sum_a, zero), one);
		idx_a = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum_a, scale), half));
		idx_a = _mm_or_si128(_mm_and_si128(idx_a, mask), x_val);

		/* Clamp, scale, and force X=255 for pixel 2 */
		sum_b = _mm_min_ps(_mm_max_ps(sum_b, zero), one);
		idx_b = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum_b, scale), half));
		idx_b = _mm_or_si128(_mm_and_si128(idx_b, mask), x_val);

		/* Pack both pixels to bytes and store 8 bytes */
		packed = _mm_packs_epi32(idx_a, idx_b);
		packed = _mm_packus_epi16(packed, packed);
		_mm_storel_epi64((__m128i *)(out + i), packed);
	}

	for (; i<len; i+=4) {
		sum_a = oil_ydot4_load_avx2(in, i, c0, c1, c2, c3);

		sum_a = _mm_min_ps(_mm_max_ps(sum_a, zero), one);
		idx_a = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum_a, scale), half));
		idx_a = _mm_or_si128(_mm_and_si128(idx_a, mask), x_val);
		packed = _mm_packs_epi32(idx_a, idx_a);
		packed = _mm_packus_epi16(packed, packed);
		*(int *)(out + i) = _mm_cvtsi128_si32(packed);
	}
}

/* AVX2 dispatch functions */

static float *get_rb_line(struct oil_scale *os, int line)
{
	int sl_len;
	sl_len = OIL_CMP(os->cs) * os->out_width;
	return os->rb + line * sl_len;
}

static void yscale_out_avx2(float *sums, int width, unsigned char *out,
	enum oil_colorspace cs, int tap)
{
	int sl_len;

	sl_len = width * OIL_CMP(cs);

	switch(cs) {
	case OIL_CS_G:
		oil_yscale_out_nonlinear_avx2(sums, sl_len, out);
		break;
	case OIL_CS_CMYK:
		oil_yscale_out_cmyk_avx2(sums, width, out, tap);
		break;
	case OIL_CS_GA:
		oil_yscale_out_ga_avx2(sums, width, out);
		break;
	case OIL_CS_RGB:
		oil_yscale_out_linear_avx2(sums, sl_len, out);
		break;
	case OIL_CS_RGBA:
		oil_yscale_out_rgba_avx2(sums, width, out, tap, 3, 0);
		break;
	case OIL_CS_ARGB:
		oil_yscale_out_rgba_avx2(sums, width, out, tap, 0, 1);
		break;
	case OIL_CS_RGBX:
		oil_yscale_out_rgbx_avx2(sums, width, out, tap);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_yscale_out_nonlinear_avx2(sums, sl_len, out);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_yscale_out_rgba_nogamma_avx2(sums, width, out, tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_yscale_out_rgbx_nogamma_avx2(sums, width, out, tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void yscale_up_avx2(float **in, int len, float *coeffs,
	unsigned char *out, enum oil_colorspace cs)
{
	switch(cs) {
	case OIL_CS_G:
	case OIL_CS_CMYK:
		oil_yscale_up_g_cmyk_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_GA:
		oil_yscale_up_ga_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_RGB:
		oil_yscale_up_rgb_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_RGBA:
		oil_yscale_up_rgba_avx2(in, len, coeffs, out, 3, 0);
		break;
	case OIL_CS_ARGB:
		oil_yscale_up_rgba_avx2(in, len, coeffs, out, 0, 1);
		break;
	case OIL_CS_RGBX:
		oil_yscale_up_rgbx_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_yscale_up_g_cmyk_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_yscale_up_rgba_nogamma_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_yscale_up_rgbx_nogamma_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void xscale_up_avx2(unsigned char *in, int width_in, float *out,
	enum oil_colorspace cs_in, float *coeff_buf, int *border_buf)
{
	switch(cs_in) {
	case OIL_CS_RGB:
		oil_xscale_up_rgb_avx2(in, width_in, out, coeff_buf, border_buf, s2l_map);
		break;
	case OIL_CS_G:
		oil_xscale_up_g_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_CMYK:
		oil_xscale_up_cmyk_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA:
		oil_xscale_up_rgba_avx2(in, width_in, out, coeff_buf, border_buf, 3, 0, s2l_map);
		break;
	case OIL_CS_GA:
		oil_xscale_up_ga_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_ARGB:
		oil_xscale_up_rgba_avx2(in, width_in, out, coeff_buf, border_buf, 0, 1, s2l_map);
		break;
	case OIL_CS_RGBX:
		oil_xscale_up_rgbx_avx2(in, width_in, out, coeff_buf, border_buf, s2l_map);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_xscale_up_rgb_avx2(in, width_in, out, coeff_buf, border_buf, i2f_map);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_xscale_up_rgba_avx2(in, width_in, out, coeff_buf, border_buf, 3, 0, i2f_map);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_xscale_up_rgbx_avx2(in, width_in, out, coeff_buf, border_buf, i2f_map);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}

static void down_scale_in_avx2(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	switch(os->cs) {
	case OIL_CS_RGB:
		oil_scale_down_rgb_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, s2l_map);
		break;
	case OIL_CS_G:
		if (os->in_width >= os->out_width * 2) {
			oil_scale_down_g_heavy_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		} else {
			oil_scale_down_g_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		}
		break;
	case OIL_CS_CMYK:
		oil_scale_down_cmyk_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBA:
		oil_scale_down_rgba_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, 3, 0, s2l_map);
		break;
	case OIL_CS_GA:
		oil_scale_down_ga_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_ARGB:
		oil_scale_down_rgba_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, 0, 1, s2l_map);
		break;
	case OIL_CS_RGBX:
		oil_scale_down_rgbx_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, s2l_map);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_scale_down_rgb_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, i2f_map);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_scale_down_rgba_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, 3, 0, i2f_map);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_scale_down_rgbx_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap, i2f_map);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}

	os->borders_y[os->out_pos] -= 1;
	os->in_pos++;
}

static void up_scale_in_avx2(struct oil_scale *os, unsigned char *in)
{
	float *tmp;

	tmp = get_rb_line(os, os->in_pos % 4);
	xscale_up_avx2(in, os->in_width, tmp, os->cs, os->coeffs_x, os->borders_x);

	os->in_pos++;
}

int oil_scale_in_avx2(struct oil_scale *os, unsigned char *in)
{
	if (oil_scale_slots(os) == 0) {
		return -1;
	}
	if (os->out_width > os->in_width) {
		up_scale_in_avx2(os, in);
	} else {
		down_scale_in_avx2(os, in);
	}
	return 0;
}

int oil_scale_out_avx2(struct oil_scale *os, unsigned char *out)
{
	int i, sl_len;
	float *in[4];

	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	if (os->out_height <= os->in_height) {
		yscale_out_avx2(os->sums_y, os->out_width, out, os->cs, os->sums_y_tap);
		os->sums_y_tap = (os->sums_y_tap + 1) & 3;
	} else {
		sl_len = OIL_CMP(os->cs) * os->out_width;
		for (i=0; i<4; i++) {
			in[i] = get_rb_line(os, (os->in_pos + i) % 4);
		}
		yscale_up_avx2(in, sl_len, os->coeffs_y + os->out_pos * 4, out,
			os->cs);
		os->borders_y[os->in_pos - 1] -= 1;
	}

	os->out_pos++;
	return 0;
}


