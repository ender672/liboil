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

static void oil_shift_left_f_avx2(float *f)
{
	__m128i v = _mm_load_si128((__m128i *)f);
	_mm_store_si128((__m128i *)f, _mm_srli_si128(v, 4));
}

static void oil_yscale_out_nonlinear_avx2(float *sums, int len, unsigned char *out)
{
	int i;
	__m128 vals, ab, cd, f0, f1, f2, f3;
	__m128 scale, half, zero, one;
	__m128i idx, v0, v1, v2, v3;

	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	zero = _mm_setzero_ps();
	one = _mm_set1_ps(1.0f);

	for (i=0; i+7<len; i+=8) {
		__m128i idx2;
		__m128i w0, w1, w2, w3;
		__m128 vals2, ab2, cd2, g0, g1, g2, g3;

		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));
		v2 = _mm_load_si128((__m128i *)(sums + 8));
		v3 = _mm_load_si128((__m128i *)(sums + 12));

		f0 = _mm_castsi128_ps(v0);
		f1 = _mm_castsi128_ps(v1);
		f2 = _mm_castsi128_ps(v2);
		f3 = _mm_castsi128_ps(v3);
		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		w0 = _mm_load_si128((__m128i *)(sums + 16));
		w1 = _mm_load_si128((__m128i *)(sums + 20));
		w2 = _mm_load_si128((__m128i *)(sums + 24));
		w3 = _mm_load_si128((__m128i *)(sums + 28));

		g0 = _mm_castsi128_ps(w0);
		g1 = _mm_castsi128_ps(w1);
		g2 = _mm_castsi128_ps(w2);
		g3 = _mm_castsi128_ps(w3);
		ab2 = _mm_shuffle_ps(g0, g1, _MM_SHUFFLE(0, 0, 0, 0));
		cd2 = _mm_shuffle_ps(g2, g3, _MM_SHUFFLE(0, 0, 0, 0));
		vals2 = _mm_shuffle_ps(ab2, cd2, _MM_SHUFFLE(2, 0, 2, 0));

		vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));

		idx = _mm_packs_epi32(idx, idx2);
		idx = _mm_packus_epi16(idx, idx);
		_mm_storel_epi64((__m128i *)(out + i), idx);

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));
		_mm_store_si128((__m128i *)(sums + 8), _mm_srli_si128(v2, 4));
		_mm_store_si128((__m128i *)(sums + 12), _mm_srli_si128(v3, 4));
		_mm_store_si128((__m128i *)(sums + 16), _mm_srli_si128(w0, 4));
		_mm_store_si128((__m128i *)(sums + 20), _mm_srli_si128(w1, 4));
		_mm_store_si128((__m128i *)(sums + 24), _mm_srli_si128(w2, 4));
		_mm_store_si128((__m128i *)(sums + 28), _mm_srli_si128(w3, 4));

		sums += 32;
	}

	for (; i+3<len; i+=4) {
		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));
		v2 = _mm_load_si128((__m128i *)(sums + 8));
		v3 = _mm_load_si128((__m128i *)(sums + 12));

		f0 = _mm_castsi128_ps(v0);
		f1 = _mm_castsi128_ps(v1);
		f2 = _mm_castsi128_ps(v2);
		f3 = _mm_castsi128_ps(v3);
		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		idx = _mm_packs_epi32(idx, idx);
		idx = _mm_packus_epi16(idx, idx);
		*(int *)(out + i) = _mm_cvtsi128_si32(idx);

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));
		_mm_store_si128((__m128i *)(sums + 8), _mm_srli_si128(v2, 4));
		_mm_store_si128((__m128i *)(sums + 12), _mm_srli_si128(v3, 4));

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
	__m128i idx, v0, v1, v2, v3;
	unsigned char *lut;

	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));

	for (i=0; i+3<len; i+=4) {
		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));
		v2 = _mm_load_si128((__m128i *)(sums + 8));
		v3 = _mm_load_si128((__m128i *)(sums + 12));

		f0 = _mm_castsi128_ps(v0);
		f1 = _mm_castsi128_ps(v1);
		f2 = _mm_castsi128_ps(v2);
		f3 = _mm_castsi128_ps(v3);
		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

		out[i]   = lut[_mm_cvtsi128_si32(idx)];
		out[i+1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+3] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 12))];

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));
		_mm_store_si128((__m128i *)(sums + 8), _mm_srli_si128(v2, 4));
		_mm_store_si128((__m128i *)(sums + 12), _mm_srli_si128(v3, 4));

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
	__m128i v0, v1;
	float gray, alpha;

	for (i=0; i<width; i++) {
		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));

		alpha = _mm_cvtss_f32(_mm_castsi128_ps(v1));
		if (alpha > 1.0f) alpha = 1.0f;
		else if (alpha < 0.0f) alpha = 0.0f;

		gray = _mm_cvtss_f32(_mm_castsi128_ps(v0));
		if (alpha != 0) {
			gray /= alpha;
		}
		if (gray > 1.0f) gray = 1.0f;
		else if (gray < 0.0f) gray = 0.0f;

		out[0] = (int)(gray * 255.0f + 0.5f);
		out[1] = (int)(alpha * 255.0f + 0.5f);

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));

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

		out[0] = lut[_mm_cvtsi128_si32(idx)];
		out[1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
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
	__m128 smp, newval, hi, coeffs, prod, t1, t2;

	smp = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		/* push_f: shift left, insert new value at position 3 */
		smp = (__m128)_mm_srli_si128((__m128i)smp, 4);
		newval = _mm_set_ss(i2f_map[in[i]]);
		hi = _mm_shuffle_ps(smp, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp = _mm_shuffle_ps(smp, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);
			__m128 p0 = _mm_mul_ps(smp, c0);
			__m128 p1 = _mm_mul_ps(smp, c1);
			__m128 lo = _mm_unpacklo_ps(p0, p1);
			__m128 hh = _mm_unpackhi_ps(p0, p1);
			__m128 sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			t2 = _mm_add_ps(sum, t1);
			out[0] = _mm_cvtss_f32(t2);
			out[1] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1)));
			out += 2;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			coeffs = _mm_load_ps(coeff_buf);
			prod = _mm_mul_ps(smp, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);
			out += 1;
			coeff_buf += 4;
		}
	}
}

static void oil_xscale_up_ga_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_g, smp_a, newval, hi;

	smp_g = _mm_setzero_ps();
	smp_a = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		/* push_f for alpha: shift left, insert new alpha at position 3 */
		float alpha_new = in[1] / 255.0f;
		smp_a = (__m128)_mm_srli_si128((__m128i)smp_a, 4);
		newval = _mm_set_ss(alpha_new);
		hi = _mm_shuffle_ps(smp_a, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_a = _mm_shuffle_ps(smp_a, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for gray: premultiplied by new alpha */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(alpha_new * i2f_map[in[0]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* gray dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			__m128 lo = _mm_unpacklo_ps(pg0, pg1);
			__m128 hh = _mm_unpackhi_ps(pg0, pg1);
			__m128 sum_g = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum_g, sum_g);
			__m128 t2_g = _mm_add_ps(sum_g, t1);

			/* alpha dot products for 2 outputs */
			__m128 pa0 = _mm_mul_ps(smp_a, c0);
			__m128 pa1 = _mm_mul_ps(smp_a, c1);
			lo = _mm_unpacklo_ps(pa0, pa1);
			hh = _mm_unpackhi_ps(pa0, pa1);
			__m128 sum_a = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum_a, sum_a);
			__m128 t2_a = _mm_add_ps(sum_a, t1);

			/* interleave: [gray0, alpha0, gray1, alpha1] */
			_mm_storeu_ps(out, _mm_unpacklo_ps(t2_g, t2_a));
			out += 4;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);

			__m128 prod_g = _mm_mul_ps(smp_g, coeffs);
			__m128 t1 = _mm_movehl_ps(prod_g, prod_g);
			__m128 t2 = _mm_add_ps(prod_g, t1);
			prod_g = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod_g);
			out[0] = _mm_cvtss_f32(t2);

			__m128 prod_a = _mm_mul_ps(smp_a, coeffs);
			t1 = _mm_movehl_ps(prod_a, prod_a);
			t2 = _mm_add_ps(prod_a, t1);
			prod_a = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod_a);
			out[1] = _mm_cvtss_f32(t2);

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
	__m128 v0, v1, v2, v3, sum, sum2;
	__m128 scale, half, zero, one;
	__m128 alpha_spread, nz_mask, safe_alpha, divided, gray_clamped, result;
	__m128 blend_mask;
	__m128i idx;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	zero = _mm_setzero_ps();
	one = _mm_set1_ps(1.0f);
	/* mask: 0 for gray positions (0,2), all-ones for alpha positions (1,3) */
	blend_mask = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));

	/* Process 4 GA pixels (8 floats) at a time */
	for (i=0; i+7<len; i+=8) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum2 = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		/* sum = [g0, a0, g1, a1], sum2 = [g2, a2, g3, a3] */

		/* Process first pair: spread alpha to both lanes */
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
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(result, scale), half));

		/* Process second pair */
		alpha_spread = _mm_shuffle_ps(sum2, sum2, _MM_SHUFFLE(3, 3, 1, 1));
		alpha_spread = _mm_min_ps(_mm_max_ps(alpha_spread, zero), one);
		nz_mask = _mm_cmpneq_ps(alpha_spread, zero);
		safe_alpha = _mm_or_ps(
			_mm_and_ps(nz_mask, alpha_spread),
			_mm_andnot_ps(nz_mask, one));
		divided = _mm_div_ps(sum2, safe_alpha);
		gray_clamped = _mm_min_ps(_mm_max_ps(divided, zero), one);
		result = _mm_or_ps(
			_mm_andnot_ps(blend_mask, gray_clamped),
			_mm_and_ps(blend_mask, alpha_spread));
		__m128i idx2 = _mm_cvttps_epi32(
			_mm_add_ps(_mm_mul_ps(result, scale), half));

		/* Pack 8 ints -> 8 bytes */
		idx = _mm_packs_epi32(idx, idx2);
		idx = _mm_packus_epi16(idx, idx);
		_mm_storel_epi64((__m128i *)(out + i), idx);
	}

	/* Process 2 GA pixels (4 floats) at a time */
	for (; i+3<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

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
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(result, scale), half));
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
	__m128 v0, v1, v2, v3, sum;
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

		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));

		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum2 = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		idx2 = _mm_cvttps_epi32(_mm_mul_ps(sum2, scale));

		out[i]   = lut[_mm_cvtsi128_si32(idx)];
		out[i+1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+3] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 12))];
		out[i+4] = lut[_mm_cvtsi128_si32(idx2)];
		out[i+5] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx2, 4))];
		out[i+6] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx2, 8))];
		out[i+7] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx2, 12))];
	}

	for (; i+3<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));
		out[i]   = lut[_mm_cvtsi128_si32(idx)];
		out[i+1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+3] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 12))];
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
	__m128 v0, v1, v2, v3, sum;
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
		/* Pixel 0: 4 floats [R, G, B, X] */
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));

		out[i]   = lut[_mm_cvtsi128_si32(idx)];
		out[i+1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+3] = 255;

		/* Pixel 1: next 4 floats */
		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));

		out[i+4] = lut[_mm_cvtsi128_si32(idx)];
		out[i+5] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+6] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+7] = 255;
	}

	for (; i+3<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		idx = _mm_cvttps_epi32(_mm_mul_ps(sum, scale));

		out[i]   = lut[_mm_cvtsi128_si32(idx)];
		out[i+1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+3] = 255;
	}
}

static void oil_xscale_up_rgb_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, newval, hi;

	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		/* push_f for R: shift left, insert new value at position 3 */
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(s2l_map[in[0]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for G */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(s2l_map[in[1]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for B */
		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(s2l_map[in[2]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* R dot products for 2 outputs */
			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			/* G dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			/* B dot products for 2 outputs */
			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

			/* Store interleaved: [R0, G0, B0, R1, G1, B1] */
			out[0] = _mm_cvtss_f32(t2_r);
			out[1] = _mm_cvtss_f32(t2_g);
			out[2] = _mm_cvtss_f32(t2_b);
			out[3] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_r, t2_r, _MM_SHUFFLE(1,1,1,1)));
			out[4] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_g, t2_g, _MM_SHUFFLE(1,1,1,1)));
			out[5] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_b, t2_b, _MM_SHUFFLE(1,1,1,1)));

			out += 6;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			out += 3;
			coeff_buf += 4;
		}

		in += 3;
	}
}

static void oil_xscale_up_rgbx_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, smp_x, newval, hi;

	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();
	smp_x = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		/* push_f for R */
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(s2l_map[in[0]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for G */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(s2l_map[in[1]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for B */
		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(s2l_map[in[2]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for X (always 1.0f) */
		smp_x = (__m128)_mm_srli_si128((__m128i)smp_x, 4);
		newval = _mm_set_ss(1.0f);
		hi = _mm_shuffle_ps(smp_x, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_x = _mm_shuffle_ps(smp_x, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* R dot products for 2 outputs */
			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			/* G dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			/* B dot products for 2 outputs */
			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

			/* X dot products for 2 outputs */
			__m128 px0 = _mm_mul_ps(smp_x, c0);
			__m128 px1 = _mm_mul_ps(smp_x, c1);
			lo = _mm_unpacklo_ps(px0, px1);
			hh = _mm_unpackhi_ps(px0, px1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_x = _mm_add_ps(sum, t1);

			/* Store interleaved: [R0, G0, B0, X0, R1, G1, B1, X1] */
			out[0] = _mm_cvtss_f32(t2_r);
			out[1] = _mm_cvtss_f32(t2_g);
			out[2] = _mm_cvtss_f32(t2_b);
			out[3] = _mm_cvtss_f32(t2_x);
			out[4] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_r, t2_r, _MM_SHUFFLE(1,1,1,1)));
			out[5] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_g, t2_g, _MM_SHUFFLE(1,1,1,1)));
			out[6] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_b, t2_b, _MM_SHUFFLE(1,1,1,1)));
			out[7] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_x, t2_x, _MM_SHUFFLE(1,1,1,1)));

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_x, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[3] = _mm_cvtss_f32(t2);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
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
			__m128 coeffs0 = _mm_load_ps(coeff_buf);
			__m128 coeffs1 = _mm_load_ps(coeff_buf + 4);

			/* First output: broadcast each coeff and multiply */
			__m128 result0 = _mm_add_ps(
				_mm_add_ps(
					_mm_mul_ps(smp0, _mm_shuffle_ps(coeffs0, coeffs0, _MM_SHUFFLE(0,0,0,0))),
					_mm_mul_ps(smp1, _mm_shuffle_ps(coeffs0, coeffs0, _MM_SHUFFLE(1,1,1,1)))),
				_mm_add_ps(
					_mm_mul_ps(smp2, _mm_shuffle_ps(coeffs0, coeffs0, _MM_SHUFFLE(2,2,2,2))),
					_mm_mul_ps(smp3, _mm_shuffle_ps(coeffs0, coeffs0, _MM_SHUFFLE(3,3,3,3)))));

			/* Second output */
			__m128 result1 = _mm_add_ps(
				_mm_add_ps(
					_mm_mul_ps(smp0, _mm_shuffle_ps(coeffs1, coeffs1, _MM_SHUFFLE(0,0,0,0))),
					_mm_mul_ps(smp1, _mm_shuffle_ps(coeffs1, coeffs1, _MM_SHUFFLE(1,1,1,1)))),
				_mm_add_ps(
					_mm_mul_ps(smp2, _mm_shuffle_ps(coeffs1, coeffs1, _MM_SHUFFLE(2,2,2,2))),
					_mm_mul_ps(smp3, _mm_shuffle_ps(coeffs1, coeffs1, _MM_SHUFFLE(3,3,3,3)))));

			_mm_storeu_ps(out, result0);
			_mm_storeu_ps(out + 4, result1);

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);

			__m128 result = _mm_add_ps(
				_mm_add_ps(
					_mm_mul_ps(smp0, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(0,0,0,0))),
					_mm_mul_ps(smp1, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(1,1,1,1)))),
				_mm_add_ps(
					_mm_mul_ps(smp2, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(2,2,2,2))),
					_mm_mul_ps(smp3, _mm_shuffle_ps(coeffs, coeffs, _MM_SHUFFLE(3,3,3,3)))));

			_mm_storeu_ps(out, result);

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
	__m128 v0, v1, v2, v3, sum;
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

		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum2 = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		sum2 = _mm_min_ps(_mm_max_ps(sum2, zero), one);
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

		v0 = _mm_loadu_ps(in[0] + i + 8);
		v1 = _mm_loadu_ps(in[1] + i + 8);
		v2 = _mm_loadu_ps(in[2] + i + 8);
		v3 = _mm_loadu_ps(in[3] + i + 8);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

		v0 = _mm_loadu_ps(in[0] + i + 12);
		v1 = _mm_loadu_ps(in[1] + i + 12);
		v2 = _mm_loadu_ps(in[2] + i + 12);
		v3 = _mm_loadu_ps(in[3] + i + 12);
		sum2 = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
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

		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		sum = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum2 = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
		sum2 = _mm_min_ps(_mm_max_ps(sum2, zero), one);
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

		idx = _mm_packs_epi32(idx, idx2);
		idx = _mm_packus_epi16(idx, idx);
		_mm_storel_epi64((__m128i *)(out + i), idx);
	}

	for (; i+3<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));
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

static void __attribute__((noinline)) oil_scale_down_g_heavy_avx2(
	unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, sample_x, sum, sum2;
	__m128 coeffs_y, sums_y, sample_y;

	coeffs_y = _mm_load_ps(coeffs_y_f);
	sum = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		sum2 = _mm_setzero_ps();
		for (j=0; j+1<border_buf[i]; j+=2) {
			coeffs_x = _mm_load_ps(coeffs_x_f);
			sample_x = _mm_set1_ps(i2f_map[in[0]]);
			sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);

			coeffs_x = _mm_load_ps(coeffs_x_f + 4);
			sample_x = _mm_set1_ps(i2f_map[in[1]]);
			sum2 = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum2);

			in += 2;
			coeffs_x_f += 8;
		}
		for (; j<border_buf[i]; j++) {
			coeffs_x = _mm_load_ps(coeffs_x_f);
			sample_x = _mm_set1_ps(i2f_map[in[0]]);
			sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);
			in += 1;
			coeffs_x_f += 4;
		}
		sum = _mm_add_ps(sum, sum2);

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum = (__m128)_mm_srli_si128(_mm_castps_si128(sum), 4);
	}
}

static void oil_scale_down_g_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, sample_x, sum;
	__m128 coeffs_y, sums_y, sample_y;

	coeffs_y = _mm_load_ps(coeffs_y_f);
	sum = _mm_setzero_ps();

	for (i=0; i<out_width; i++) {
		for (j=0; j<border_buf[i]; j++) {
			coeffs_x = _mm_load_ps(coeffs_x_f);
			sample_x = _mm_set1_ps(i2f_map[in[0]]);
			sum = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum);
			in += 1;
			coeffs_x_f += 4;
		}

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum = (__m128)_mm_srli_si128(_mm_castps_si128(sum), 4);
	}
}

static void oil_scale_down_ga_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	float alpha;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_g, sum_a;
	__m128 sum_g2, sum_a2;
	__m128 coeffs_y, sums_y, sample_y;

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

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_g, sum_g, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_a, sum_a, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_a = (__m128)_mm_srli_si128(_mm_castps_si128(sum_a), 4);
	}
}

static void oil_scale_down_rgb_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m128 coeffs_y, sums_y, sample_y;

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

				sample_x = _mm_set1_ps(s2l_map[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(s2l_map[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(s2l_map[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				sample_x = _mm_set1_ps(s2l_map[in[3]]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_r2);

				sample_x = _mm_set1_ps(s2l_map[in[4]]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_g2);

				sample_x = _mm_set1_ps(s2l_map[in[5]]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_b2);

				in += 6;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(s2l_map[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(s2l_map[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(s2l_map[in[2]]);
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

				sample_x = _mm_set1_ps(s2l_map[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(s2l_map[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(s2l_map[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				in += 3;
				coeffs_x_f += 4;
			}
		}

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_r, sum_r, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_g, sum_g, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_b, sum_b, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
	}
}

static void oil_yscale_out_rgba_avx2(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	__m128 scale, one, zero;
	__m128 vals, alpha_v;
	__m128i idx, z;
	float alpha;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i<width; i++) {
		/* Read only the current tap */
		vals = _mm_load_ps(sums + tap_off);

		/* Clamp alpha to [0, 1] */
		alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
		}

		/* Clamp RGB to [0, 1] and compute l2s_map indices */
		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

		out[0] = lut[_mm_cvtsi128_si32(idx)];
		out[1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[3] = (int)(alpha * 255.0f + 0.5f);

		/* Zero consumed tap */
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_up_rgba_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 v0, v1, v2, v3, sum;
	__m128 scale, one, zero;
	__m128 alpha_v, clamped;
	__m128i idx;
	unsigned char *lut;
	float alpha;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();

	for (i=0; i<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		/* Clamp alpha to [0, 1] */
		alpha_v = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			sum = _mm_mul_ps(sum, _mm_rcp_ps(alpha_v));
		}

		/* Clamp to [0, 1] and compute l2s_map indices */
		clamped = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_mul_ps(clamped, scale));

		out[i]   = lut[_mm_cvtsi128_si32(idx)];
		out[i+1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[i+3] = (int)(alpha * 255.0f + 0.5f);
	}
}

static void oil_xscale_up_rgba_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, smp_a, newval, hi;
	float *sl;

	sl = s2l_map;
	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();
	smp_a = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		float alpha_new = i2f_map[in[3]];

		/* push_f for A */
		smp_a = (__m128)_mm_srli_si128((__m128i)smp_a, 4);
		newval = _mm_set_ss(alpha_new);
		hi = _mm_shuffle_ps(smp_a, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_a = _mm_shuffle_ps(smp_a, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for R: premultiplied by alpha */
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(alpha_new * sl[in[0]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for G: premultiplied by alpha */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(alpha_new * sl[in[1]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for B: premultiplied by alpha */
		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(alpha_new * sl[in[2]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* R dot products for 2 outputs */
			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			/* G dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			/* B dot products for 2 outputs */
			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

			/* A dot products for 2 outputs */
			__m128 pa0 = _mm_mul_ps(smp_a, c0);
			__m128 pa1 = _mm_mul_ps(smp_a, c1);
			lo = _mm_unpacklo_ps(pa0, pa1);
			hh = _mm_unpackhi_ps(pa0, pa1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_a = _mm_add_ps(sum, t1);

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

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_a, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[3] = _mm_cvtss_f32(t2);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_scale_down_rgba_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	__m128 coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	__m128 sum_r, sum_g, sum_b, sum_a;
	__m128 sum_r2, sum_g2, sum_b2, sum_a2;
	__m128 cy0, cy1, cy2, cy3;
	float *sl;

	sl = s2l_map;
	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = _mm_set1_ps(coeffs_y_f[0]);
	cy1 = _mm_set1_ps(coeffs_y_f[1]);
	cy2 = _mm_set1_ps(coeffs_y_f[2]);
	cy3 = _mm_set1_ps(coeffs_y_f[3]);

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

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[px0 >> 24]));

				sample_x = _mm_set1_ps(sl[px0 & 0xFF]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[(px0 >> 8) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[(px0 >> 16) & 0xFF]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				coeffs_x2_a = _mm_mul_ps(coeffs_x2, _mm_set1_ps(i2f_map[px1 >> 24]));

				sample_x = _mm_set1_ps(sl[px1 & 0xFF]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_r2);

				sample_x = _mm_set1_ps(sl[(px1 >> 8) & 0xFF]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_g2);

				sample_x = _mm_set1_ps(sl[(px1 >> 16) & 0xFF]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = _mm_add_ps(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[3]]));

				sample_x = _mm_set1_ps(sl[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[in[2]]);
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

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[3]]));

				sample_x = _mm_set1_ps(sl[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			__m128 rg, ba, rgba, sy;

			rg = _mm_unpacklo_ps(sum_r, sum_g);
			ba = _mm_unpacklo_ps(sum_b, sum_a);
			rgba = _mm_movelh_ps(rg, ba);

			sy = _mm_load_ps(sums_y_out + off0);
			sy = _mm_add_ps(_mm_mul_ps(cy0, rgba), sy);
			_mm_store_ps(sums_y_out + off0, sy);

			sy = _mm_load_ps(sums_y_out + off1);
			sy = _mm_add_ps(_mm_mul_ps(cy1, rgba), sy);
			_mm_store_ps(sums_y_out + off1, sy);

			sy = _mm_load_ps(sums_y_out + off2);
			sy = _mm_add_ps(_mm_mul_ps(cy2, rgba), sy);
			_mm_store_ps(sums_y_out + off2, sy);

			sy = _mm_load_ps(sums_y_out + off3);
			sy = _mm_add_ps(_mm_mul_ps(cy3, rgba), sy);
			_mm_store_ps(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
		sum_a = (__m128)_mm_srli_si128(_mm_castps_si128(sum_a), 4);
	}
}

static void oil_yscale_out_argb_avx2(float *sums, int width, unsigned char *out,
	int tap)
{
	int i, tap_off;
	__m128 scale, one, zero;
	__m128 vals, alpha_v;
	__m128i idx, z;
	float alpha;
	unsigned char *lut;

	lut = l2s_map;
	tap_off = tap * 4;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i<width; i++) {
		/* Read only the current tap */
		vals = _mm_load_ps(sums + tap_off);

		/* Clamp alpha to [0, 1] */
		alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
		}

		/* Clamp RGB to [0, 1] and compute l2s_map indices */
		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

		out[0] = (int)(alpha * 255.0f + 0.5f);
		out[1] = lut[_mm_cvtsi128_si32(idx)];
		out[2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[3] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];

		/* Zero consumed tap */
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		sums += 16;
		out += 4;
	}
}

static void oil_yscale_up_argb_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 v0, v1, v2, v3, sum;
	__m128 scale, one, zero;
	__m128 alpha_v, clamped;
	__m128i idx;
	unsigned char *lut;
	float alpha;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();

	for (i=0; i<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		/* Clamp alpha to [0, 1] */
		alpha_v = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);

		/* Divide RGB by alpha (skip if alpha == 0) */
		if (alpha != 0) {
			sum = _mm_mul_ps(sum, _mm_rcp_ps(alpha_v));
		}

		/* Clamp to [0, 1] and compute l2s_map indices */
		clamped = _mm_min_ps(_mm_max_ps(sum, zero), one);
		idx = _mm_cvttps_epi32(_mm_mul_ps(clamped, scale));

		out[i]   = (int)(alpha * 255.0f + 0.5f);
		out[i+1] = lut[_mm_cvtsi128_si32(idx)];
		out[i+2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[i+3] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
	}
}

static void oil_xscale_up_argb_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, smp_a, newval, hi;
	float *sl;

	sl = s2l_map;
	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();
	smp_a = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		float alpha_new = i2f_map[in[0]];

		/* push_f for A */
		smp_a = (__m128)_mm_srli_si128((__m128i)smp_a, 4);
		newval = _mm_set_ss(alpha_new);
		hi = _mm_shuffle_ps(smp_a, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_a = _mm_shuffle_ps(smp_a, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for R: premultiplied by alpha */
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(alpha_new * sl[in[1]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for G: premultiplied by alpha */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(alpha_new * sl[in[2]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for B: premultiplied by alpha */
		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(alpha_new * sl[in[3]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* R dot products for 2 outputs */
			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			/* G dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			/* B dot products for 2 outputs */
			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

			/* A dot products for 2 outputs */
			__m128 pa0 = _mm_mul_ps(smp_a, c0);
			__m128 pa1 = _mm_mul_ps(smp_a, c1);
			lo = _mm_unpacklo_ps(pa0, pa1);
			hh = _mm_unpackhi_ps(pa0, pa1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_a = _mm_add_ps(sum, t1);

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

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_a, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[3] = _mm_cvtss_f32(t2);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_scale_down_argb_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	__m128 coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	__m128 sum_r, sum_g, sum_b, sum_a;
	__m128 sum_r2, sum_g2, sum_b2, sum_a2;
	__m128 cy0, cy1, cy2, cy3;
	float *sl;

	sl = s2l_map;
	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = _mm_set1_ps(coeffs_y_f[0]);
	cy1 = _mm_set1_ps(coeffs_y_f[1]);
	cy2 = _mm_set1_ps(coeffs_y_f[2]);
	cy3 = _mm_set1_ps(coeffs_y_f[3]);

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

				/* ARGB little-endian: px = B<<24|G<<16|R<<8|A */
				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[px0 & 0xFF]));

				sample_x = _mm_set1_ps(sl[(px0 >> 8) & 0xFF]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[(px0 >> 16) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[px0 >> 24]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				coeffs_x2_a = _mm_mul_ps(coeffs_x2, _mm_set1_ps(i2f_map[px1 & 0xFF]));

				sample_x = _mm_set1_ps(sl[(px1 >> 8) & 0xFF]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_r2);

				sample_x = _mm_set1_ps(sl[(px1 >> 16) & 0xFF]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_g2);

				sample_x = _mm_set1_ps(sl[px1 >> 24]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = _mm_add_ps(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[0]]));

				sample_x = _mm_set1_ps(sl[in[1]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[in[2]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[in[3]]);
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

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[0]]));

				sample_x = _mm_set1_ps(sl[in[1]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[in[2]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[in[3]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			__m128 rg, ba, rgba, sy;

			rg = _mm_unpacklo_ps(sum_r, sum_g);
			ba = _mm_unpacklo_ps(sum_b, sum_a);
			rgba = _mm_movelh_ps(rg, ba);

			sy = _mm_load_ps(sums_y_out + off0);
			sy = _mm_add_ps(_mm_mul_ps(cy0, rgba), sy);
			_mm_store_ps(sums_y_out + off0, sy);

			sy = _mm_load_ps(sums_y_out + off1);
			sy = _mm_add_ps(_mm_mul_ps(cy1, rgba), sy);
			_mm_store_ps(sums_y_out + off1, sy);

			sy = _mm_load_ps(sums_y_out + off2);
			sy = _mm_add_ps(_mm_mul_ps(cy2, rgba), sy);
			_mm_store_ps(sums_y_out + off2, sy);

			sy = _mm_load_ps(sums_y_out + off3);
			sy = _mm_add_ps(_mm_mul_ps(cy3, rgba), sy);
			_mm_store_ps(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
		sum_a = (__m128)_mm_srli_si128(_mm_castps_si128(sum_a), 4);
	}
}

static void oil_yscale_out_cmyk_avx2(float *sums, int len, unsigned char *out)
{
	int i;
	__m128 scale, vals, ab, cd, f0, f1, f2, f3;
	__m128i idx, clamped, v0, v1, v2, v3;

	scale = _mm_set1_ps(255.0f);

	for (i=0; i+3<len; i+=4) {
		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));
		v2 = _mm_load_si128((__m128i *)(sums + 8));
		v3 = _mm_load_si128((__m128i *)(sums + 12));

		f0 = _mm_castsi128_ps(v0);
		f1 = _mm_castsi128_ps(v1);
		f2 = _mm_castsi128_ps(v2);
		f3 = _mm_castsi128_ps(v3);
		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		/* clamp to [0, 1] then scale to [0, 255] */
		vals = _mm_min_ps(_mm_max_ps(vals, _mm_setzero_ps()), _mm_set1_ps(1.0f));
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), _mm_set1_ps(0.5f)));

		/* Pack 32-bit ints to 16-bit then to 8-bit */
		clamped = _mm_packs_epi32(idx, idx);
		clamped = _mm_packus_epi16(clamped, clamped);

		*(int *)&out[i] = _mm_cvtsi128_si32(clamped);

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));
		_mm_store_si128((__m128i *)(sums + 8), _mm_srli_si128(v2, 4));
		_mm_store_si128((__m128i *)(sums + 12), _mm_srli_si128(v3, 4));

		sums += 16;
	}

	for (; i<len; i++) {
		float v = *sums;
		if (v < 0.0f) v = 0.0f;
		if (v > 1.0f) v = 1.0f;
		out[i] = (int)(v * 255.0f + 0.5f);
		oil_shift_left_f_avx2(sums);
		sums += 4;
	}
}

static void oil_scale_down_cmyk_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_c, sum_m, sum_y, sum_k;
	__m128 sum_c2, sum_m2, sum_y2, sum_k2;
	__m128 coeffs_y, sums_y, sample_y;

	coeffs_y = _mm_load_ps(coeffs_y_f);

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

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_c, sum_c, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_m, sum_m, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_y, sum_y, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_k, sum_k, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_c = (__m128)_mm_srli_si128(_mm_castps_si128(sum_c), 4);
		sum_m = (__m128)_mm_srli_si128(_mm_castps_si128(sum_m), 4);
		sum_y = (__m128)_mm_srli_si128(_mm_castps_si128(sum_y), 4);
		sum_k = (__m128)_mm_srli_si128(_mm_castps_si128(sum_k), 4);
	}
}

static void oil_scale_down_rgbx_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m128 cy0, cy1, cy2, cy3;
	float *lut;

	lut = s2l_map;
	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = _mm_set1_ps(coeffs_y_f[0]);
	cy1 = _mm_set1_ps(coeffs_y_f[1]);
	cy2 = _mm_set1_ps(coeffs_y_f[2]);
	cy3 = _mm_set1_ps(coeffs_y_f[3]);

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
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				sample_x = _mm_set1_ps(lut[px1 & 0xFF]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_r2);

				sample_x = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_g2);

				sample_x = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_b2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(lut[px & 0xFF]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

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
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			__m128 rg, bx, rgbx, sy;

			rg = _mm_unpacklo_ps(sum_r, sum_g);
			bx = _mm_unpacklo_ps(sum_b, sum_b);
			rgbx = _mm_movelh_ps(rg, bx);

			sy = _mm_load_ps(sums_y_out + off0);
			sy = _mm_add_ps(_mm_mul_ps(cy0, rgbx), sy);
			_mm_store_ps(sums_y_out + off0, sy);

			sy = _mm_load_ps(sums_y_out + off1);
			sy = _mm_add_ps(_mm_mul_ps(cy1, rgbx), sy);
			_mm_store_ps(sums_y_out + off1, sy);

			sy = _mm_load_ps(sums_y_out + off2);
			sy = _mm_add_ps(_mm_mul_ps(cy2, rgbx), sy);
			_mm_store_ps(sums_y_out + off2, sy);

			sy = _mm_load_ps(sums_y_out + off3);
			sy = _mm_add_ps(_mm_mul_ps(cy3, rgbx), sy);
			_mm_store_ps(sums_y_out + off3, sy);

			sums_y_out += 16;
		}

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
	}
}

static void oil_xscale_up_rgb_nogamma_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, newval, hi;

	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(i2f_map[in[0]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(i2f_map[in[1]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(i2f_map[in[2]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

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

		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			out += 3;
			coeff_buf += 4;
		}

		in += 3;
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

static void oil_scale_down_rgbx_nogamma_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m256 cy_lo, cy_hi;
	float *lut;

	lut = i2f_map;

	/* Precompute 256-bit coefficient vectors ordered by physical slot */
	{
		float cy_slot[4];
		int k;
		for (k = 0; k < 4; k++)
			cy_slot[k] = coeffs_y_f[(k - tap + 4) & 3];
		cy_lo = _mm256_set_m128(
			_mm_set1_ps(cy_slot[1]),
			_mm_set1_ps(cy_slot[0]));
		cy_hi = _mm256_set_m128(
			_mm_set1_ps(cy_slot[3]),
			_mm_set1_ps(cy_slot[2]));
	}

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

		/* Vertical accumulation using 256-bit AVX2 */
		{
			__m128 rg, bx, rgbx;
			__m256 rgbx256, sy;

			/* Prefetch next pixel's sums_y */
			_mm_prefetch((const char *)(sums_y_out + 16), _MM_HINT_T0);

			rg = _mm_unpacklo_ps(sum_r, sum_g);
			bx = _mm_unpacklo_ps(sum_b, sum_b);
			rgbx = _mm_movelh_ps(rg, bx);

			rgbx256 = _mm256_set_m128(rgbx, rgbx);

			sy = _mm256_loadu_ps(sums_y_out);
			sy = _mm256_fmadd_ps(cy_lo, rgbx256, sy);
			_mm256_storeu_ps(sums_y_out, sy);

			sy = _mm256_loadu_ps(sums_y_out + 8);
			sy = _mm256_fmadd_ps(cy_hi, rgbx256, sy);
			_mm256_storeu_ps(sums_y_out + 8, sy);

			sums_y_out += 16;
		}

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
	}
}

static void oil_scale_down_rgb_nogamma_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m128 coeffs_y, sums_y, sample_y;

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

				sample_x = _mm_set1_ps(i2f_map[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(i2f_map[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(i2f_map[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				sample_x = _mm_set1_ps(i2f_map[in[3]]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_r2);

				sample_x = _mm_set1_ps(i2f_map[in[4]]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_g2);

				sample_x = _mm_set1_ps(i2f_map[in[5]]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_b2);

				in += 6;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(i2f_map[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(i2f_map[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(i2f_map[in[2]]);
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

				sample_x = _mm_set1_ps(i2f_map[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_r);

				sample_x = _mm_set1_ps(i2f_map[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_g);

				sample_x = _mm_set1_ps(i2f_map[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_b);

				in += 3;
				coeffs_x_f += 4;
			}
		}

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_r, sum_r, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_g, sum_g, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_b, sum_b, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
	}
}

static void oil_yscale_out_rgba_nogamma_avx2(float *sums, int width,
	unsigned char *out, int tap)
{
	int i, tap_off;
	__m128 scale, half, one, zero;
	__m128 vals, alpha_v;
	__m128i idx, packed;
	__m128i z;

	tap_off = tap * 4;
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i+3<width; i+=4) {
		__m128i idx2, idx3, idx4;
		__m128 vals2, vals3, vals4, alpha_v2, alpha_v3, alpha_v4;
		__m128i packed2;

		/* Pixel 1 */
		vals = _mm_load_ps(sums + tap_off);
		alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		if (_mm_cvtss_f32(alpha_v) != 0)
			vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		{
			__m128 hi = _mm_shuffle_ps(vals, alpha_v, _MM_SHUFFLE(0, 0, 2, 2));
			vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
		}
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		/* Pixel 2 */
		vals2 = _mm_load_ps(sums + 16 + tap_off);
		alpha_v2 = _mm_shuffle_ps(vals2, vals2, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v2 = _mm_min_ps(_mm_max_ps(alpha_v2, zero), one);
		if (_mm_cvtss_f32(alpha_v2) != 0)
			vals2 = _mm_mul_ps(vals2, _mm_rcp_ps(alpha_v2));
		vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
		{
			__m128 hi2 = _mm_shuffle_ps(vals2, alpha_v2, _MM_SHUFFLE(0, 0, 2, 2));
			vals2 = _mm_shuffle_ps(vals2, hi2, _MM_SHUFFLE(2, 0, 1, 0));
		}
		idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));
		_mm_store_si128((__m128i *)(sums + 16 + tap_off), z);

		packed = _mm_packs_epi32(idx, idx2);

		/* Pixel 3 */
		vals3 = _mm_load_ps(sums + 32 + tap_off);
		alpha_v3 = _mm_shuffle_ps(vals3, vals3, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v3 = _mm_min_ps(_mm_max_ps(alpha_v3, zero), one);
		if (_mm_cvtss_f32(alpha_v3) != 0)
			vals3 = _mm_mul_ps(vals3, _mm_rcp_ps(alpha_v3));
		vals3 = _mm_min_ps(_mm_max_ps(vals3, zero), one);
		{
			__m128 hi3 = _mm_shuffle_ps(vals3, alpha_v3, _MM_SHUFFLE(0, 0, 2, 2));
			vals3 = _mm_shuffle_ps(vals3, hi3, _MM_SHUFFLE(2, 0, 1, 0));
		}
		idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals3, scale), half));
		_mm_store_si128((__m128i *)(sums + 32 + tap_off), z);

		/* Pixel 4 */
		vals4 = _mm_load_ps(sums + 48 + tap_off);
		alpha_v4 = _mm_shuffle_ps(vals4, vals4, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v4 = _mm_min_ps(_mm_max_ps(alpha_v4, zero), one);
		if (_mm_cvtss_f32(alpha_v4) != 0)
			vals4 = _mm_mul_ps(vals4, _mm_rcp_ps(alpha_v4));
		vals4 = _mm_min_ps(_mm_max_ps(vals4, zero), one);
		{
			__m128 hi4 = _mm_shuffle_ps(vals4, alpha_v4, _MM_SHUFFLE(0, 0, 2, 2));
			vals4 = _mm_shuffle_ps(vals4, hi4, _MM_SHUFFLE(2, 0, 1, 0));
		}
		idx4 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals4, scale), half));
		_mm_store_si128((__m128i *)(sums + 48 + tap_off), z);

		packed2 = _mm_packs_epi32(idx3, idx4);
		packed = _mm_packus_epi16(packed, packed2);
		_mm_storeu_si128((__m128i *)out, packed);

		sums += 64;
		out += 16;
	}

	for (; i<width; i++) {
		vals = _mm_load_ps(sums + tap_off);

		alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		if (_mm_cvtss_f32(alpha_v) != 0)
			vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		{
			__m128 hi = _mm_shuffle_ps(vals, alpha_v,
				_MM_SHUFFLE(0, 0, 2, 2));
			vals = _mm_shuffle_ps(vals, hi,
				_MM_SHUFFLE(2, 0, 1, 0));
		}
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
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
	__m128 v0, v1, v2, v3, sum_a, sum_b;
	__m128 scale, half, one, zero;
	__m128 alpha_v, clamped;
	__m128i idx_a, idx_b, packed;
	float alpha;

	c0 = _mm_set1_ps(coeffs[0]);
	c1 = _mm_set1_ps(coeffs[1]);
	c2 = _mm_set1_ps(coeffs[2]);
	c3 = _mm_set1_ps(coeffs[3]);
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();

	for (i=0; i+7<len; i+=8) {
		/* Pixel 1 */
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum_a = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		/* Pixel 2 */
		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum_b = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		/* Unpremultiply pixel 1 */
		alpha_v = _mm_shuffle_ps(sum_a, sum_a, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);
		if (alpha != 0) {
			sum_a = _mm_mul_ps(sum_a, _mm_rcp_ps(alpha_v));
		}
		clamped = _mm_min_ps(_mm_max_ps(sum_a, zero), one);
		{
			__m128 hi = _mm_shuffle_ps(clamped, alpha_v,
				_MM_SHUFFLE(0, 0, 2, 2));
			clamped = _mm_shuffle_ps(clamped, hi,
				_MM_SHUFFLE(2, 0, 1, 0));
		}
		idx_a = _mm_cvttps_epi32(_mm_add_ps(
			_mm_mul_ps(clamped, scale), half));

		/* Unpremultiply pixel 2 */
		alpha_v = _mm_shuffle_ps(sum_b, sum_b, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);
		if (alpha != 0) {
			sum_b = _mm_mul_ps(sum_b, _mm_rcp_ps(alpha_v));
		}
		clamped = _mm_min_ps(_mm_max_ps(sum_b, zero), one);
		{
			__m128 hi = _mm_shuffle_ps(clamped, alpha_v,
				_MM_SHUFFLE(0, 0, 2, 2));
			clamped = _mm_shuffle_ps(clamped, hi,
				_MM_SHUFFLE(2, 0, 1, 0));
		}
		idx_b = _mm_cvttps_epi32(_mm_add_ps(
			_mm_mul_ps(clamped, scale), half));

		/* Pack both pixels to bytes and store 8 bytes */
		packed = _mm_packs_epi32(idx_a, idx_b);
		packed = _mm_packus_epi16(packed, packed);
		_mm_storel_epi64((__m128i *)(out + i), packed);
	}

	for (; i<len; i+=4) {
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum_a = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		alpha_v = _mm_shuffle_ps(sum_a, sum_a, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);
		if (alpha != 0) {
			sum_a = _mm_mul_ps(sum_a, _mm_rcp_ps(alpha_v));
		}
		clamped = _mm_min_ps(_mm_max_ps(sum_a, zero), one);
		{
			__m128 hi = _mm_shuffle_ps(clamped, alpha_v,
				_MM_SHUFFLE(0, 0, 2, 2));
			clamped = _mm_shuffle_ps(clamped, hi,
				_MM_SHUFFLE(2, 0, 1, 0));
		}
		idx_a = _mm_cvttps_epi32(_mm_add_ps(
			_mm_mul_ps(clamped, scale), half));
		packed = _mm_packs_epi32(idx_a, idx_a);
		packed = _mm_packus_epi16(packed, packed);
		*(int *)(out + i) = _mm_cvtsi128_si32(packed);
	}
}

static void oil_xscale_up_rgba_nogamma_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, smp_a, newval, hi;
	float *lut;

	lut = i2f_map;
	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();
	smp_a = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		float alpha_new = lut[in[3]];

		/* push_f for A */
		smp_a = (__m128)_mm_srli_si128((__m128i)smp_a, 4);
		newval = _mm_set_ss(alpha_new);
		hi = _mm_shuffle_ps(smp_a, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_a = _mm_shuffle_ps(smp_a, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for R: premultiplied by alpha */
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(alpha_new * lut[in[0]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for G: premultiplied by alpha */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(alpha_new * lut[in[1]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for B: premultiplied by alpha */
		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(alpha_new * lut[in[2]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* R dot products for 2 outputs */
			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			/* G dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			/* B dot products for 2 outputs */
			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

			/* A dot products for 2 outputs */
			__m128 pa0 = _mm_mul_ps(smp_a, c0);
			__m128 pa1 = _mm_mul_ps(smp_a, c1);
			lo = _mm_unpacklo_ps(pa0, pa1);
			hh = _mm_unpackhi_ps(pa0, pa1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_a = _mm_add_ps(sum, t1);

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

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_a, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[3] = _mm_cvtss_f32(t2);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
	}
}

static void oil_scale_down_rgba_nogamma_avx2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	__m128 sum_r, sum_g, sum_b, sum_a;
	__m128 sum_r2, sum_g2, sum_b2, sum_a2;
	float *lut;
	__m256 cy256_lo, cy256_hi;
	{
		float cy_phys[4];
		cy_phys[tap & 3] = coeffs_y_f[0];
		cy_phys[(tap + 1) & 3] = coeffs_y_f[1];
		cy_phys[(tap + 2) & 3] = coeffs_y_f[2];
		cy_phys[(tap + 3) & 3] = coeffs_y_f[3];
		cy256_lo = _mm256_set_m128(
			_mm_set1_ps(cy_phys[1]),
			_mm_set1_ps(cy_phys[0]));
		cy256_hi = _mm256_set_m128(
			_mm_set1_ps(cy_phys[3]),
			_mm_set1_ps(cy_phys[2]));
	}

	lut = i2f_map;

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

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(lut[px0 >> 24]));

				sample_x = _mm_set1_ps(lut[px0 & 0xFF]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				coeffs_x2_a = _mm_mul_ps(coeffs_x2, _mm_set1_ps(lut[px1 >> 24]));

				sample_x = _mm_set1_ps(lut[px1 & 0xFF]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_r2);

				sample_x = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_g2);

				sample_x = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_b2);

				sum_a2 = _mm_add_ps(coeffs_x2_a, sum_a2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				unsigned int px;
				memcpy(&px, in, 4);

				coeffs_x = _mm_load_ps(coeffs_x_f);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(lut[px >> 24]));

				sample_x = _mm_set1_ps(lut[px & 0xFF]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
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

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(lut[in[3]]));

				sample_x = _mm_set1_ps(lut[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(lut[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(lut[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				in += 4;
				coeffs_x_f += 4;
			}
		}

		/* Vertical accumulation using ring buffer offsets */
		{
			__m128 rg, ba, rgba;

			rg = _mm_unpacklo_ps(sum_r, sum_g);
			ba = _mm_unpacklo_ps(sum_b, sum_a);
			rgba = _mm_movelh_ps(rg, ba);

			{
				__m256 rgba256, sy_lo, sy_hi;
				rgba256 = _mm256_set_m128(rgba, rgba);
				sy_lo = _mm256_loadu_ps(sums_y_out);
				sy_hi = _mm256_loadu_ps(sums_y_out + 8);
				sy_lo = _mm256_fmadd_ps(cy256_lo, rgba256, sy_lo);
				sy_hi = _mm256_fmadd_ps(cy256_hi, rgba256, sy_hi);
				_mm256_storeu_ps(sums_y_out, sy_lo);
				_mm256_storeu_ps(sums_y_out + 8, sy_hi);
			}
			sums_y_out += 16;
		}

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
		sum_a = (__m128)_mm_srli_si128(_mm_castps_si128(sum_a), 4);
	}
}

static void oil_yscale_up_rgbx_nogamma_avx2(float **in, int len, float *coeffs,
	unsigned char *out)
{
	int i;
	__m128 c0, c1, c2, c3;
	__m128 v0, v1, v2, v3, sum_a, sum_b;
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
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum_a = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		/* Pixel 2 */
		v0 = _mm_loadu_ps(in[0] + i + 4);
		v1 = _mm_loadu_ps(in[1] + i + 4);
		v2 = _mm_loadu_ps(in[2] + i + 4);
		v3 = _mm_loadu_ps(in[3] + i + 4);
		sum_b = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

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
		v0 = _mm_loadu_ps(in[0] + i);
		v1 = _mm_loadu_ps(in[1] + i);
		v2 = _mm_loadu_ps(in[2] + i);
		v3 = _mm_loadu_ps(in[3] + i);
		sum_a = _mm_add_ps(
			_mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
			_mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)));

		sum_a = _mm_min_ps(_mm_max_ps(sum_a, zero), one);
		idx_a = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum_a, scale), half));
		idx_a = _mm_or_si128(_mm_and_si128(idx_a, mask), x_val);
		packed = _mm_packs_epi32(idx_a, idx_a);
		packed = _mm_packus_epi16(packed, packed);
		*(int *)(out + i) = _mm_cvtsi128_si32(packed);
	}
}

static void oil_xscale_up_rgbx_nogamma_avx2(unsigned char *in, int width_in, float *out,
	float *coeff_buf, int *border_buf)
{
	int i, j;
	__m128 smp_r, smp_g, smp_b, smp_x, newval, hi;
	float *lut;

	lut = i2f_map;
	smp_r = _mm_setzero_ps();
	smp_g = _mm_setzero_ps();
	smp_b = _mm_setzero_ps();
	smp_x = _mm_setzero_ps();

	for (i=0; i<width_in; i++) {
		/* push_f for R */
		smp_r = (__m128)_mm_srli_si128((__m128i)smp_r, 4);
		newval = _mm_set_ss(lut[in[0]]);
		hi = _mm_shuffle_ps(smp_r, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_r = _mm_shuffle_ps(smp_r, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for G */
		smp_g = (__m128)_mm_srli_si128((__m128i)smp_g, 4);
		newval = _mm_set_ss(lut[in[1]]);
		hi = _mm_shuffle_ps(smp_g, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_g = _mm_shuffle_ps(smp_g, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for B */
		smp_b = (__m128)_mm_srli_si128((__m128i)smp_b, 4);
		newval = _mm_set_ss(lut[in[2]]);
		hi = _mm_shuffle_ps(smp_b, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_b = _mm_shuffle_ps(smp_b, hi, _MM_SHUFFLE(2, 0, 1, 0));

		/* push_f for X (always 1.0f) */
		smp_x = (__m128)_mm_srli_si128((__m128i)smp_x, 4);
		newval = _mm_set_ss(1.0f);
		hi = _mm_shuffle_ps(smp_x, newval, _MM_SHUFFLE(0, 0, 3, 2));
		smp_x = _mm_shuffle_ps(smp_x, hi, _MM_SHUFFLE(2, 0, 1, 0));

		j = border_buf[i];

		/* process pairs of outputs */
		while (j >= 2) {
			__m128 c0 = _mm_load_ps(coeff_buf);
			__m128 c1 = _mm_load_ps(coeff_buf + 4);

			/* R dot products for 2 outputs */
			__m128 pr0 = _mm_mul_ps(smp_r, c0);
			__m128 pr1 = _mm_mul_ps(smp_r, c1);
			__m128 lo = _mm_unpacklo_ps(pr0, pr1);
			__m128 hh = _mm_unpackhi_ps(pr0, pr1);
			__m128 sum = _mm_add_ps(lo, hh);
			__m128 t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_r = _mm_add_ps(sum, t1);

			/* G dot products for 2 outputs */
			__m128 pg0 = _mm_mul_ps(smp_g, c0);
			__m128 pg1 = _mm_mul_ps(smp_g, c1);
			lo = _mm_unpacklo_ps(pg0, pg1);
			hh = _mm_unpackhi_ps(pg0, pg1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_g = _mm_add_ps(sum, t1);

			/* B dot products for 2 outputs */
			__m128 pb0 = _mm_mul_ps(smp_b, c0);
			__m128 pb1 = _mm_mul_ps(smp_b, c1);
			lo = _mm_unpacklo_ps(pb0, pb1);
			hh = _mm_unpackhi_ps(pb0, pb1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_b = _mm_add_ps(sum, t1);

			/* X dot products for 2 outputs */
			__m128 px0 = _mm_mul_ps(smp_x, c0);
			__m128 px1 = _mm_mul_ps(smp_x, c1);
			lo = _mm_unpacklo_ps(px0, px1);
			hh = _mm_unpackhi_ps(px0, px1);
			sum = _mm_add_ps(lo, hh);
			t1 = _mm_movehl_ps(sum, sum);
			__m128 t2_x = _mm_add_ps(sum, t1);

			/* Store interleaved: [R0, G0, B0, X0, R1, G1, B1, X1] */
			out[0] = _mm_cvtss_f32(t2_r);
			out[1] = _mm_cvtss_f32(t2_g);
			out[2] = _mm_cvtss_f32(t2_b);
			out[3] = _mm_cvtss_f32(t2_x);
			out[4] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_r, t2_r, _MM_SHUFFLE(1,1,1,1)));
			out[5] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_g, t2_g, _MM_SHUFFLE(1,1,1,1)));
			out[6] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_b, t2_b, _MM_SHUFFLE(1,1,1,1)));
			out[7] = _mm_cvtss_f32(
				_mm_shuffle_ps(t2_x, t2_x, _MM_SHUFFLE(1,1,1,1)));

			out += 8;
			coeff_buf += 8;
			j -= 2;
		}

		/* process remaining single output */
		if (j) {
			__m128 coeffs = _mm_load_ps(coeff_buf);

			__m128 prod = _mm_mul_ps(smp_r, coeffs);
			__m128 t1 = _mm_movehl_ps(prod, prod);
			__m128 t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[0] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_g, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[1] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_b, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[2] = _mm_cvtss_f32(t2);

			prod = _mm_mul_ps(smp_x, coeffs);
			t1 = _mm_movehl_ps(prod, prod);
			t2 = _mm_add_ps(prod, t1);
			prod = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1,1,1,1));
			t2 = _mm_add_ss(t2, prod);
			out[3] = _mm_cvtss_f32(t2);

			out += 4;
			coeff_buf += 4;
		}

		in += 4;
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
		oil_yscale_out_cmyk_avx2(sums, sl_len, out);
		break;
	case OIL_CS_GA:
		oil_yscale_out_ga_avx2(sums, width, out);
		break;
	case OIL_CS_RGB:
		oil_yscale_out_linear_avx2(sums, sl_len, out);
		break;
	case OIL_CS_RGBA:
		oil_yscale_out_rgba_avx2(sums, width, out, tap);
		break;
	case OIL_CS_ARGB:
		oil_yscale_out_argb_avx2(sums, width, out, tap);
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
		oil_yscale_up_rgba_avx2(in, len, coeffs, out);
		break;
	case OIL_CS_ARGB:
		oil_yscale_up_argb_avx2(in, len, coeffs, out);
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
		oil_xscale_up_rgb_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_G:
		oil_xscale_up_g_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_CMYK:
		oil_xscale_up_cmyk_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA:
		oil_xscale_up_rgba_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_GA:
		oil_xscale_up_ga_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_ARGB:
		oil_xscale_up_argb_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBX:
		oil_xscale_up_rgbx_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_xscale_up_rgb_nogamma_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_xscale_up_rgba_nogamma_avx2(in, width_in, out, coeff_buf, border_buf);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_xscale_up_rgbx_nogamma_avx2(in, width_in, out, coeff_buf, border_buf);
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
		oil_scale_down_rgb_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_G:
		if (os->in_width >= os->out_width * 2) {
			oil_scale_down_g_heavy_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		} else {
			oil_scale_down_g_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		}
		break;
	case OIL_CS_CMYK:
		oil_scale_down_cmyk_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_RGBA:
		oil_scale_down_rgba_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_GA:
		oil_scale_down_ga_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_ARGB:
		oil_scale_down_argb_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX:
		oil_scale_down_rgbx_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGB_NOGAMMA:
		oil_scale_down_rgb_nogamma_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y);
		break;
	case OIL_CS_RGBA_NOGAMMA:
		oil_scale_down_rgba_nogamma_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_scale_down_rgbx_nogamma_avx2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
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


