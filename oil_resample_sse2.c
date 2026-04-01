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
#include <immintrin.h>

void oil_shift_left_f_sse2(float *f)
{
	__m128i v = _mm_load_si128((__m128i *)f);
	_mm_store_si128((__m128i *)f, _mm_srli_si128(v, 4));
}

void oil_yscale_out_nonlinear_sse2(float *sums, int len, unsigned char *out)
{
	int i;
	__m128 vals, ab, cd, f0, f1, f2, f3;
	__m128 scale, half, zero, one;
	__m128i idx, v0, v1, v2, v3;

	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	zero = _mm_setzero_ps();
	one = _mm_set1_ps(1.0f);

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

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		out[i]   = _mm_cvtsi128_si32(idx);
		out[i+1] = _mm_cvtsi128_si32(_mm_srli_si128(idx, 4));
		out[i+2] = _mm_cvtsi128_si32(_mm_srli_si128(idx, 8));
		out[i+3] = _mm_cvtsi128_si32(_mm_srli_si128(idx, 12));

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
		oil_shift_left_f_sse2(sums);
		sums += 4;
	}
}

void oil_yscale_out_linear_sse2(float *sums, int len, unsigned char *out)
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
		oil_shift_left_f_sse2(sums);
		sums += 4;
	}
}

void oil_yscale_out_ga_sse2(float *sums, int width, unsigned char *out)
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

void oil_yscale_out_rgbx_sse2(float *sums, int width, unsigned char *out)
{
	int i;
	__m128 scale, vals, ab, cd, f0, f1, f2;
	__m128i idx, v0, v1, v2;
	unsigned char *lut;

	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));

	for (i=0; i<width; i++) {
		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));
		v2 = _mm_load_si128((__m128i *)(sums + 8));

		f0 = _mm_castsi128_ps(v0);
		f1 = _mm_castsi128_ps(v1);
		f2 = _mm_castsi128_ps(v2);
		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f2, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

		idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

		out[0] = lut[_mm_cvtsi128_si32(idx)];
		out[1] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 4))];
		out[2] = lut[_mm_cvtsi128_si32(_mm_srli_si128(idx, 8))];
		out[3] = 255;

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));
		_mm_store_si128((__m128i *)(sums + 8), _mm_srli_si128(v2, 4));

		sums += 12;
		out += 4;
	}
}

void oil_scale_down_g_sse2(unsigned char *in, float *sums_y_out,
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

void oil_scale_down_ga_sse2(unsigned char *in, float *sums_y_out,
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

void oil_scale_down_rgb_sse2(unsigned char *in, float *sums_y_out,
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

void oil_yscale_out_rgba_sse2(float *sums, int width, unsigned char *out)
{
	int i;
	__m128 scale, one, zero;
	__m128 f0, f1, f2, f3, ab, cd, vals, alpha_v;
	__m128i idx, v0, v1, v2, v3;
	float alpha;
	unsigned char *lut;

	lut = l2s_map;
	scale = _mm_set1_ps((float)(l2s_len - 1));
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();


	for (i=0; i<width; i++) {
		v0 = _mm_load_si128((__m128i *)sums);
		v1 = _mm_load_si128((__m128i *)(sums + 4));
		v2 = _mm_load_si128((__m128i *)(sums + 8));
		v3 = _mm_load_si128((__m128i *)(sums + 12));

		/* Gather first element of each accumulator: {R, G, B, A} */
		f0 = _mm_castsi128_ps(v0);
		f1 = _mm_castsi128_ps(v1);
		f2 = _mm_castsi128_ps(v2);
		f3 = _mm_castsi128_ps(v3);
		ab = _mm_shuffle_ps(f0, f1, _MM_SHUFFLE(0, 0, 0, 0));
		cd = _mm_shuffle_ps(f2, f3, _MM_SHUFFLE(0, 0, 0, 0));
		vals = _mm_shuffle_ps(ab, cd, _MM_SHUFFLE(2, 0, 2, 0));

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

		_mm_store_si128((__m128i *)sums, _mm_srli_si128(v0, 4));
		_mm_store_si128((__m128i *)(sums + 4), _mm_srli_si128(v1, 4));
		_mm_store_si128((__m128i *)(sums + 8), _mm_srli_si128(v2, 4));
		_mm_store_si128((__m128i *)(sums + 12), _mm_srli_si128(v3, 4));

		sums += 16;
		out += 4;
	}
}

void oil_scale_down_rgba_sse2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	__m128 sum_r, sum_g, sum_b, sum_a;
	__m128 sum_r2, sum_g2, sum_b2, sum_a2;
	__m128 coeffs_y, sums_y, sample_y;
	float *sl;

	sl = s2l_map;
	coeffs_y = _mm_load_ps(coeffs_y_f);

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
				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				coeffs_x_a = _mm_mul_ps(coeffs_x, _mm_set1_ps(i2f_map[in[3]]));

				sample_x = _mm_set1_ps(sl[in[0]]);
				sum_r = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_r);

				sample_x = _mm_set1_ps(sl[in[1]]);
				sum_g = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_g);

				sample_x = _mm_set1_ps(sl[in[2]]);
				sum_b = _mm_add_ps(_mm_mul_ps(coeffs_x_a, sample_x), sum_b);

				sum_a = _mm_add_ps(coeffs_x_a, sum_a);

				coeffs_x2_a = _mm_mul_ps(coeffs_x2, _mm_set1_ps(i2f_map[in[7]]));

				sample_x = _mm_set1_ps(sl[in[4]]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_r2);

				sample_x = _mm_set1_ps(sl[in[5]]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2_a, sample_x), sum_g2);

				sample_x = _mm_set1_ps(sl[in[6]]);
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

		sums_y = _mm_load_ps(sums_y_out);
		sample_y = _mm_shuffle_ps(sum_a, sum_a, _MM_SHUFFLE(0, 0, 0, 0));
		sums_y = _mm_add_ps(_mm_mul_ps(coeffs_y, sample_y), sums_y);
		_mm_store_ps(sums_y_out, sums_y);
		sums_y_out += 4;

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
		sum_a = (__m128)_mm_srli_si128(_mm_castps_si128(sum_a), 4);
	}
}

void oil_yscale_out_cmyk_sse2(float *sums, int len, unsigned char *out)
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
		oil_shift_left_f_sse2(sums);
		sums += 4;
	}
}

void oil_scale_down_cmyk_sse2(unsigned char *in, float *sums_y_out,
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
		if (border_buf[i] >= 4) {
			sum_c2 = _mm_setzero_ps();
			sum_m2 = _mm_setzero_ps();
			sum_y2 = _mm_setzero_ps();
			sum_k2 = _mm_setzero_ps();

			for (j=0; j+1<border_buf[i]; j+=2) {
				coeffs_x = _mm_load_ps(coeffs_x_f);
				coeffs_x2 = _mm_load_ps(coeffs_x_f + 4);

				sample_x = _mm_set1_ps(in[0] * (1.0f/255.0f));
				sum_c = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_c);

				sample_x = _mm_set1_ps(in[1] * (1.0f/255.0f));
				sum_m = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_m);

				sample_x = _mm_set1_ps(in[2] * (1.0f/255.0f));
				sum_y = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_y);

				sample_x = _mm_set1_ps(i2f_map[in[3]]);
				sum_k = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_k);

				sample_x = _mm_set1_ps(in[4] * (1.0f/255.0f));
				sum_c2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_c2);

				sample_x = _mm_set1_ps(in[5] * (1.0f/255.0f));
				sum_m2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_m2);

				sample_x = _mm_set1_ps(in[6] * (1.0f/255.0f));
				sum_y2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_y2);

				sample_x = _mm_set1_ps(i2f_map[in[7]]);
				sum_k2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_k2);

				in += 8;
				coeffs_x_f += 8;
			}

			for (; j<border_buf[i]; j++) {
				coeffs_x = _mm_load_ps(coeffs_x_f);

				sample_x = _mm_set1_ps(in[0] * (1.0f/255.0f));
				sum_c = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_c);

				sample_x = _mm_set1_ps(in[1] * (1.0f/255.0f));
				sum_m = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_m);

				sample_x = _mm_set1_ps(in[2] * (1.0f/255.0f));
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

				sample_x = _mm_set1_ps(in[0] * (1.0f/255.0f));
				sum_c = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_c);

				sample_x = _mm_set1_ps(in[1] * (1.0f/255.0f));
				sum_m = _mm_add_ps(_mm_mul_ps(coeffs_x, sample_x), sum_m);

				sample_x = _mm_set1_ps(in[2] * (1.0f/255.0f));
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

void oil_scale_down_rgbx_sse2(unsigned char *in, float *sums_y_out,
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
		if (border_buf[i] >= 2) {
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

				sample_x = _mm_set1_ps(s2l_map[in[4]]);
				sum_r2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_r2);

				sample_x = _mm_set1_ps(s2l_map[in[5]]);
				sum_g2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_g2);

				sample_x = _mm_set1_ps(s2l_map[in[6]]);
				sum_b2 = _mm_add_ps(_mm_mul_ps(coeffs_x2, sample_x), sum_b2);

				in += 8;
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

				in += 4;
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

				in += 4;
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
