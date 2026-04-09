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


static void oil_yscale_out_rgbx_nogamma_sse2(float *sums, int width,
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

	for (i=0; i+1<width; i+=2) {
		/* Pixel 1: read only the current tap */
		vals = _mm_load_ps(sums + tap_off);

		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
		idx = _mm_or_si128(_mm_and_si128(idx, mask), x_val);

		/* Zero consumed tap */
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		/* Pixel 2 */
		{
			__m128i idx2;
			__m128 vals2;

			vals2 = _mm_load_ps(sums + 16 + tap_off);

			vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
			idx2 = _mm_cvttps_epi32(_mm_add_ps(
				_mm_mul_ps(vals2, scale), half));
			idx2 = _mm_or_si128(_mm_and_si128(idx2, mask), x_val);

			packed = _mm_packs_epi32(idx, idx2);
			packed = _mm_packus_epi16(packed, packed);
			_mm_storel_epi64((__m128i *)out, packed);

			/* Zero consumed tap */
			_mm_store_si128((__m128i *)(sums + 16 + tap_off), z);
		}

		sums += 32;
		out += 8;
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

static void oil_scale_down_rgbx_nogamma_sse2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	int off0, off1, off2, off3;
	__m128 coeffs_x, coeffs_x2, sample_x, sum_r, sum_g, sum_b;
	__m128 sum_r2, sum_g2, sum_b2;
	__m128 cy0, cy1, cy2, cy3;
	float *lut;

	lut = i2f_map;
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


static void oil_yscale_out_rgba_nogamma_sse2(float *sums, int width,
	unsigned char *out, int tap)
{
	int i, tap_off;
	__m128 scale, half, one, zero;
	__m128 vals, alpha_v;
	__m128i idx, packed;
	__m128i z;
	float alpha;

	tap_off = tap * 4;
	scale = _mm_set1_ps(255.0f);
	half = _mm_set1_ps(0.5f);
	one = _mm_set1_ps(1.0f);
	zero = _mm_setzero_ps();
	z = _mm_setzero_si128();

	for (i=0; i+1<width; i+=2) {
		/* Pixel 1: read only the current tap, zero it */
		vals = _mm_load_ps(sums + tap_off);

		alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);
		if (alpha != 0) {
			vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
		}
		vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
		{
			__m128 hi = _mm_shuffle_ps(vals, alpha_v,
				_MM_SHUFFLE(0, 0, 2, 2));
			vals = _mm_shuffle_ps(vals, hi,
				_MM_SHUFFLE(2, 0, 1, 0));
		}
		idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

		/* Zero consumed tap */
		_mm_store_si128((__m128i *)(sums + tap_off), z);

		/* Pixel 2 */
		{
			__m128i idx2;
			__m128 vals2, alpha_v2;

			vals2 = _mm_load_ps(sums + 16 + tap_off);

			alpha_v2 = _mm_shuffle_ps(vals2, vals2,
				_MM_SHUFFLE(3, 3, 3, 3));
			alpha_v2 = _mm_min_ps(_mm_max_ps(alpha_v2, zero), one);
			alpha = _mm_cvtss_f32(alpha_v2);
			if (alpha != 0) {
				vals2 = _mm_mul_ps(vals2, _mm_rcp_ps(alpha_v2));
			}
			vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
			{
				__m128 hi2 = _mm_shuffle_ps(vals2, alpha_v2,
					_MM_SHUFFLE(0, 0, 2, 2));
				vals2 = _mm_shuffle_ps(vals2, hi2,
					_MM_SHUFFLE(2, 0, 1, 0));
			}
			idx2 = _mm_cvttps_epi32(_mm_add_ps(
				_mm_mul_ps(vals2, scale), half));

			packed = _mm_packs_epi32(idx, idx2);
			packed = _mm_packus_epi16(packed, packed);
			_mm_storel_epi64((__m128i *)out, packed);

			/* Zero consumed tap */
			_mm_store_si128((__m128i *)(sums + 16 + tap_off), z);
		}

		sums += 32;
		out += 8;
	}

	for (; i<width; i++) {
		vals = _mm_load_ps(sums + tap_off);

		alpha_v = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
		alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
		alpha = _mm_cvtss_f32(alpha_v);
		if (alpha != 0) {
			vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
		}
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


static void oil_scale_down_rgba_nogamma_sse2(unsigned char *in, float *sums_y_out,
	int out_width, float *coeffs_x_f, int *border_buf, float *coeffs_y_f,
	int tap)
{
	int i, j;
	__m128 coeffs_x, coeffs_x2, coeffs_x_a, coeffs_x2_a, sample_x;
	__m128 sum_r, sum_g, sum_b, sum_a;
	__m128 sum_r2, sum_g2, sum_b2, sum_a2;
	float *lut;
	int off0, off1, off2, off3;
	__m128 cy0, cy1, cy2, cy3;
	off0 = tap * 4;
	off1 = ((tap + 1) & 3) * 4;
	off2 = ((tap + 2) & 3) * 4;
	off3 = ((tap + 3) & 3) * 4;
	cy0 = _mm_set1_ps(coeffs_y_f[0]);
	cy1 = _mm_set1_ps(coeffs_y_f[1]);
	cy2 = _mm_set1_ps(coeffs_y_f[2]);
	cy3 = _mm_set1_ps(coeffs_y_f[3]);

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
				__m128 sy;
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
			}
			sums_y_out += 16;
		}

		sum_r = (__m128)_mm_srli_si128(_mm_castps_si128(sum_r), 4);
		sum_g = (__m128)_mm_srli_si128(_mm_castps_si128(sum_g), 4);
		sum_b = (__m128)_mm_srli_si128(_mm_castps_si128(sum_b), 4);
		sum_a = (__m128)_mm_srli_si128(_mm_castps_si128(sum_a), 4);
	}
}


/* SSE2 dispatch functions */


static void yscale_out_sse2(float *sums, int width, unsigned char *out,
	enum oil_colorspace cs, int tap)
{
	switch(cs) {
	case OIL_CS_RGBA_NOGAMMA:
		oil_yscale_out_rgba_nogamma_sse2(sums, width, out, tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_yscale_out_rgbx_nogamma_sse2(sums, width, out, tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}
}


static void down_scale_in_sse2(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	switch(os->cs) {
	case OIL_CS_RGBA_NOGAMMA:
		oil_scale_down_rgba_nogamma_sse2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_scale_down_rgbx_nogamma_sse2(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_UNKNOWN:
		break;
	}

	os->borders_y[os->out_pos] -= 1;
	os->in_pos++;
}


int oil_scale_in_sse2(struct oil_scale *os, unsigned char *in)
{
	if (oil_scale_slots(os) == 0) {
		return -1;
	}
	down_scale_in_sse2(os, in);
	return 0;
}

int oil_scale_out_sse2(struct oil_scale *os, unsigned char *out)
{
	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	yscale_out_sse2(os->sums_y, os->out_width, out, os->cs, os->sums_y_tap);
	os->sums_y_tap = (os->sums_y_tap + 1) & 3;

	os->out_pos++;
	return 0;
}


