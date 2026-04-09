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

static void yscale_out_neon(float *sums, int width, unsigned char *out,
	enum oil_colorspace cs, int tap)
{
	switch(cs) {
	case OIL_CS_RGBA_NOGAMMA:
		oil_yscale_out_rgba_nogamma_neon(sums, width, out, tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_yscale_out_rgbx_nogamma_neon(sums, width, out, tap);
		break;
	}
}

static void down_scale_in_neon(struct oil_scale *os, unsigned char *in)
{
	float *coeffs_y;

	coeffs_y = os->coeffs_y + os->in_pos * 4;

	switch(os->cs) {
	case OIL_CS_RGBA_NOGAMMA:
		oil_scale_down_rgba_nogamma_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	case OIL_CS_RGBX_NOGAMMA:
		oil_scale_down_rgbx_nogamma_neon(in, os->sums_y, os->out_width, os->coeffs_x, os->borders_x, coeffs_y, os->sums_y_tap);
		break;
	}

	os->borders_y[os->out_pos] -= 1;
	os->in_pos++;
}

int oil_scale_in_neon(struct oil_scale *os, unsigned char *in)
{
	if (oil_scale_slots(os) == 0) {
		return -1;
	}
	down_scale_in_neon(os, in);
	return 0;
}

int oil_scale_out_neon(struct oil_scale *os, unsigned char *out)
{
	if (oil_scale_slots(os) != 0) {
		return -1;
	}

	yscale_out_neon(os->sums_y, os->out_width, out, os->cs, os->sums_y_tap);
	os->sums_y_tap = (os->sums_y_tap + 1) & 3;

	os->out_pos++;
	return 0;
}
