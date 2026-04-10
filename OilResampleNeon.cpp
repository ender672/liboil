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

#include "OilResample.h"
#include "OilResampleInternal.h"

#include <cstring>
#include <arm_neon.h>

namespace mozilla {

static void OilYscaleOutRgbaNogammaNeon(float* aSums, int aWidth,
                                        unsigned char* aOut, int aTap) {
  int i, tapOff;
  float32x4_t scaleV, one, zero, half;
  float32x4_t vals, alphaV;
  int32x4_t idx;
  float32x4_t z;
  float alpha;

  tapOff = aTap * 4;
  scaleV = vdupq_n_f32(255.0f);
  one = vdupq_n_f32(1.0f);
  zero = vdupq_n_f32(0.0f);
  half = vdupq_n_f32(0.5f);
  z = vdupq_n_f32(0.0f);

  for (i = 0; i < aWidth; i++) {
    /* Read [R, G, B, A] from current tap slot */
    vals = vld1q_f32(aSums + tapOff);

    /* Clamp alpha to [0, 1] */
    alpha = vgetq_lane_f32(vals, 3);
    if (alpha > 1.0f) {
      alpha = 1.0f;
    } else if (alpha < 0.0f) {
      alpha = 0.0f;
    }
    alphaV = vdupq_n_f32(alpha);

    /* Divide RGB by alpha (skip if alpha == 0) */
    if (alpha != 0) {
      vals = vdivq_f32(vals, alphaV);
    }

    /* Clamp RGB to [0, 1], scale to [0, 255], round */
    vals = vminq_f32(vmaxq_f32(vals, zero), one);
    idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scaleV), half));

    aOut[0] = vgetq_lane_s32(idx, 0);
    aOut[1] = vgetq_lane_s32(idx, 1);
    aOut[2] = vgetq_lane_s32(idx, 2);
    aOut[3] = static_cast<int>(alpha * 255.0f + 0.5f);

    /* Zero consumed tap */
    vst1q_f32(aSums + tapOff, z);

    aSums += 16;
    aOut += 4;
  }
}

static void OilScaleDownRgbaNogammaNeon(unsigned char* aIn, float* aSumsYOut,
                                        int aOutWidth, float* aCoeffsXF,
                                        int* aBorderBuf, float* aCoeffsYF,
                                        int aTap) {
  int i, j;
  int off0, off1, off2, off3;
  float32x4_t coeffsX, coeffsX2, coeffsXA, coeffsX2A, sampleX;
  float32x4_t sumR, sumG, sumB, sumA;
  float32x4_t sumR2, sumG2, sumB2, sumA2;
  float32x4_t cy0, cy1, cy2, cy3;

  off0 = aTap * 4;
  off1 = ((aTap + 1) & 3) * 4;
  off2 = ((aTap + 2) & 3) * 4;
  off3 = ((aTap + 3) & 3) * 4;
  cy0 = vdupq_n_f32(aCoeffsYF[0]);
  cy1 = vdupq_n_f32(aCoeffsYF[1]);
  cy2 = vdupq_n_f32(aCoeffsYF[2]);
  cy3 = vdupq_n_f32(aCoeffsYF[3]);

  sumR = vdupq_n_f32(0.0f);
  sumG = vdupq_n_f32(0.0f);
  sumB = vdupq_n_f32(0.0f);
  sumA = vdupq_n_f32(0.0f);

  for (i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      sumR2 = vdupq_n_f32(0.0f);
      sumG2 = vdupq_n_f32(0.0f);
      sumB2 = vdupq_n_f32(0.0f);
      sumA2 = vdupq_n_f32(0.0f);

      for (j = 0; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        coeffsX = vld1q_f32(aCoeffsXF);
        coeffsX2 = vld1q_f32(aCoeffsXF + 4);

        coeffsXA = vmulq_f32(coeffsX, vdupq_n_f32(gI2fMap[px0 >> 24]));

        sampleX = vdupq_n_f32(gI2fMap[px0 & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumB);

        sumA = vaddq_f32(coeffsXA, sumA);

        coeffsX2A = vmulq_f32(coeffsX2, vdupq_n_f32(gI2fMap[px1 >> 24]));

        sampleX = vdupq_n_f32(gI2fMap[px1 & 0xFF]);
        sumR2 = vaddq_f32(vmulq_f32(coeffsX2A, sampleX), sumR2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 8) & 0xFF]);
        sumG2 = vaddq_f32(vmulq_f32(coeffsX2A, sampleX), sumG2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 16) & 0xFF]);
        sumB2 = vaddq_f32(vmulq_f32(coeffsX2A, sampleX), sumB2);

        sumA2 = vaddq_f32(coeffsX2A, sumA2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = vld1q_f32(aCoeffsXF);

        coeffsXA = vmulq_f32(coeffsX, vdupq_n_f32(gI2fMap[px >> 24]));

        sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumB);

        sumA = vaddq_f32(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = vaddq_f32(sumR, sumR2);
      sumG = vaddq_f32(sumG, sumG2);
      sumB = vaddq_f32(sumB, sumB2);
      sumA = vaddq_f32(sumA, sumA2);
    } else {
      for (j = 0; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = vld1q_f32(aCoeffsXF);

        coeffsXA = vmulq_f32(coeffsX, vdupq_n_f32(gI2fMap[px >> 24]));

        sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsXA, sampleX), sumB);

        sumA = vaddq_f32(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      float32x4_t rgba, sy;

      rgba = vsetq_lane_f32(vgetq_lane_f32(sumR, 0), vdupq_n_f32(0), 0);
      rgba = vsetq_lane_f32(vgetq_lane_f32(sumG, 0), rgba, 1);
      rgba = vsetq_lane_f32(vgetq_lane_f32(sumB, 0), rgba, 2);
      rgba = vsetq_lane_f32(vgetq_lane_f32(sumA, 0), rgba, 3);

      sy = vld1q_f32(aSumsYOut + off0);
      sy = vfmaq_f32(sy, cy0, rgba);
      vst1q_f32(aSumsYOut + off0, sy);

      sy = vld1q_f32(aSumsYOut + off1);
      sy = vfmaq_f32(sy, cy1, rgba);
      vst1q_f32(aSumsYOut + off1, sy);

      sy = vld1q_f32(aSumsYOut + off2);
      sy = vfmaq_f32(sy, cy2, rgba);
      vst1q_f32(aSumsYOut + off2, sy);

      sy = vld1q_f32(aSumsYOut + off3);
      sy = vfmaq_f32(sy, cy3, rgba);
      vst1q_f32(aSumsYOut + off3, sy);

      aSumsYOut += 16;
    }

    sumR = vextq_f32(sumR, vdupq_n_f32(0), 1);
    sumG = vextq_f32(sumG, vdupq_n_f32(0), 1);
    sumB = vextq_f32(sumB, vdupq_n_f32(0), 1);
    sumA = vextq_f32(sumA, vdupq_n_f32(0), 1);
  }
}

static void OilYscaleOutRgbxNogammaNeon(float* aSums, int aWidth,
                                        unsigned char* aOut, int aTap) {
  int i, tapOff;
  float32x4_t scaleV, one, zero, half;
  float32x4_t z;
  uint8x16_t alphaMask;

  tapOff = aTap * 4;
  scaleV = vdupq_n_f32(255.0f);
  one = vdupq_n_f32(1.0f);
  zero = vdupq_n_f32(0.0f);
  half = vdupq_n_f32(0.5f);
  z = vdupq_n_f32(0.0f);

  {
    static const uint8_t amask[16] = {0, 0, 0, 255, 0, 0, 0, 255,
                                      0, 0, 0, 255, 0, 0, 0, 255};
    alphaMask = vld1q_u8(amask);
  }

  for (i = 0; i + 3 < aWidth; i += 4) {
    float32x4_t v0, v1, v2, v3;
    int32x4_t i0, i1, i2, i3;
    int16x4_t h0, h1, h2, h3;
    int16x8_t h01, h23;
    uint8x8_t b01, b23;
    uint8x16_t result;

    v0 = vld1q_f32(aSums + tapOff);
    v1 = vld1q_f32(aSums + 16 + tapOff);
    v2 = vld1q_f32(aSums + 32 + tapOff);
    v3 = vld1q_f32(aSums + 48 + tapOff);

    v0 = vminq_f32(vmaxq_f32(v0, zero), one);
    v1 = vminq_f32(vmaxq_f32(v1, zero), one);
    v2 = vminq_f32(vmaxq_f32(v2, zero), one);
    v3 = vminq_f32(vmaxq_f32(v3, zero), one);

    i0 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v0, scaleV), half));
    i1 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v1, scaleV), half));
    i2 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v2, scaleV), half));
    i3 = vcvtq_s32_f32(vaddq_f32(vmulq_f32(v3, scaleV), half));

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
    result = vbslq_u8(alphaMask, vdupq_n_u8(255), result);

    vst1q_u8(aOut, result);

    /* Zero consumed taps */
    vst1q_f32(aSums + tapOff, z);
    vst1q_f32(aSums + 16 + tapOff, z);
    vst1q_f32(aSums + 32 + tapOff, z);
    vst1q_f32(aSums + 48 + tapOff, z);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    float32x4_t vals;
    int32x4_t idx;

    vals = vld1q_f32(aSums + tapOff);
    vals = vminq_f32(vmaxq_f32(vals, zero), one);
    idx = vcvtq_s32_f32(vaddq_f32(vmulq_f32(vals, scaleV), half));

    aOut[0] = vgetq_lane_s32(idx, 0);
    aOut[1] = vgetq_lane_s32(idx, 1);
    aOut[2] = vgetq_lane_s32(idx, 2);
    aOut[3] = 255;

    vst1q_f32(aSums + tapOff, z);

    aSums += 16;
    aOut += 4;
  }
}

static void OilScaleDownRgbxNogammaNeon(unsigned char* aIn, float* aSumsYOut,
                                        int aOutWidth, float* aCoeffsXF,
                                        int* aBorderBuf, float* aCoeffsYF,
                                        int aTap) {
  int i, j;
  int off0, off1, off2, off3;
  float32x4_t coeffsX, coeffsX2, sampleX, sumR, sumG, sumB, sumX;
  float32x4_t sumR2, sumG2, sumB2, sumX2;
  float32x4_t oneV;
  float32x4_t cy0, cy1, cy2, cy3;

  off0 = aTap * 4;
  off1 = ((aTap + 1) & 3) * 4;
  off2 = ((aTap + 2) & 3) * 4;
  off3 = ((aTap + 3) & 3) * 4;
  cy0 = vdupq_n_f32(aCoeffsYF[0]);
  cy1 = vdupq_n_f32(aCoeffsYF[1]);
  cy2 = vdupq_n_f32(aCoeffsYF[2]);
  cy3 = vdupq_n_f32(aCoeffsYF[3]);
  oneV = vdupq_n_f32(1.0f);

  sumR = vdupq_n_f32(0.0f);
  sumG = vdupq_n_f32(0.0f);
  sumB = vdupq_n_f32(0.0f);
  sumX = vdupq_n_f32(0.0f);

  for (i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      sumR2 = vdupq_n_f32(0.0f);
      sumG2 = vdupq_n_f32(0.0f);
      sumB2 = vdupq_n_f32(0.0f);
      sumX2 = vdupq_n_f32(0.0f);

      for (j = 0; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        coeffsX = vld1q_f32(aCoeffsXF);
        coeffsX2 = vld1q_f32(aCoeffsXF + 4);

        sampleX = vdupq_n_f32(gI2fMap[px0 & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px0 >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumB);

        sumX = vaddq_f32(vmulq_f32(coeffsX, oneV), sumX);

        sampleX = vdupq_n_f32(gI2fMap[px1 & 0xFF]);
        sumR2 = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sumR2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 8) & 0xFF]);
        sumG2 = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sumG2);

        sampleX = vdupq_n_f32(gI2fMap[(px1 >> 16) & 0xFF]);
        sumB2 = vaddq_f32(vmulq_f32(coeffsX2, sampleX), sumB2);

        sumX2 = vaddq_f32(vmulq_f32(coeffsX2, oneV), sumX2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = vld1q_f32(aCoeffsXF);

        sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumB);

        sumX = vaddq_f32(vmulq_f32(coeffsX, oneV), sumX);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = vaddq_f32(sumR, sumR2);
      sumG = vaddq_f32(sumG, sumG2);
      sumB = vaddq_f32(sumB, sumB2);
      sumX = vaddq_f32(sumX, sumX2);
    } else {
      for (j = 0; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = vld1q_f32(aCoeffsXF);

        sampleX = vdupq_n_f32(gI2fMap[px & 0xFF]);
        sumR = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumR);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 8) & 0xFF]);
        sumG = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumG);

        sampleX = vdupq_n_f32(gI2fMap[(px >> 16) & 0xFF]);
        sumB = vaddq_f32(vmulq_f32(coeffsX, sampleX), sumB);

        sumX = vaddq_f32(vmulq_f32(coeffsX, oneV), sumX);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      float32x4_t rgbx, sy;

      rgbx = vsetq_lane_f32(vgetq_lane_f32(sumR, 0), vdupq_n_f32(0), 0);
      rgbx = vsetq_lane_f32(vgetq_lane_f32(sumG, 0), rgbx, 1);
      rgbx = vsetq_lane_f32(vgetq_lane_f32(sumB, 0), rgbx, 2);
      rgbx = vsetq_lane_f32(vgetq_lane_f32(sumX, 0), rgbx, 3);

      sy = vld1q_f32(aSumsYOut + off0);
      sy = vfmaq_f32(sy, cy0, rgbx);
      vst1q_f32(aSumsYOut + off0, sy);

      sy = vld1q_f32(aSumsYOut + off1);
      sy = vfmaq_f32(sy, cy1, rgbx);
      vst1q_f32(aSumsYOut + off1, sy);

      sy = vld1q_f32(aSumsYOut + off2);
      sy = vfmaq_f32(sy, cy2, rgbx);
      vst1q_f32(aSumsYOut + off2, sy);

      sy = vld1q_f32(aSumsYOut + off3);
      sy = vfmaq_f32(sy, cy3, rgbx);
      vst1q_f32(aSumsYOut + off3, sy);

      aSumsYOut += 16;
    }

    sumR = vextq_f32(sumR, vdupq_n_f32(0), 1);
    sumG = vextq_f32(sumG, vdupq_n_f32(0), 1);
    sumB = vextq_f32(sumB, vdupq_n_f32(0), 1);
    sumX = vextq_f32(sumX, vdupq_n_f32(0), 1);
  }
}

/* NEON dispatch functions */

static void YscaleOutNeon(float* aSums, int aWidth, unsigned char* aOut,
                          OilColorspace aCs, int aTap) {
  switch (aCs) {
    case OilColorspace::RgbaNogamma:
      OilYscaleOutRgbaNogammaNeon(aSums, aWidth, aOut, aTap);
      break;
    case OilColorspace::RgbxNogamma:
      OilYscaleOutRgbxNogammaNeon(aSums, aWidth, aOut, aTap);
      break;
  }
}

static void DownScaleInNeon(OilScale* aOs, unsigned char* aIn) {
  float* coeffsY;

  coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  switch (aOs->mCs) {
    case OilColorspace::RgbaNogamma:
      OilScaleDownRgbaNogammaNeon(aIn, aOs->mSumsY, aOs->mOutWidth,
                                  aOs->mCoeffsX, aOs->mBordersX, coeffsY,
                                  aOs->mSumsYTap);
      break;
    case OilColorspace::RgbxNogamma:
      OilScaleDownRgbxNogammaNeon(aIn, aOs->mSumsY, aOs->mOutWidth,
                                  aOs->mCoeffsX, aOs->mBordersX, coeffsY,
                                  aOs->mSumsYTap);
      break;
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

int OilScaleInNeon(OilScale* aOs, unsigned char* aIn) {
  if (OilScaleSlots(aOs) == 0) {
    return -1;
  }
  DownScaleInNeon(aOs, aIn);
  return 0;
}

int OilScaleOutNeon(OilScale* aOs, unsigned char* aOut) {
  if (OilScaleSlots(aOs) != 0) {
    return -1;
  }

  YscaleOutNeon(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mCs, aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;

  aOs->mOutPos++;
  return 0;
}

}  // namespace mozilla
