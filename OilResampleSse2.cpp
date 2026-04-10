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
#include <immintrin.h>
#include <cstring>

namespace mozilla {

static void OilYscaleOutRgbxNogammaSse2(float* aSums, int aWidth,
                                        unsigned char* aOut, int aTap) {
  int i, tapOff;
  __m128 scale, half, one, zero;
  __m128 vals;
  __m128i idx, packed;
  __m128i z, mask, xVal;

  tapOff = aTap * 4;
  scale = _mm_set1_ps(255.0f);
  half = _mm_set1_ps(0.5f);
  one = _mm_set1_ps(1.0f);
  zero = _mm_setzero_ps();
  z = _mm_setzero_si128();
  mask = _mm_set_epi32(0, -1, -1, -1);
  xVal = _mm_set_epi32(255, 0, 0, 0);

  for (i = 0; i + 1 < aWidth; i += 2) {
    /* Pixel 1: read only the current tap */
    vals = _mm_load_ps(aSums + tapOff);

    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    idx = _mm_or_si128(_mm_and_si128(idx, mask), xVal);

    /* Zero consumed tap */
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    /* Pixel 2 */
    {
      __m128i idx2;
      __m128 vals2;

      vals2 = _mm_load_ps(aSums + 16 + tapOff);

      vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
      idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));
      idx2 = _mm_or_si128(_mm_and_si128(idx2, mask), xVal);

      packed = _mm_packs_epi32(idx, idx2);
      packed = _mm_packus_epi16(packed, packed);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(aOut), packed);

      /* Zero consumed tap */
      _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    }

    aSums += 32;
    aOut += 8;
  }

  for (; i < aWidth; i++) {
    vals = _mm_load_ps(aSums + tapOff);

    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    idx = _mm_or_si128(_mm_and_si128(idx, mask), xVal);
    packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);
    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void OilScaleDownRgbxNogammaSse2(unsigned char* aIn, float* aSumsYOut,
                                        int aOutWidth, float* aCoeffsXF,
                                        int* aBorderBuf, float* aCoeffsYF,
                                        int aTap) {
  int i, j;
  int off0, off1, off2, off3;
  __m128 coeffsX, coeffsX2, sampleX, sumR, sumG, sumB;
  __m128 sumR2, sumG2, sumB2;
  __m128 cy0, cy1, cy2, cy3;
  float* lut;

  lut = gI2fMap;
  off0 = aTap * 4;
  off1 = ((aTap + 1) & 3) * 4;
  off2 = ((aTap + 2) & 3) * 4;
  off3 = ((aTap + 3) & 3) * 4;
  cy0 = _mm_set1_ps(aCoeffsYF[0]);
  cy1 = _mm_set1_ps(aCoeffsYF[1]);
  cy2 = _mm_set1_ps(aCoeffsYF[2]);
  cy3 = _mm_set1_ps(aCoeffsYF[3]);

  sumR = _mm_setzero_ps();
  sumG = _mm_setzero_ps();
  sumB = _mm_setzero_ps();

  for (i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      sumR2 = _mm_setzero_ps();
      sumG2 = _mm_setzero_ps();
      sumB2 = _mm_setzero_ps();

      for (j = 0; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);
        coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_add_ps(_mm_mul_ps(coeffsX2, sampleX), sumB2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = _mm_add_ps(sumR, sumR2);
      sumG = _mm_add_ps(sumG, sumG2);
      sumB = _mm_add_ps(sumB, sumB2);
    } else {
      for (j = 0; j < aBorderBuf[i]; j++) {
        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[aIn[0]]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsX, sampleX), sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      __m128 rg, bx, rgbx, sy;

      rg = _mm_unpacklo_ps(sumR, sumG);
      bx = _mm_unpacklo_ps(sumB, sumB);
      rgbx = _mm_movelh_ps(rg, bx);

      sy = _mm_load_ps(aSumsYOut + off0);
      sy = _mm_add_ps(_mm_mul_ps(cy0, rgbx), sy);
      _mm_store_ps(aSumsYOut + off0, sy);

      sy = _mm_load_ps(aSumsYOut + off1);
      sy = _mm_add_ps(_mm_mul_ps(cy1, rgbx), sy);
      _mm_store_ps(aSumsYOut + off1, sy);

      sy = _mm_load_ps(aSumsYOut + off2);
      sy = _mm_add_ps(_mm_mul_ps(cy2, rgbx), sy);
      _mm_store_ps(aSumsYOut + off2, sy);

      sy = _mm_load_ps(aSumsYOut + off3);
      sy = _mm_add_ps(_mm_mul_ps(cy3, rgbx), sy);
      _mm_store_ps(aSumsYOut + off3, sy);

      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
  }
}

static void OilYscaleOutRgbaNogammaSse2(float* aSums, int aWidth,
                                        unsigned char* aOut, int aTap) {
  int i, tapOff;
  __m128 scale, half, one, zero;
  __m128 vals, alphaV;
  __m128i idx, packed;
  __m128i z;
  float alpha;

  tapOff = aTap * 4;
  scale = _mm_set1_ps(255.0f);
  half = _mm_set1_ps(0.5f);
  one = _mm_set1_ps(1.0f);
  zero = _mm_setzero_ps();
  z = _mm_setzero_si128();

  for (i = 0; i + 1 < aWidth; i += 2) {
    /* Pixel 1: read only the current tap, zero it */
    vals = _mm_load_ps(aSums + tapOff);

    alphaV = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV = _mm_min_ps(_mm_max_ps(alphaV, zero), one);
    alpha = _mm_cvtss_f32(alphaV);
    if (alpha != 0) {
      vals = _mm_mul_ps(vals, _mm_rcp_ps(alphaV));
    }
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    {
      __m128 hi = _mm_shuffle_ps(vals, alphaV, _MM_SHUFFLE(0, 0, 2, 2));
      vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
    }
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));

    /* Zero consumed tap */
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    /* Pixel 2 */
    {
      __m128i idx2;
      __m128 vals2, alphaV2;

      vals2 = _mm_load_ps(aSums + 16 + tapOff);

      alphaV2 = _mm_shuffle_ps(vals2, vals2, _MM_SHUFFLE(3, 3, 3, 3));
      alphaV2 = _mm_min_ps(_mm_max_ps(alphaV2, zero), one);
      alpha = _mm_cvtss_f32(alphaV2);
      if (alpha != 0) {
        vals2 = _mm_mul_ps(vals2, _mm_rcp_ps(alphaV2));
      }
      vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
      {
        __m128 hi2 = _mm_shuffle_ps(vals2, alphaV2, _MM_SHUFFLE(0, 0, 2, 2));
        vals2 = _mm_shuffle_ps(vals2, hi2, _MM_SHUFFLE(2, 0, 1, 0));
      }
      idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));

      packed = _mm_packs_epi32(idx, idx2);
      packed = _mm_packus_epi16(packed, packed);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(aOut), packed);

      /* Zero consumed tap */
      _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    }

    aSums += 32;
    aOut += 8;
  }

  for (; i < aWidth; i++) {
    vals = _mm_load_ps(aSums + tapOff);

    alphaV = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV = _mm_min_ps(_mm_max_ps(alphaV, zero), one);
    alpha = _mm_cvtss_f32(alphaV);
    if (alpha != 0) {
      vals = _mm_mul_ps(vals, _mm_rcp_ps(alphaV));
    }
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    {
      __m128 hi = _mm_shuffle_ps(vals, alphaV, _MM_SHUFFLE(0, 0, 2, 2));
      vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
    }
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    packed = _mm_packs_epi32(idx, idx);
    packed = _mm_packus_epi16(packed, packed);
    *reinterpret_cast<int*>(aOut) = _mm_cvtsi128_si32(packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    aSums += 16;
    aOut += 4;
  }
}

static void OilScaleDownRgbaNogammaSse2(unsigned char* aIn, float* aSumsYOut,
                                        int aOutWidth, float* aCoeffsXF,
                                        int* aBorderBuf, float* aCoeffsYF,
                                        int aTap) {
  int i, j;
  __m128 coeffsX, coeffsX2, coeffsXA, coeffsX2A, sampleX;
  __m128 sumR, sumG, sumB, sumA;
  __m128 sumR2, sumG2, sumB2, sumA2;
  float* lut;
  int off0, off1, off2, off3;
  __m128 cy0, cy1, cy2, cy3;
  off0 = aTap * 4;
  off1 = ((aTap + 1) & 3) * 4;
  off2 = ((aTap + 2) & 3) * 4;
  off3 = ((aTap + 3) & 3) * 4;
  cy0 = _mm_set1_ps(aCoeffsYF[0]);
  cy1 = _mm_set1_ps(aCoeffsYF[1]);
  cy2 = _mm_set1_ps(aCoeffsYF[2]);
  cy3 = _mm_set1_ps(aCoeffsYF[3]);

  lut = gI2fMap;

  sumR = _mm_setzero_ps();
  sumG = _mm_setzero_ps();
  sumB = _mm_setzero_ps();
  sumA = _mm_setzero_ps();

  for (i = 0; i < aOutWidth; i++) {
    if (aBorderBuf[i] >= 4) {
      sumR2 = _mm_setzero_ps();
      sumG2 = _mm_setzero_ps();
      sumB2 = _mm_setzero_ps();
      sumA2 = _mm_setzero_ps();

      for (j = 0; j + 1 < aBorderBuf[i]; j += 2) {
        unsigned int px0, px1;
        memcpy(&px0, aIn, 4);
        memcpy(&px1, aIn + 4, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);
        coeffsX2 = _mm_load_ps(aCoeffsXF + 4);

        coeffsXA = _mm_mul_ps(coeffsX, _mm_set1_ps(lut[px0 >> 24]));

        sampleX = _mm_set1_ps(lut[px0 & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumB);

        sumA = _mm_add_ps(coeffsXA, sumA);

        coeffsX2A = _mm_mul_ps(coeffsX2, _mm_set1_ps(lut[px1 >> 24]));

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_add_ps(_mm_mul_ps(coeffsX2A, sampleX), sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_add_ps(_mm_mul_ps(coeffsX2A, sampleX), sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_add_ps(_mm_mul_ps(coeffsX2A, sampleX), sumB2);

        sumA2 = _mm_add_ps(coeffsX2A, sumA2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);

        coeffsXA = _mm_mul_ps(coeffsX, _mm_set1_ps(lut[px >> 24]));

        sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumB);

        sumA = _mm_add_ps(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }

      sumR = _mm_add_ps(sumR, sumR2);
      sumG = _mm_add_ps(sumG, sumG2);
      sumB = _mm_add_ps(sumB, sumB2);
      sumA = _mm_add_ps(sumA, sumA2);
    } else {
      for (j = 0; j < aBorderBuf[i]; j++) {
        coeffsX = _mm_load_ps(aCoeffsXF);

        coeffsXA = _mm_mul_ps(coeffsX, _mm_set1_ps(lut[aIn[3]]));

        sampleX = _mm_set1_ps(lut[aIn[0]]);
        sumR = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_add_ps(_mm_mul_ps(coeffsXA, sampleX), sumB);

        sumA = _mm_add_ps(coeffsXA, sumA);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using ring buffer offsets */
    {
      __m128 rg, ba, rgba;

      rg = _mm_unpacklo_ps(sumR, sumG);
      ba = _mm_unpacklo_ps(sumB, sumA);
      rgba = _mm_movelh_ps(rg, ba);

      {
        __m128 sy;
        sy = _mm_load_ps(aSumsYOut + off0);
        sy = _mm_add_ps(_mm_mul_ps(cy0, rgba), sy);
        _mm_store_ps(aSumsYOut + off0, sy);

        sy = _mm_load_ps(aSumsYOut + off1);
        sy = _mm_add_ps(_mm_mul_ps(cy1, rgba), sy);
        _mm_store_ps(aSumsYOut + off1, sy);

        sy = _mm_load_ps(aSumsYOut + off2);
        sy = _mm_add_ps(_mm_mul_ps(cy2, rgba), sy);
        _mm_store_ps(aSumsYOut + off2, sy);

        sy = _mm_load_ps(aSumsYOut + off3);
        sy = _mm_add_ps(_mm_mul_ps(cy3, rgba), sy);
        _mm_store_ps(aSumsYOut + off3, sy);
      }
      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
    sumA = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumA), 4));
  }
}

/* SSE2 dispatch functions */

static void YscaleOutSse2(float* aSums, int aWidth, unsigned char* aOut,
                          OilColorspace aCs, int aTap) {
  switch (aCs) {
    case OilColorspace::RgbaNogamma:
      OilYscaleOutRgbaNogammaSse2(aSums, aWidth, aOut, aTap);
      break;
    case OilColorspace::RgbxNogamma:
      OilYscaleOutRgbxNogammaSse2(aSums, aWidth, aOut, aTap);
      break;
  }
}

static void DownScaleInSse2(OilScale* aOs, unsigned char* aIn) {
  float* coeffsY;

  coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  switch (aOs->mCs) {
    case OilColorspace::RgbaNogamma:
      OilScaleDownRgbaNogammaSse2(aIn, aOs->mSumsY, aOs->mOutWidth,
                                  aOs->mCoeffsX, aOs->mBordersX, coeffsY,
                                  aOs->mSumsYTap);
      break;
    case OilColorspace::RgbxNogamma:
      OilScaleDownRgbxNogammaSse2(aIn, aOs->mSumsY, aOs->mOutWidth,
                                  aOs->mCoeffsX, aOs->mBordersX, coeffsY,
                                  aOs->mSumsYTap);
      break;
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

int OilScaleInSse2(OilScale* aOs, unsigned char* aIn) {
  if (OilScaleSlots(aOs) == 0) {
    return -1;
  }
  DownScaleInSse2(aOs, aIn);
  return 0;
}

int OilScaleOutSse2(OilScale* aOs, unsigned char* aOut) {
  if (OilScaleSlots(aOs) != 0) {
    return -1;
  }

  YscaleOutSse2(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mCs, aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;

  aOs->mOutPos++;
  return 0;
}

}  // namespace mozilla
