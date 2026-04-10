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

static void OilYscaleOutRgbxNogammaAvx2(float* aSums, int aWidth,
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

  for (i = 0; i + 3 < aWidth; i += 4) {
    __m128 v0, v1, v2, v3;
    __m128i i0, i1, i2, i3, p01, p23;

    v0 = _mm_load_ps(aSums + tapOff);
    v1 = _mm_load_ps(aSums + 16 + tapOff);
    v2 = _mm_load_ps(aSums + 32 + tapOff);
    v3 = _mm_load_ps(aSums + 48 + tapOff);

    v0 = _mm_min_ps(_mm_max_ps(v0, zero), one);
    v1 = _mm_min_ps(_mm_max_ps(v1, zero), one);
    v2 = _mm_min_ps(_mm_max_ps(v2, zero), one);
    v3 = _mm_min_ps(_mm_max_ps(v3, zero), one);

    i0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v0, scale), half));
    i1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v1, scale), half));
    i2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v2, scale), half));
    i3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v3, scale), half));

    i0 = _mm_or_si128(_mm_and_si128(i0, mask), xVal);
    i1 = _mm_or_si128(_mm_and_si128(i1, mask), xVal);
    i2 = _mm_or_si128(_mm_and_si128(i2, mask), xVal);
    i3 = _mm_or_si128(_mm_and_si128(i3, mask), xVal);

    p01 = _mm_packs_epi32(i0, i1);
    p23 = _mm_packs_epi32(i2, i3);
    packed = _mm_packus_epi16(p01, p23);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aOut), packed);

    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 32 + tapOff), z);
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 48 + tapOff), z);

    aSums += 64;
    aOut += 16;
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

static void OilScaleDownRgbxNogammaAvx2(unsigned char* aIn, float* aSumsYOut,
                                        int aOutWidth, float* aCoeffsXF,
                                        int* aBorderBuf, float* aCoeffsYF,
                                        int aTap) {
  int i, j;
  __m128 coeffsX, coeffsX2, sampleX, sumR, sumG, sumB;
  __m128 sumR2, sumG2, sumB2;
  __m256 cyLo, cyHi;
  float* lut;

  lut = gI2fMap;

  /* Precompute 256-bit coefficient vectors ordered by physical slot */
  {
    float cySlot[4];
    int k;
    for (k = 0; k < 4; k++) {
      cySlot[k] = aCoeffsYF[(k - aTap + 4) & 3];
    }
    cyLo = _mm256_set_m128(_mm_set1_ps(cySlot[1]), _mm_set1_ps(cySlot[0]));
    cyHi = _mm256_set_m128(_mm_set1_ps(cySlot[3]), _mm_set1_ps(cySlot[2]));
  }

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
        sumR = _mm_fmadd_ps(coeffsX, sampleX, sumR);

        sampleX = _mm_set1_ps(lut[(px0 >> 8) & 0xFF]);
        sumG = _mm_fmadd_ps(coeffsX, sampleX, sumG);

        sampleX = _mm_set1_ps(lut[(px0 >> 16) & 0xFF]);
        sumB = _mm_fmadd_ps(coeffsX, sampleX, sumB);

        sampleX = _mm_set1_ps(lut[px1 & 0xFF]);
        sumR2 = _mm_fmadd_ps(coeffsX2, sampleX, sumR2);

        sampleX = _mm_set1_ps(lut[(px1 >> 8) & 0xFF]);
        sumG2 = _mm_fmadd_ps(coeffsX2, sampleX, sumG2);

        sampleX = _mm_set1_ps(lut[(px1 >> 16) & 0xFF]);
        sumB2 = _mm_fmadd_ps(coeffsX2, sampleX, sumB2);

        aIn += 8;
        aCoeffsXF += 8;
      }

      for (; j < aBorderBuf[i]; j++) {
        unsigned int px;
        memcpy(&px, aIn, 4);

        coeffsX = _mm_load_ps(aCoeffsXF);

        sampleX = _mm_set1_ps(lut[px & 0xFF]);
        sumR = _mm_fmadd_ps(coeffsX, sampleX, sumR);

        sampleX = _mm_set1_ps(lut[(px >> 8) & 0xFF]);
        sumG = _mm_fmadd_ps(coeffsX, sampleX, sumG);

        sampleX = _mm_set1_ps(lut[(px >> 16) & 0xFF]);
        sumB = _mm_fmadd_ps(coeffsX, sampleX, sumB);

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
        sumR = _mm_fmadd_ps(coeffsX, sampleX, sumR);

        sampleX = _mm_set1_ps(lut[aIn[1]]);
        sumG = _mm_fmadd_ps(coeffsX, sampleX, sumG);

        sampleX = _mm_set1_ps(lut[aIn[2]]);
        sumB = _mm_fmadd_ps(coeffsX, sampleX, sumB);

        aIn += 4;
        aCoeffsXF += 4;
      }
    }

    /* Vertical accumulation using 256-bit AVX2 */
    {
      __m128 rg, bx, rgbx;
      __m256 rgbx256, sy;

      /* Prefetch next pixel's sums_y */
      _mm_prefetch(reinterpret_cast<const char*>(aSumsYOut + 16), _MM_HINT_T0);

      rg = _mm_unpacklo_ps(sumR, sumG);
      bx = _mm_unpacklo_ps(sumB, sumB);
      rgbx = _mm_movelh_ps(rg, bx);

      rgbx256 = _mm256_set_m128(rgbx, rgbx);

      sy = _mm256_loadu_ps(aSumsYOut);
      sy = _mm256_fmadd_ps(cyLo, rgbx256, sy);
      _mm256_storeu_ps(aSumsYOut, sy);

      sy = _mm256_loadu_ps(aSumsYOut + 8);
      sy = _mm256_fmadd_ps(cyHi, rgbx256, sy);
      _mm256_storeu_ps(aSumsYOut + 8, sy);

      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
  }
}

static void OilYscaleOutRgbaNogammaAvx2(float* aSums, int aWidth,
                                        unsigned char* aOut, int aTap) {
  int i, tapOff;
  __m128 scale, half, one, zero;
  __m128 vals, alphaV;
  __m128i idx, packed;
  __m128i z;

  tapOff = aTap * 4;
  scale = _mm_set1_ps(255.0f);
  half = _mm_set1_ps(0.5f);
  one = _mm_set1_ps(1.0f);
  zero = _mm_setzero_ps();
  z = _mm_setzero_si128();

  for (i = 0; i + 3 < aWidth; i += 4) {
    __m128i idx2, idx3, idx4;
    __m128 vals2, vals3, vals4, alphaV2, alphaV3, alphaV4;
    __m128i packed2;

    /* Pixel 1 */
    vals = _mm_load_ps(aSums + tapOff);
    alphaV = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV = _mm_min_ps(_mm_max_ps(alphaV, zero), one);
    if (_mm_cvtss_f32(alphaV) != 0) {
      vals = _mm_mul_ps(vals, _mm_rcp_ps(alphaV));
    }
    vals = _mm_min_ps(_mm_max_ps(vals, zero), one);
    {
      __m128 hi = _mm_shuffle_ps(vals, alphaV, _MM_SHUFFLE(0, 0, 2, 2));
      vals = _mm_shuffle_ps(vals, hi, _MM_SHUFFLE(2, 0, 1, 0));
    }
    idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + tapOff), z);

    /* Pixel 2 */
    vals2 = _mm_load_ps(aSums + 16 + tapOff);
    alphaV2 = _mm_shuffle_ps(vals2, vals2, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV2 = _mm_min_ps(_mm_max_ps(alphaV2, zero), one);
    if (_mm_cvtss_f32(alphaV2) != 0) {
      vals2 = _mm_mul_ps(vals2, _mm_rcp_ps(alphaV2));
    }
    vals2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);
    {
      __m128 hi2 = _mm_shuffle_ps(vals2, alphaV2, _MM_SHUFFLE(0, 0, 2, 2));
      vals2 = _mm_shuffle_ps(vals2, hi2, _MM_SHUFFLE(2, 0, 1, 0));
    }
    idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals2, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 16 + tapOff), z);

    packed = _mm_packs_epi32(idx, idx2);

    /* Pixel 3 */
    vals3 = _mm_load_ps(aSums + 32 + tapOff);
    alphaV3 = _mm_shuffle_ps(vals3, vals3, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV3 = _mm_min_ps(_mm_max_ps(alphaV3, zero), one);
    if (_mm_cvtss_f32(alphaV3) != 0) {
      vals3 = _mm_mul_ps(vals3, _mm_rcp_ps(alphaV3));
    }
    vals3 = _mm_min_ps(_mm_max_ps(vals3, zero), one);
    {
      __m128 hi3 = _mm_shuffle_ps(vals3, alphaV3, _MM_SHUFFLE(0, 0, 2, 2));
      vals3 = _mm_shuffle_ps(vals3, hi3, _MM_SHUFFLE(2, 0, 1, 0));
    }
    idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals3, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 32 + tapOff), z);

    /* Pixel 4 */
    vals4 = _mm_load_ps(aSums + 48 + tapOff);
    alphaV4 = _mm_shuffle_ps(vals4, vals4, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV4 = _mm_min_ps(_mm_max_ps(alphaV4, zero), one);
    if (_mm_cvtss_f32(alphaV4) != 0) {
      vals4 = _mm_mul_ps(vals4, _mm_rcp_ps(alphaV4));
    }
    vals4 = _mm_min_ps(_mm_max_ps(vals4, zero), one);
    {
      __m128 hi4 = _mm_shuffle_ps(vals4, alphaV4, _MM_SHUFFLE(0, 0, 2, 2));
      vals4 = _mm_shuffle_ps(vals4, hi4, _MM_SHUFFLE(2, 0, 1, 0));
    }
    idx4 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(vals4, scale), half));
    _mm_store_si128(reinterpret_cast<__m128i*>(aSums + 48 + tapOff), z);

    packed2 = _mm_packs_epi32(idx3, idx4);
    packed = _mm_packus_epi16(packed, packed2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(aOut), packed);

    aSums += 64;
    aOut += 16;
  }

  for (; i < aWidth; i++) {
    vals = _mm_load_ps(aSums + tapOff);

    alphaV = _mm_shuffle_ps(vals, vals, _MM_SHUFFLE(3, 3, 3, 3));
    alphaV = _mm_min_ps(_mm_max_ps(alphaV, zero), one);
    if (_mm_cvtss_f32(alphaV) != 0) {
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

static void OilScaleDownRgbaNogammaAvx2(unsigned char* aIn, float* aSumsYOut,
                                        int aOutWidth, float* aCoeffsXF,
                                        int* aBorderBuf, float* aCoeffsYF,
                                        int aTap) {
  int i, j;
  __m128 coeffsX, coeffsX2, coeffsXA, coeffsX2A, sampleX;
  __m128 sumR, sumG, sumB, sumA;
  __m128 sumR2, sumG2, sumB2, sumA2;
  float* lut;
  __m256 cy256Lo, cy256Hi;
  {
    float cyPhys[4];
    cyPhys[aTap & 3] = aCoeffsYF[0];
    cyPhys[(aTap + 1) & 3] = aCoeffsYF[1];
    cyPhys[(aTap + 2) & 3] = aCoeffsYF[2];
    cyPhys[(aTap + 3) & 3] = aCoeffsYF[3];
    cy256Lo = _mm256_set_m128(_mm_set1_ps(cyPhys[1]), _mm_set1_ps(cyPhys[0]));
    cy256Hi = _mm256_set_m128(_mm_set1_ps(cyPhys[3]), _mm_set1_ps(cyPhys[2]));
  }

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
        __m256 rgba256, syLo, syHi;
        rgba256 = _mm256_set_m128(rgba, rgba);
        syLo = _mm256_loadu_ps(aSumsYOut);
        syHi = _mm256_loadu_ps(aSumsYOut + 8);
        syLo = _mm256_fmadd_ps(cy256Lo, rgba256, syLo);
        syHi = _mm256_fmadd_ps(cy256Hi, rgba256, syHi);
        _mm256_storeu_ps(aSumsYOut, syLo);
        _mm256_storeu_ps(aSumsYOut + 8, syHi);
      }
      aSumsYOut += 16;
    }

    sumR = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumR), 4));
    sumG = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumG), 4));
    sumB = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumB), 4));
    sumA = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sumA), 4));
  }
}

/* AVX2 dispatch functions */

static void YscaleOutAvx2(float* aSums, int aWidth, unsigned char* aOut,
                          OilColorspace aCs, int aTap) {
  switch (aCs) {
    case OilColorspace::RgbaNogamma:
      OilYscaleOutRgbaNogammaAvx2(aSums, aWidth, aOut, aTap);
      break;
    case OilColorspace::RgbxNogamma:
      OilYscaleOutRgbxNogammaAvx2(aSums, aWidth, aOut, aTap);
      break;
  }
}

static void DownScaleInAvx2(OilScale* aOs, unsigned char* aIn) {
  float* coeffsY;

  coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  switch (aOs->mCs) {
    case OilColorspace::RgbaNogamma:
      OilScaleDownRgbaNogammaAvx2(aIn, aOs->mSumsY, aOs->mOutWidth,
                                  aOs->mCoeffsX, aOs->mBordersX, coeffsY,
                                  aOs->mSumsYTap);
      break;
    case OilColorspace::RgbxNogamma:
      OilScaleDownRgbxNogammaAvx2(aIn, aOs->mSumsY, aOs->mOutWidth,
                                  aOs->mCoeffsX, aOs->mBordersX, coeffsY,
                                  aOs->mSumsYTap);
      break;
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

int OilScaleInAvx2(OilScale* aOs, unsigned char* aIn) {
  if (OilScaleSlots(aOs) == 0) {
    return -1;
  }
  DownScaleInAvx2(aOs, aIn);
  return 0;
}

int OilScaleOutAvx2(OilScale* aOs, unsigned char* aOut) {
  if (OilScaleSlots(aOs) != 0) {
    return -1;
  }

  YscaleOutAvx2(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mCs, aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;

  aOs->mOutPos++;
  return 0;
}

}  // namespace mozilla
