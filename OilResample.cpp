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
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace mozilla {

/**
 * When shrinking a 10 million pixel wide scanline down to a single pixel, we
 * reach the limits of single-precision floats. Limit input dimensions to one
 * million by one million pixels to avoid this issue as well as overflow issues
 * with 32-bit ints.
 */
static constexpr int kMaxDimension = 1000000;

/**
 * Bicubic interpolation. 2 base taps on either side.
 */
static constexpr int kTaps = 4;

static int Max(int aA, int aB) { return aA > aB ? aA : aB; }

static int Min(int aA, int aB) { return aA < aB ? aA : aB; }

/**
 * Clamp a float between 0 and 1.
 */
static float ClampF(float aX) {
  if (aX > 1.0f) {
    return 1.0f;
  } else if (aX < 0.0f) {
    return 0.0f;
  }
  return aX;
}

/**
 * Convert a float to an int. When compiling on x86 without march=native, this
 * performs much better than roundf().
 */
static int F2I(float aX) { return aX + 0.5f; }

/**
 * Map from the discreet dest coordinate pos to a continuous source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
static double Map(int aDimIn, int aDimOut, int aPos) {
  return (aPos + 0.5) * (static_cast<double>(aDimIn) / aDimOut) - 0.5;
}

/**
 * Returns the mapped input position and put the sub-pixel remainder in rest.
 */
static int SplitMap(int aDimIn, int aDimOut, int aPos, float* aRest) {
  double smp;
  int smpI;

  smp = Map(aDimIn, aDimOut, aPos);
  smpI = smp < 0 ? -1 : smp;
  *aRest = smp - smpI;
  return smpI;
}

/**
 * Given input and output dimension, calculate the total number of taps that
 * will be needed to calculate an output sample.
 *
 * When we reduce an image by a factor of two, we need to scale our resampling
 * function by two as well in order to avoid aliasing.
 */
static int CalcTaps(int aDimIn, int aDimOut) {
  int tmp;
  if (aDimOut > aDimIn) {
    return kTaps;
  }
  tmp = kTaps * aDimIn / aDimOut;
  return tmp - (tmp & 1);
}

/**
 * Catmull-Rom interpolator.
 */
static float Catrom(float aX) {
  if (aX < 1) {
    return (1.5f * aX - 2.5f) * aX * aX + 1;
  }
  return (((5 - aX) * aX - 8) * aX + 4) / 2;
}

/**
 * Given an offset tx, calculate taps coefficients.
 */
static void CalcCoeffs(float* aCoeffs, float aTx, int aTaps, int aLtrim,
                       int aRtrim) {
  int i;
  float tmp, tapMult, fudge;

  tapMult = static_cast<float>(aTaps) / kTaps;
  aTx = 1 - aTx - aTaps / 2 + aLtrim;
  fudge = 0.0f;

  for (i = aLtrim; i < aTaps - aRtrim; i++) {
    tmp = Catrom(fabsf(aTx) / tapMult) / tapMult;
    fudge += tmp;
    aCoeffs[i] = tmp;
    aTx += 1;
  }
  fudge = 1 / fudge;
  for (i = aLtrim; i < aTaps - aRtrim; i++) {
    aCoeffs[i] *= fudge;
  }
}

/**
 * Takes a sample value, an array of 4 coefficients & 4 accumulators, and
 * adds the product of sample * coeffs[n] to each accumulator.
 */
static void AddSampleToSumF(float aSample, float* aCoeffs, float* aSum) {
  int i;
  for (i = 0; i < 4; i++) {
    aSum[i] += aSample * aCoeffs[i];
  }
}

/**
 * Takes an array of 4 floats and shifts them left. The rightmost element is
 * set to 0.0.
 */
static void ShiftLeftF(float* aF) {
  aF[0] = aF[1];
  aF[1] = aF[2];
  aF[2] = aF[3];
  aF[3] = 0.0f;
}

static void YScaleOutRgbaNogamma(float* aSums, int aWidth, unsigned char* aOut,
                                 int aTap) {
  int i, j, tapOff;
  float alpha, val;

  tapOff = aTap * 4;
  for (i = 0; i < aWidth; i++) {
    alpha = ClampF(aSums[tapOff + 3]);
    for (j = 0; j < 3; j++) {
      val = aSums[tapOff + j];
      if (alpha != 0) {
        val /= alpha;
      }
      aOut[j] = F2I(ClampF(val) * 255.0f);
      aSums[tapOff + j] = 0.0f;
    }
    aOut[3] = round(alpha * 255.0f);
    aSums[tapOff + 3] = 0.0f;
    aSums += 16;
    aOut += 4;
  }
}

static void YScaleOutRgbxNogamma(float* aSums, int aWidth, unsigned char* aOut,
                                 int aTap) {
  int i, j, tapOff;

  tapOff = aTap * 4;
  for (i = 0; i < aWidth; i++) {
    for (j = 0; j < 3; j++) {
      aOut[j] = F2I(ClampF(aSums[tapOff + j]) * 255.0f);
      aSums[tapOff + j] = 0.0f;
    }
    aOut[3] = 255;
    aSums[tapOff + 3] = 0.0f;
    aSums += 16;
    aOut += 4;
  }
}

static void YScaleOut(float* aSums, int aWidth, unsigned char* aOut,
                      OilColorspace aCs, int aTap) {
  switch (aCs) {
    case OilColorspace::RgbaNogamma:
      YScaleOutRgbaNogamma(aSums, aWidth, aOut, aTap);
      break;
    case OilColorspace::RgbxNogamma:
      YScaleOutRgbxNogamma(aSums, aWidth, aOut, aTap);
      break;
  }
}

/* horizontal scaling */

float gI2fMap[256];

static void BuildI2f() {
  int i;

  for (i = 0; i <= 255; i++) {
    gI2fMap[i] = i / 255.0f;
  }
}

/**
 * Given input & output dimensions, populate a buffer of coefficients and
 * border counters.
 *
 * This method assumes that in_dim >= out_dim.
 *
 * It generates 4 * in_dim coefficients -- 4 for every input sample.
 *
 * It generates out_dim border counters, these indicate how many input samples
 * to process before the next output sample is finished.
 */
static void ScaleDownCoeffs(int aInDim, int aOutDim, float* aCoeffBuf,
                            int* aBorderBuf, float* aTmpCoeffs) {
  int smpI, i, j, taps, offset, pos, ltrim, rtrim, smpEnd, smpStart, ends[4];
  float tx;

  taps = CalcTaps(aInDim, aOutDim);
  for (i = 0; i < 4; i++) {
    ends[i] = -1;
  }

  for (i = 0; i < aOutDim; i++) {
    smpI = SplitMap(aInDim, aOutDim, i, &tx);

    smpStart = smpI - (taps / 2 - 1);
    smpEnd = smpI + taps / 2;
    if (smpEnd >= aInDim) {
      smpEnd = aInDim - 1;
    }
    ends[i % 4] = smpEnd;
    aBorderBuf[i] = smpEnd - ends[(i + 3) % 4];

    ltrim = 0;
    if (smpStart < 0) {
      ltrim = -1 * smpStart;
    }
    rtrim = smpStart + (taps - 1) - smpEnd;
    CalcCoeffs(aTmpCoeffs, tx, taps, ltrim, rtrim);

    for (j = ltrim; j < taps - rtrim; j++) {
      pos = smpStart + j;

      offset = 3;
      if (pos > ends[(i + 3) % 4]) {
        offset = 0;
      } else if (pos > ends[(i + 2) % 4]) {
        offset = 1;
      } else if (pos > ends[(i + 1) % 4]) {
        offset = 2;
      }

      aCoeffBuf[pos * 4 + offset] = aTmpCoeffs[j];
    }
  }
}

static void ScaleDownRgbaNogamma(unsigned char* aIn, float* aSumsY,
                                 int aOutWidth, float* aCoeffsX,
                                 int* aBorderBuf, float* aCoeffsY, int aTap) {
  int i, j, k;
  float alpha, sum[4][4] = {{0.0f}};

  for (i = 0; i < aOutWidth; i++) {
    for (j = 0; j < aBorderBuf[i]; j++) {
      alpha = gI2fMap[aIn[3]];
      for (k = 0; k < 3; k++) {
        AddSampleToSumF(gI2fMap[aIn[k]] * alpha, aCoeffsX, sum[k]);
      }
      AddSampleToSumF(alpha, aCoeffsX, sum[3]);
      aIn += 4;
      aCoeffsX += 4;
    }

    {
      float samples[4];
      for (j = 0; j < 4; j++) {
        samples[j] = sum[j][0];
        ShiftLeftF(sum[j]);
      }
      for (j = 0; j < 4; j++) {
        float cy = aCoeffsY[j];
        int off = ((aTap + j) & 3) * 4;
        aSumsY[off + 0] += samples[0] * cy;
        aSumsY[off + 1] += samples[1] * cy;
        aSumsY[off + 2] += samples[2] * cy;
        aSumsY[off + 3] += samples[3] * cy;
      }
      aSumsY += 16;
    }
  }
}

static void ScaleDownRgbxNogamma(unsigned char* aIn, float* aSumsY,
                                 int aOutWidth, float* aCoeffsX,
                                 int* aBorderBuf, float* aCoeffsY, int aTap) {
  int i, j, k;
  float sum[4][4] = {{0.0f}};

  for (i = 0; i < aOutWidth; i++) {
    for (j = 0; j < aBorderBuf[i]; j++) {
      for (k = 0; k < 3; k++) {
        AddSampleToSumF(gI2fMap[aIn[k]], aCoeffsX, sum[k]);
      }
      AddSampleToSumF(1.0f, aCoeffsX, sum[3]);
      aIn += 4;
      aCoeffsX += 4;
    }

    {
      float samples[4];
      for (j = 0; j < 4; j++) {
        samples[j] = sum[j][0];
        ShiftLeftF(sum[j]);
      }
      for (j = 0; j < 4; j++) {
        float cy = aCoeffsY[j];
        int off = ((aTap + j) & 3) * 4;
        aSumsY[off + 0] += samples[0] * cy;
        aSumsY[off + 1] += samples[1] * cy;
        aSumsY[off + 2] += samples[2] * cy;
        aSumsY[off + 3] += samples[3] * cy;
      }
      aSumsY += 16;
    }
  }
}

/* Global functions */
void OilGlobalInit() { BuildI2f(); }

static constexpr int Align16(int aX) { return (aX + 15) & ~15; }

static int CalcCoeffsLen(int aInDim, int aOutDim) {
  return kTaps * Max(aInDim, aOutDim) * sizeof(float);
}

static int CalcBordersLen(int aInDim, int aOutDim) {
  return Min(aInDim, aOutDim) * sizeof(int);
}

static int DownscaleAllocSize(int aInHeight, int aOutHeight, int aInWidth,
                              int aOutWidth, OilColorspace aCs) {
  int tapsX, tapsY;

  tapsX = CalcTaps(aInWidth, aOutWidth);
  tapsY = CalcTaps(aInHeight, aOutHeight);

  return Align16(CalcCoeffsLen(aInWidth, aOutWidth)) +
         Align16(CalcBordersLen(aInWidth, aOutWidth)) +
         Align16(CalcCoeffsLen(aInHeight, aOutHeight)) +
         Align16(CalcBordersLen(aInHeight, aOutHeight)) +
         Align16(Max(tapsX, tapsY) * sizeof(float)) +
         Align16(aOutWidth * 4 * kTaps * sizeof(float));
}

static void DownscaleInit(OilScale* aOs) {
  int coeffsXLen, coeffsYLen, bordersXLen, bordersYLen, sumsLen;
  char* p;

  coeffsXLen = Align16(CalcCoeffsLen(aOs->mInWidth, aOs->mOutWidth));
  bordersXLen = Align16(CalcBordersLen(aOs->mInWidth, aOs->mOutWidth));
  coeffsYLen = Align16(CalcCoeffsLen(aOs->mInHeight, aOs->mOutHeight));
  bordersYLen = Align16(CalcBordersLen(aOs->mInHeight, aOs->mOutHeight));
  sumsLen = Align16(aOs->mOutWidth * 4 * kTaps * sizeof(float));

  p = static_cast<char*>(aOs->mBuf);
  aOs->mCoeffsX = reinterpret_cast<float*>(p);
  p += coeffsXLen;
  aOs->mBordersX = reinterpret_cast<int*>(p);
  p += bordersXLen;
  aOs->mCoeffsY = reinterpret_cast<float*>(p);
  p += coeffsYLen;
  aOs->mBordersY = reinterpret_cast<int*>(p);
  p += bordersYLen;
  aOs->mSumsY = reinterpret_cast<float*>(p);
  p += sumsLen;
  aOs->mTmpCoeffs = reinterpret_cast<float*>(p);

  ScaleDownCoeffs(aOs->mInWidth, aOs->mOutWidth, aOs->mCoeffsX, aOs->mBordersX,
                  aOs->mTmpCoeffs);
  ScaleDownCoeffs(aOs->mInHeight, aOs->mOutHeight, aOs->mCoeffsY,
                  aOs->mBordersY, aOs->mTmpCoeffs);
}

int OilScaleAllocSize(int aInHeight, int aOutHeight, int aInWidth,
                      int aOutWidth, OilColorspace aCs) {
  return DownscaleAllocSize(aInHeight, aOutHeight, aInWidth, aOutWidth, aCs);
}

int OilScaleInitAllocated(OilScale* aOs, int aInHeight, int aOutHeight,
                          int aInWidth, int aOutWidth, OilColorspace aCs,
                          void* aBuf) {
  /* sanity check on arguments */
  if (!aOs || !aBuf || aInHeight > kMaxDimension ||
      aOutHeight > kMaxDimension || aInHeight < 1 || aOutHeight < 1 ||
      aInWidth > kMaxDimension || aOutWidth > kMaxDimension || aInWidth < 1 ||
      aOutWidth < 1) {
    return -1;
  }

  /* only downscaling is supported */
  if (aOutHeight > aInHeight || aOutWidth > aInWidth) {
    return -1;
  }

  // Lazy perform global init, in case OilGlobalInit() hasn't been
  // called yet.
  if (!gI2fMap[128]) {
    OilGlobalInit();
  }

  memset(aOs, 0, sizeof(OilScale));
  aOs->mInHeight = aInHeight;
  aOs->mOutHeight = aOutHeight;
  aOs->mInWidth = aInWidth;
  aOs->mOutWidth = aOutWidth;
  aOs->mCs = aCs;
  aOs->mBuf = aBuf;

  DownscaleInit(aOs);

  return 0;
}

int OilScaleInit(OilScale* aOs, int aInHeight, int aOutHeight, int aInWidth,
                 int aOutWidth, OilColorspace aCs) {
  int allocSize, ret;
  void* buf;

  allocSize =
      OilScaleAllocSize(aInHeight, aOutHeight, aInWidth, aOutWidth, aCs);
  buf = calloc(1, allocSize);
  if (!buf) {
    return -2;
  }

  ret = OilScaleInitAllocated(aOs, aInHeight, aOutHeight, aInWidth, aOutWidth,
                              aCs, buf);
  if (ret) {
    free(buf);
    return ret;
  }

  return 0;
}

void OilScaleRestart(OilScale* aOs) {
  aOs->mInPos = aOs->mOutPos = 0;
  aOs->mSumsYTap = 0;
}

void OilScaleFree(OilScale* aOs) {
  if (!aOs) {
    return;
  }

  free(aOs->mBuf);
  aOs->mBuf = nullptr;
  aOs->mCoeffsX = nullptr;
  aOs->mBordersX = nullptr;
  aOs->mCoeffsY = nullptr;
  aOs->mBordersY = nullptr;
  aOs->mSumsY = nullptr;
  aOs->mTmpCoeffs = nullptr;
}

int OilScaleSlots(OilScale* aOs) { return aOs->mBordersY[aOs->mOutPos]; }

static void DownScaleIn(OilScale* aOs, unsigned char* aIn) {
  float* coeffsY;

  coeffsY = aOs->mCoeffsY + aOs->mInPos * 4;

  switch (aOs->mCs) {
    case OilColorspace::RgbaNogamma:
      ScaleDownRgbaNogamma(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                           aOs->mBordersX, coeffsY, aOs->mSumsYTap);
      break;
    case OilColorspace::RgbxNogamma:
      ScaleDownRgbxNogamma(aIn, aOs->mSumsY, aOs->mOutWidth, aOs->mCoeffsX,
                           aOs->mBordersX, coeffsY, aOs->mSumsYTap);
      break;
  }

  aOs->mBordersY[aOs->mOutPos] -= 1;
  aOs->mInPos++;
}

int OilScaleIn(OilScale* aOs, unsigned char* aIn) {
  if (OilScaleSlots(aOs) == 0) {
    return -1;
  }
  DownScaleIn(aOs, aIn);
  return 0;
}

int OilScaleOut(OilScale* aOs, unsigned char* aOut) {
  if (OilScaleSlots(aOs) != 0) {
    return -1;
  }

  YScaleOut(aOs->mSumsY, aOs->mOutWidth, aOut, aOs->mCs, aOs->mSumsYTap);
  aOs->mSumsYTap = (aOs->mSumsYTap + 1) & 3;

  aOs->mOutPos++;
  return 0;
}

}  // namespace mozilla
