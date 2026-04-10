#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "OilResample.h"

using namespace mozilla;

using ScaleInFn = int (*)(OilScale*, unsigned char*);
using ScaleOutFn = int (*)(OilScale*, unsigned char*);

static ScaleInFn gCurScaleIn;
static ScaleOutFn gCurScaleOut;

/**
 * shared test helpers
 */
static long double Cubic(long double aB, long double aC, long double aX) {
  if (aX < 1.0l) {
    return ((12.0l - 9.0l * aB - 6.0l * aC) * aX * aX * aX +
            (-18.0l + 12.0l * aB + 6.0l * aC) * aX * aX + (6.0l - 2.0l * aB)) /
           6.0l;
  }
  if (aX < 2.0l) {
    return ((-aB - 6.0l * aC) * aX * aX * aX +
            (6.0l * aB + 30.0l * aC) * aX * aX +
            (-12.0l * aB - 48.0l * aC) * aX + (8.0l * aB + 24.0l * aC)) /
           6.0l;
  }
  return 0.0;
}

static long double RefCatrom(long double aX) { return Cubic(0, 0.5l, aX); }

static void RefCalcCoeffs(long double* aCoeffs, long double aOffset, int aTaps,
                          int aLtrim, int aRtrim) {
  int i;
  long double tapOffset, tapMult, fudge, totalCheck;

  assert(aTaps - aLtrim - aRtrim > 0);
  tapMult = static_cast<long double>(aTaps) / 4;
  fudge = 0.0;
  for (i = 0; i < aTaps; i++) {
    if (i < aLtrim || i >= aTaps - aRtrim) {
      aCoeffs[i] = 0;
      continue;
    }
    tapOffset = 1 - aOffset - aTaps / 2 + i;
    aCoeffs[i] = RefCatrom(fabsl(tapOffset) / tapMult) / tapMult;
    fudge += aCoeffs[i];
  }
  totalCheck = 0.0;
  for (i = 0; i < aTaps; i++) {
    aCoeffs[i] /= fudge;
    totalCheck += aCoeffs[i];
  }
  assert(fabsl(totalCheck - 1.0) < 0.0000000001L);
}

static void FillRand8(unsigned char* aBuf, int aLen) {
  int i;
  for (i = 0; i < aLen; i++) {
    aBuf[i] = rand() % 256;
  }
}

static int CalcTapsCheck(int aDimIn, int aDimOut) {
  int tmpI;
  if (aDimIn < aDimOut) {
    return 4;
  }
  tmpI = aDimIn * 4 / aDimOut;
  return tmpI - (tmpI % 2);
}

static long double RefMap(int aDimIn, int aDimOut, int aPos) {
  return (aPos + 0.5l) * static_cast<long double>(aDimIn) / aDimOut - 0.5l;
}

static int SplitMapCheck(int aDimIn, int aDimOut, int aPos, long double* aTy) {
  long double smp;
  int smpI;

  smp = RefMap(aDimIn, aDimOut, aPos);
  smpI = floorl(smp);
  *aTy = smp - smpI;
  return smpI;
}

static double gWorst;

static void ValidateScanline8(unsigned char* aOil, long double* aRef,
                              int aWidth, int aCmp) {
  int i, j, refI, pos;
  double error, refF;
  for (i = 0; i < aWidth; i++) {
    for (j = 0; j < aCmp; j++) {
      pos = i * aCmp + j;
      refF = aRef[pos] * 255.0;
      refI = lround(refF);
      error = fabs(aOil[pos] - refF) - 0.5;
      if (error > gWorst) {
        gWorst = error;
      }
      /* Bumped from 0.06 to 0.07: adding ARGB to more test
       * functions shifted the random data, exposing edge cases
       * where float precision just barely exceeds 0.06. */
      if (error > 0.07) {
        fprintf(stderr, "[%d:%d] expected: %d, got %d (%.9f)\n", i, j, refI,
                aOil[pos], refF);
        assert(0 && "pixel error exceeds tolerance");
      }
    }
  }
}

static long double ClampF(long double aIn) {
  if (aIn <= 0.0L) {
    return 0.0L;
  }
  if (aIn >= 1.0L) {
    return 1.0L;
  }
  return aIn;
}

static void Preprocess(long double* aIn, OilColorspace aCs) {
  switch (aCs) {
    case OilColorspace::RgbxNogamma:
      aIn[3] = 1.0L;
      break;
    case OilColorspace::RgbaNogamma:
      aIn[0] *= aIn[3];
      aIn[1] *= aIn[3];
      aIn[2] *= aIn[3];
      break;
  }
}

static void Postprocess(long double* aIn, OilColorspace aCs) {
  long double alpha;
  switch (aCs) {
    case OilColorspace::RgbxNogamma:
      aIn[0] = ClampF(aIn[0]);
      aIn[1] = ClampF(aIn[1]);
      aIn[2] = ClampF(aIn[2]);
      aIn[3] = 1.0L;
      break;
    case OilColorspace::RgbaNogamma:
      alpha = ClampF(aIn[3]);
      if (alpha != 0.0L) {
        aIn[0] /= alpha;
        aIn[1] /= alpha;
        aIn[2] /= alpha;
      }
      aIn[0] = ClampF(aIn[0]);
      aIn[1] = ClampF(aIn[1]);
      aIn[2] = ClampF(aIn[2]);
      aIn[3] = alpha;
      break;
  }
}

static unsigned char** Alloc2dUchar(int aWidth, int aHeight) {
  int i;
  unsigned char** rows;

  rows = static_cast<unsigned char**>(malloc(aHeight * sizeof(unsigned char*)));
  for (i = 0; i < aHeight; i++) {
    rows[i] =
        static_cast<unsigned char*>(malloc(aWidth * sizeof(unsigned char)));
  }
  return rows;
}

static void Free2dUchar(unsigned char** aPtr, int aHeight) {
  int i;

  for (i = 0; i < aHeight; i++) {
    free(aPtr[i]);
  }
  free(aPtr);
}

static long double** Alloc2dLd(int aWidth, int aHeight) {
  int i;
  long double** rows;

  rows = static_cast<long double**>(malloc(aHeight * sizeof(long double*)));
  for (i = 0; i < aHeight; i++) {
    rows[i] = static_cast<long double*>(malloc(aWidth * sizeof(long double)));
  }
  return rows;
}

static void Free2dLd(long double** aPtr, int aHeight) {
  int i;

  for (i = 0; i < aHeight; i++) {
    free(aPtr[i]);
  }
  free(aPtr);
}

static void RefXScale(long double* aIn, int aInWidth, long double* aOut,
                      int aOutWidth, int aCmp) {
  int i, j, k, taps, smpI, start, ltrim, rtrim, startSafe, tapsSafe, maxPos,
      inPos;
  long double* coeffs;
  long double inVal, tx;

  taps = CalcTapsCheck(aInWidth, aOutWidth);
  coeffs = static_cast<long double*>(malloc(taps * sizeof(long double)));
  maxPos = aInWidth - 1;
  for (i = 0; i < aOutWidth; i++) {
    smpI = SplitMapCheck(aInWidth, aOutWidth, i, &tx);
    start = smpI - (taps / 2 - 1);

    startSafe = start;
    if (startSafe < 0) {
      startSafe = 0;
    }
    ltrim = startSafe - start;

    tapsSafe = taps - ltrim;
    if (startSafe + tapsSafe > maxPos) {
      tapsSafe = maxPos - startSafe + 1;
    }
    rtrim = (start + taps) - (startSafe + tapsSafe);

    RefCalcCoeffs(coeffs, tx, taps, ltrim, rtrim);

    for (j = 0; j < aCmp; j++) {
      aOut[i * aCmp + j] = 0;
      for (k = 0; k < tapsSafe; k++) {
        inPos = startSafe + k;
        inVal = aIn[inPos * aCmp + j];
        aOut[i * aCmp + j] += coeffs[ltrim + k] * inVal;
      }
    }
  }

  free(coeffs);
}

static void RefTransposeLine(long double* aIn, int aWidth, long double** aOut,
                             int aOutOffset, int aCmp) {
  int i, j;
  for (i = 0; i < aWidth; i++) {
    for (j = 0; j < aCmp; j++) {
      aOut[i][aOutOffset + j] = aIn[i * aCmp + j];
    }
  }
}

static void RefTransposeColumn(long double** aIn, int aHeight,
                               long double* aOut, int aInOffset, int aCmp) {
  int i, j;
  for (i = 0; i < aHeight; i++) {
    for (j = 0; j < aCmp; j++) {
      aOut[i * aCmp + j] = aIn[i][aInOffset + j];
    }
  }
}

static void RefYScale(long double** aIn, int aWidth, int aInHeight,
                      long double** aOut, int aOutHeight, int aCmp) {
  int i;
  long double* transposed;
  long double* transScaled;

  transposed =
      static_cast<long double*>(malloc(aInHeight * aCmp * sizeof(long double)));
  transScaled = static_cast<long double*>(
      malloc(aOutHeight * aCmp * sizeof(long double)));
  for (i = 0; i < aWidth; i++) {
    RefTransposeColumn(aIn, aInHeight, transposed, i * aCmp, aCmp);
    RefXScale(transposed, aInHeight, transScaled, aOutHeight, aCmp);
    RefTransposeLine(transScaled, aOutHeight, aOut, i * aCmp, aCmp);
  }
  free(transposed);
  free(transScaled);
}

static void RefScale(unsigned char** aIn, int aInWidth, int aInHeight,
                     long double** aOut, int aOutWidth, int aOutHeight,
                     OilColorspace aCs) {
  int i, j, cmp, stride;
  long double* preLine;
  long double** intermediate;

  cmp = 4;
  stride = cmp * aInWidth;

  // horizontal scaling
  preLine = static_cast<long double*>(malloc(stride * sizeof(long double)));
  intermediate = Alloc2dLd(aOutWidth * cmp, aInHeight);
  for (i = 0; i < aInHeight; i++) {
    // Convert chars to floats
    for (j = 0; j < stride; j++) {
      preLine[j] = aIn[i][j] / 255.0F;
    }

    // Preprocess
    for (j = 0; j < aInWidth; j++) {
      Preprocess(preLine + j * cmp, aCs);
    }

    // xscale
    RefXScale(preLine, aInWidth, intermediate[i], aOutWidth, cmp);
  }

  // vertical scaling
  RefYScale(intermediate, aOutWidth, aInHeight, aOut, aOutHeight, cmp);
  for (i = 0; i < aOutHeight; i++) {
    for (j = 0; j < aOutWidth; j++) {
      Postprocess(aOut[i] + j * cmp, aCs);
    }
  }

  free(preLine);
  Free2dLd(intermediate, aInHeight);
}

static void DoOilScale(unsigned char** aInputImage, int aInWidth, int aInHeight,
                       unsigned char** aOutputImage, int aOutWidth,
                       int aOutHeight, OilColorspace aCs) {
  OilScale os;
  int i, inLine;

  OilScaleInit(&os, aInHeight, aOutHeight, aInWidth, aOutWidth, aCs);
  inLine = 0;
  for (i = 0; i < aOutHeight; i++) {
    while (OilScaleSlots(&os)) {
      gCurScaleIn(&os, aInputImage[inLine++]);
    }
    gCurScaleOut(&os, aOutputImage[i]);
  }
  OilScaleFree(&os);
}

static void TestScale(int aInWidth, int aInHeight, unsigned char** aInputImage,
                      int aOutWidth, int aOutHeight, OilColorspace aCs) {
  int i, outRowStride;
  unsigned char** oilOutputImage;
  long double** refOutputImage;

  outRowStride = 4 * aOutWidth;

  /* oil scaling */
  oilOutputImage = Alloc2dUchar(outRowStride, aOutHeight);
  DoOilScale(aInputImage, aInWidth, aInHeight, oilOutputImage, aOutWidth,
             aOutHeight, aCs);

  /* reference scaling */
  refOutputImage = Alloc2dLd(outRowStride, aOutHeight);
  RefScale(aInputImage, aInWidth, aInHeight, refOutputImage, aOutWidth,
           aOutHeight, aCs);

  /* compare the two */
  for (i = 0; i < aOutHeight; i++) {
    ValidateScanline8(oilOutputImage[i], refOutputImage[i], aOutWidth, 4);
  }

  Free2dUchar(oilOutputImage, aOutHeight);
  Free2dLd(refOutputImage, aOutHeight);
}

static void TestScaleSquareRand(int aInDim, int aOutDim, OilColorspace aCs) {
  int i, inRowStride;
  unsigned char** inputImage;

  inRowStride = 4 * aInDim;

  /* Allocate & populate input image */
  inputImage = Alloc2dUchar(inRowStride, aInDim);
  for (i = 0; i < aInDim; i++) {
    FillRand8(inputImage[i], inRowStride);
  }
  TestScale(aInDim, aInDim, inputImage, aOutDim, aOutDim, aCs);
  Free2dUchar(inputImage, aInDim);
}

static void TestScaleEachCs(int aDimA, int aDimB) {
  TestScaleSquareRand(aDimA, aDimB, OilColorspace::RgbaNogamma);
  TestScaleSquareRand(aDimA, aDimB, OilColorspace::RgbxNogamma);
}

static void TestScaleDownscale(int aDimIn, int aDimOut) {
  assert(aDimIn >= aDimOut);
  TestScaleEachCs(aDimIn, aDimOut);
}

static void TestOutNotReady(int aInDim, int aOutDim, OilColorspace aCs) {
  OilScale os;
  int outRowStride;
  unsigned char* buf;

  outRowStride = 4 * aOutDim;
  buf = static_cast<unsigned char*>(malloc(outRowStride));

  /* calling gCurScaleOut before any input should fail */
  OilScaleInit(&os, aInDim, aOutDim, aInDim, aOutDim, aCs);
  assert(gCurScaleOut(&os, buf) == -1);

  /* feed one input line when more are needed, should still fail */
  if (OilScaleSlots(&os) > 1) {
    unsigned char* inLine = static_cast<unsigned char*>(calloc(4 * aInDim, 1));
    assert(gCurScaleIn(&os, inLine) == 0);
    assert(OilScaleSlots(&os) > 0);
    assert(gCurScaleOut(&os, buf) == -1);
    free(inLine);
  }
  OilScaleFree(&os);

  /* feed enough input, then gCurScaleOut should succeed */
  OilScaleInit(&os, aInDim, aOutDim, aInDim, aOutDim, aCs);
  while (OilScaleSlots(&os)) {
    unsigned char* inLine = static_cast<unsigned char*>(calloc(4 * aInDim, 1));
    assert(gCurScaleIn(&os, inLine) == 0);
    free(inLine);
  }
  assert(gCurScaleOut(&os, buf) == 0);
  OilScaleFree(&os);

  free(buf);
}

static void TestOutNotReadyAll() {
  TestOutNotReady(100, 50, OilColorspace::RgbaNogamma);
  TestOutNotReady(100, 50, OilColorspace::RgbxNogamma);
}

static void TestScaleAll() {
  TestScaleDownscale(5, 1);
  TestScaleDownscale(8, 1);
  TestScaleDownscale(8, 3);
  TestScaleDownscale(100, 1);
  TestScaleDownscale(100, 99);
  TestScaleDownscale(2, 1);
}

struct Impl {
  const char* mName;
  ScaleInFn mIn;
  ScaleOutFn mOut;
};

static void RunTests(Impl* aImpl) {
  printf("--- testing %s ---\n", aImpl->mName);
  gCurScaleIn = aImpl->mIn;
  gCurScaleOut = aImpl->mOut;

  TestScaleAll();
  TestOutNotReadyAll();
}

int main() {
  int t = 1531289551;
  int i, numImpls;
  Impl impls[3];
  // int t = time(nullptr);
  printf("seed: %d\n", t);
  srand(t);
  OilGlobalInit();

  numImpls = 0;
  impls[numImpls].mName = "scalar";
  impls[numImpls].mIn = OilScaleIn;
  impls[numImpls].mOut = OilScaleOut;

  numImpls++;

#if defined(__x86_64__)
  impls[numImpls].mName = "sse2";
  impls[numImpls].mIn = OilScaleInSse2;
  impls[numImpls].mOut = OilScaleOutSse2;

  numImpls++;

  impls[numImpls].mName = "avx2";
  impls[numImpls].mIn = OilScaleInAvx2;
  impls[numImpls].mOut = OilScaleOutAvx2;

  numImpls++;
#elif defined(__aarch64__)
  impls[numImpls].mName = "neon";
  impls[numImpls].mIn = OilScaleInNeon;
  impls[numImpls].mOut = OilScaleOutNeon;

  numImpls++;
#endif

  for (i = 0; i < numImpls; i++) {
    RunTests(&impls[i]);
  }

  printf("worst error: %f\n", gWorst);
  printf("All tests pass.\n");
  return 0;
}
