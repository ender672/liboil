#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <png.h>
#include "OilResample.h"

using namespace mozilla;

using ScaleInFn = int (*)(OilScale*, unsigned char*);
using ScaleOutFn = int (*)(OilScale*, unsigned char*);

struct BenchImage {
  unsigned char* mBuffer;
  int mWidth;
  int mHeight;
  OilColorspace mCs;
};

static BenchImage LoadPng(char* aPath, OilColorspace aCs) {
  BenchImage image;
  int i;
  png_structp rpng;
  png_infop rinfo;
  FILE* input;
  size_t rowStride, bufLen;
  unsigned char** bufPtrs;

  rpng =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!rpng) {
    fprintf(stderr, "Unable to create PNG read struct.\n");
    exit(1);
  }
  if (setjmp(png_jmpbuf(rpng))) {
    fprintf(stderr, "PNG Decoding Error.\n");
    exit(1);
  }

  input = fopen(aPath, "rb");
  if (!input) {
    fprintf(stderr, "Unable to open file: %s\n", aPath);
    exit(1);
  }

  rinfo = png_create_info_struct(rpng);
  png_init_io(rpng, input);
  png_read_info(rpng, rinfo);

  if (png_get_color_type(rpng, rinfo) != PNG_COLOR_TYPE_RGBA) {
    fprintf(stderr, "Input image must be RGBA.\n");
    exit(1);
  }

  switch (aCs) {
    case OilColorspace::RgbxNogamma:
      png_set_strip_alpha(rpng);
      png_set_filler(rpng, 0xffff, PNG_FILLER_AFTER);
      break;
    case OilColorspace::RgbaNogamma:
      break;
  }

  image.mWidth = png_get_image_width(rpng, rinfo);
  image.mHeight = png_get_image_height(rpng, rinfo);
  image.mCs = aCs;

  rowStride = image.mWidth * 4;
  bufLen = static_cast<size_t>(image.mHeight) * rowStride;
  image.mBuffer = static_cast<unsigned char*>(malloc(bufLen));
  bufPtrs = static_cast<unsigned char**>(
      malloc(image.mHeight * sizeof(unsigned char*)));
  if (!image.mBuffer || !bufPtrs) {
    fprintf(stderr, "Unable to allocate buffers.\n");
    exit(1);
  }

  for (i = 0; i < image.mHeight; i++) {
    bufPtrs[i] = image.mBuffer + i * rowStride;
  }

  png_read_image(rpng, bufPtrs);
  png_destroy_read_struct(&rpng, &rinfo, nullptr);

  free(bufPtrs);
  fclose(input);
  return image;
}

static double TimeToMs(clock_t aT) {
  return static_cast<double>(aT) * 1000.0 / CLOCKS_PER_SEC;
}

static clock_t Resize(BenchImage aImage, int aOutWidth, int aOutHeight,
                      ScaleInFn aDoIn, ScaleOutFn aDoOut) {
  int i;
  OilColorspace cs;
  OilScale os;
  unsigned char* inbuf;
  unsigned char* outbuf;
  size_t inRowStride;
  clock_t t;

  cs = aImage.mCs;
  inRowStride = aImage.mWidth * 4;

  inbuf = aImage.mBuffer;
  outbuf = static_cast<unsigned char*>(malloc(aOutWidth * 4));
  if (!outbuf) {
    fprintf(stderr, "Unable to allocate output buffer.\n");
    exit(1);
  }

  t = clock();
  OilScaleInit(&os, aImage.mHeight, aOutHeight, aImage.mWidth, aOutWidth, cs);
  for (i = 0; i < aOutHeight; i++) {
    while (OilScaleSlots(&os)) {
      aDoIn(&os, inbuf);
      inbuf += inRowStride;
    }
    aDoOut(&os, outbuf);
  }
  t = clock() - t;
  free(outbuf);
  OilScaleFree(&os);
  return t;
}

static void DoBench(BenchImage aImage, double aRatio, int aIterations,
                    ScaleInFn aDoIn, ScaleOutFn aDoOut) {
  int i, outWidth, outHeight;
  clock_t tMin, tTmp;

  outWidth = round(aImage.mWidth * aRatio);
  outHeight = round(aImage.mHeight * aRatio);

  tMin = 0;
  for (i = 0; i < aIterations; i++) {
    tTmp = Resize(aImage, outWidth, outHeight, aDoIn, aDoOut);
    if (!tMin || tTmp < tMin) {
      tMin = tTmp;
    }
  }

  printf("    to %4dx%4d %6.2fms\n", outWidth, outHeight, TimeToMs(tMin));
}

static void DoBenchSizes(const char* aName, char* aPath, OilColorspace aCs,
                         int aIterations, const char* aImplName,
                         ScaleInFn aDoIn, ScaleOutFn aDoOut) {
  BenchImage image;
  double ratios[] = {0.01, 0.125, 0.8};
  size_t i, numRatios;

  image = LoadPng(aPath, aCs);

  printf("%dx%d %s [%s]\n", image.mWidth, image.mHeight, aName, aImplName);

  numRatios = sizeof(ratios) / sizeof(ratios[0]);
  for (i = 0; i < numRatios; i++) {
    DoBench(image, ratios[i], aIterations, aDoIn, aDoOut);
  }

  free(image.mBuffer);
}

struct Impl {
  const char* mName;
  ScaleInFn mIn;
  ScaleOutFn mOut;
};

static void RunBench(char* aPath, char* aCsArg, int aIterations, Impl* aImpls,
                     int aNumImpls) {
  size_t i, j, numSpaces;
  clock_t t;

  OilColorspace spaces[] = {
      OilColorspace::RgbaNogamma,
      OilColorspace::RgbxNogamma,
  };

  const char* spaceNames[] = {
      "RGBA_NOGAMMA",
      "RGBX_NOGAMMA",
  };

  t = clock();
  OilGlobalInit();
  t = clock() - t;
  printf("global init: %6.2fms\n", TimeToMs(t));

  numSpaces = sizeof(spaces) / sizeof(spaces[0]);

  if (aCsArg) {
    for (i = 0; i < numSpaces; i++) {
      if (strcmp(spaceNames[i], aCsArg) == 0) {
        break;
      }
    }
    if (i >= numSpaces) {
      fprintf(stderr, "Colorspace not recognized.\n");
      exit(1);
    }
    /* single colorspace */
    for (j = 0; j < static_cast<size_t>(aNumImpls); j++) {
      DoBenchSizes(spaceNames[i], aPath, spaces[i], aIterations,
                   aImpls[j].mName, aImpls[j].mIn, aImpls[j].mOut);
    }
    return;
  }

  for (i = 0; i < numSpaces; i++) {
    for (j = 0; j < static_cast<size_t>(aNumImpls); j++) {
      DoBenchSizes(spaceNames[i], aPath, spaces[i], aIterations,
                   aImpls[j].mName, aImpls[j].mIn, aImpls[j].mOut);
    }
  }
}

int main(int argc, char* argv[]) {
  int iterations, argPos, implMode, remaining;
  char* end;
  char* path;
  char* csArg;
  unsigned long ul;
  Impl impls[3];
  int numImpls;

  /* Parse flags */
  implMode = 0; /* 0=both, 1=scalar, 2=simd */
  argPos = 1;
  while (argPos < argc && argv[argPos][0] == '-') {
    if (strcmp(argv[argPos], "--scalar") == 0) {
      implMode = 1;
    } else if (strcmp(argv[argPos], "--sse2") == 0) {
      implMode = 3;
    } else if (strcmp(argv[argPos], "--avx2") == 0) {
      implMode = 4;
    } else if (strcmp(argv[argPos], "--neon") == 0) {
      implMode = 5;
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[argPos]);
      return 1;
    }
    argPos++;
  }

  remaining = argc - argPos;
  if (remaining < 1 || remaining > 2) {
    fprintf(
        stderr,
        "Usage: %s [--scalar|--sse2|--avx2|--neon] <path.png> [colorspace]\n",
        argv[0]);
    return 1;
  }

  path = argv[argPos];
  csArg = (remaining == 2) ? argv[argPos + 1] : nullptr;

  iterations = 100;
  if (getenv("OILITERATIONS")) {
    errno = 0;
    ul = strtoul(getenv("OILITERATIONS"), &end, 10);
    if (*end != '\0' || errno != 0 || ul == 0 || ul > INT_MAX) {
      fprintf(stderr, "Invalid environment variable OILITERATIONS.");
      return 1;
    }
    iterations = static_cast<int>(ul);
  }
  fprintf(stderr, "Iterations: %d\n", iterations);

  numImpls = 0;

  if (implMode == 0 || implMode == 1) {
    impls[numImpls].mName = "scalar";
    impls[numImpls].mIn = OilScaleIn;
    impls[numImpls].mOut = OilScaleOut;
    numImpls++;
  }

#if defined(__x86_64__)
  if (implMode == 0 || implMode == 3) {
    impls[numImpls].mName = "sse2";
    impls[numImpls].mIn = OilScaleInSse2;
    impls[numImpls].mOut = OilScaleOutSse2;
    numImpls++;
  }
  if (implMode == 0 || implMode == 4) {
    impls[numImpls].mName = "avx2";
    impls[numImpls].mIn = OilScaleInAvx2;
    impls[numImpls].mOut = OilScaleOutAvx2;
    numImpls++;
  }
  if (implMode == 5) {
    fprintf(stderr, "NEON not available on x86_64.\n");
    return 1;
  }
#elif defined(__aarch64__)
  if (implMode == 0 || implMode == 5) {
    impls[numImpls].mName = "neon";
    impls[numImpls].mIn = OilScaleInNeon;
    impls[numImpls].mOut = OilScaleOutNeon;
    numImpls++;
  }
  if (implMode == 3 || implMode == 4) {
    fprintf(stderr, "SSE2/AVX2 not available on AArch64.\n");
    return 1;
  }
#else
  if (implMode >= 3) {
    fprintf(stderr, "No SIMD support compiled in.\n");
    return 1;
  }
#endif

  RunBench(path, csArg, iterations, impls, numImpls);
  return 0;
}
