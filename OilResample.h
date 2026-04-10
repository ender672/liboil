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

#ifndef OIL_RESAMPLE_H_
#define OIL_RESAMPLE_H_

namespace mozilla {

/**
 * Color spaces currently supported by oil.
 */
enum class OilColorspace {
  Rgba = 0x0604,
  Rgbx = 0x0704,
};

/**
 * Struct to hold state for scaling. Changing these will produce unpredictable
 * results.
 */
struct OilScale {
  int mInHeight;
  int mOutHeight;
  int mInWidth;
  int mOutWidth;
  OilColorspace mCs;
  int mInPos;
  int mOutPos;
  float* mCoeffsY;
  float* mCoeffsX;
  int* mBordersX;
  int* mBordersY;
  float* mSumsY;
  float* mTmpCoeffs;
  void* mBuf;
  int mSumsYTap;
};

/**
 * Calculate the buffer size needed for an oil scaler struct.
 */
int OilScaleAllocSize(int aInHeight, int aOutHeight, int aInWidth,
                      int aOutWidth, OilColorspace aCs);

/**
 * Initialize an oil scaler struct with a pre-allocated buffer.
 *
 * Returns 0 on success.
 * Returns -1 if an argument is bad.
 */
[[nodiscard]] int OilScaleInitAllocated(OilScale* aOs, int aInHeight,
                                        int aOutHeight, int aInWidth,
                                        int aOutWidth, OilColorspace aCs,
                                        void* aBuf);

/**
 * Reset row counters in an oil scaler struct.
 */
void OilScaleRestart(OilScale* aOs);

/**
 * Clear an oil scaler struct.
 */
void OilScaleFree(OilScale* aOs);

/**
 * Return the number of input scanlines needed before the next output scanline
 * can be produced.
 */
int OilScaleSlots(OilScale* aOs);

/**
 * Ingest & buffer an input scanline. Input is unsigned chars.
 *
 * Returns 0 on success.
 * Returns -1 if an output scanline is ready and must be consumed first.
 */
[[nodiscard]] int OilScaleIn(OilScale* aOs, unsigned char* aIn);

/**
 * Scale previously ingested & buffered contents to produce the next scaled
 * output scanline.
 *
 * Returns 0 on success.
 * Returns -1 if not enough input scanlines have been fed yet.
 */
[[nodiscard]] int OilScaleOut(OilScale* aOs, unsigned char* aOut);

/**
 * SSE2-optimized version of OilScaleIn().
 */
[[nodiscard]] int OilScaleInSse2(OilScale* aOs, unsigned char* aIn);

/**
 * SSE2-optimized version of OilScaleOut().
 */
[[nodiscard]] int OilScaleOutSse2(OilScale* aOs, unsigned char* aOut);

/**
 * AVX2-optimized version of OilScaleIn().
 */
[[nodiscard]] int OilScaleInAvx2(OilScale* aOs, unsigned char* aIn);

/**
 * AVX2-optimized version of OilScaleOut().
 */
[[nodiscard]] int OilScaleOutAvx2(OilScale* aOs, unsigned char* aOut);

/**
 * NEON-optimized version of OilScaleIn().
 */
[[nodiscard]] int OilScaleInNeon(OilScale* aOs, unsigned char* aIn);

/**
 * NEON-optimized version of OilScaleOut().
 */
[[nodiscard]] int OilScaleOutNeon(OilScale* aOs, unsigned char* aOut);

}  // namespace mozilla

#endif
