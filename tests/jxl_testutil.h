/*
 * Shared scaffolding for the JPEG XL tests and benchmarks: encode a gradient to
 * memory, and stand a decoder up to JXL_DEC_BASIC_INFO. Test helpers -- they
 * assert() on failure rather than returning errors.
 */
#ifndef JXL_TESTUTIL_H
#define JXL_TESTUTIL_H

#include <stddef.h>
#include <jxl/decode.h>
#include <jxl/codestream_header.h>

/* Monotonic wall-clock seconds. */
double now_sec(void);

/* qsort comparator for doubles (ascending). */
int cmp_double(const void *a, const void *b);

/* Encode a deterministic w*h sRGB RGB8 gradient as a lossless codestream into a
 * malloc'd buffer (*out, *out_size); caller frees *out. */
void encode_gradient_jxl(int w, int h, unsigned char **out, size_t *out_size);

/* Decoder over [data,data+size) advanced to JXL_DEC_BASIC_INFO with *info
 * filled; binds a runner returned via *runner_out. Caller destroys both
 * (JxlDecoderDestroy / JxlThreadParallelRunnerDestroy). */
JxlDecoder *open_jxl(unsigned char *data, size_t size,
	void **runner_out, JxlBasicInfo *info);

#endif
