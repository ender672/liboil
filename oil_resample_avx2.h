/**
 * Copyright (c) 2014-2019 Timothy Elliott
 *
 * AVX2/FMA primitives used by the coefficient builders in oil_resample.c.
 * Included only when the including TU is compiled with -mavx2 -mfma; the
 * corresponding scalar fallbacks live next to each call site.
 *
 * These are static inline because both call sites are tight (the 4-tap
 * runs once per output pixel during upscale init), and a cross-TU call
 * eats most of the savings.
 */

#ifndef OIL_RESAMPLE_AVX2_H
#define OIL_RESAMPLE_AVX2_H

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

/* Walk samples in groups of 8, evaluating catrom(|x|) * inv_tm at each
 * lane and stepping x by inv_tm per lane. Writes scaled coefficients to
 * tmp_coeffs[0..returned-1], advances *x_io to the next x, and adds the
 * sum of the written coefficients into *fudge_io. Returns the number of
 * samples consumed (n_samples & ~7), or 0 if n_samples < 8. */
static inline int oil_catrom_block8_avx2(float *tmp_coeffs, int n_samples,
	float inv_tm, float *x_io, float *fudge_io)
{
	const __m256 vsign = _mm256_set1_ps(-0.0f);
	const __m256 vone = _mm256_set1_ps(1.0f);
	const __m256 vc1_5 = _mm256_set1_ps(1.5f);
	const __m256 vcn2_5 = _mm256_set1_ps(-2.5f);
	const __m256 vc5 = _mm256_set1_ps(5.0f);
	const __m256 vcn8 = _mm256_set1_ps(-8.0f);
	const __m256 vc4 = _mm256_set1_ps(4.0f);
	const __m256 vc0_5 = _mm256_set1_ps(0.5f);
	__m256 vinv_tm, vstride, vx, vfudge;
	__m128 lo, hi, s, sh;
	float x, carry[8];
	int j, j8;

	j8 = n_samples & ~7;
	if (j8 == 0) {
		return 0;
	}

	x = *x_io;
	vinv_tm = _mm256_set1_ps(inv_tm);
	vstride = _mm256_set1_ps(8.0f * inv_tm);
	vx = _mm256_setr_ps(x,
		x + inv_tm, x + 2.0f*inv_tm, x + 3.0f*inv_tm,
		x + 4.0f*inv_tm, x + 5.0f*inv_tm, x + 6.0f*inv_tm,
		x + 7.0f*inv_tm);
	vfudge = _mm256_setzero_ps();

	for (j = 0; j < j8; j += 8) {
		__m256 vax = _mm256_andnot_ps(vsign, vx);
		__m256 vax2 = _mm256_mul_ps(vax, vax);
		__m256 vp = _mm256_fmadd_ps(vc1_5, vax, vcn2_5);
		__m256 vn, vlobe, vc;
		vp = _mm256_fmadd_ps(vp, vax2, vone);
		vn = _mm256_sub_ps(vc5, vax);
		vn = _mm256_fmadd_ps(vn, vax, vcn8);
		vn = _mm256_fmadd_ps(vn, vax, vc4);
		vn = _mm256_mul_ps(vn, vc0_5);
		vlobe = _mm256_cmp_ps(vax, vone, _CMP_LT_OQ);
		vc = _mm256_blendv_ps(vn, vp, vlobe);
		vc = _mm256_mul_ps(vc, vinv_tm);
		_mm256_storeu_ps(tmp_coeffs + j, vc);
		vfudge = _mm256_add_ps(vfudge, vc);
		vx = _mm256_add_ps(vx, vstride);
	}

	lo = _mm256_castps256_ps128(vfudge);
	hi = _mm256_extractf128_ps(vfudge, 1);
	s = _mm_add_ps(lo, hi);
	sh = _mm_movehl_ps(s, s);
	s = _mm_add_ps(s, sh);
	sh = _mm_shuffle_ps(s, s, 0x55);
	s = _mm_add_ss(s, sh);
	*fudge_io += _mm_cvtss_f32(s);

	_mm256_storeu_ps(carry, vx);
	*x_io = carry[0];
	return j8;
}

/* Evaluate the 4 Catmull-Rom coefficients for an interior upscale tap
 * (offsets -1-tx, -tx, 1-tx, 2-tx from the sample center). Lanes 1,2
 * use the positive lobe, lanes 0,3 the negative — selected with a
 * constant blend mask. Writes 4 floats to coeff_buf. */
static inline void oil_catrom_4tap_avx2(float *coeff_buf, float tx)
{
	__m128 vd = _mm_setr_ps(1.0f + tx, tx, 1.0f - tx, 2.0f - tx);
	__m128 vd2 = _mm_mul_ps(vd, vd);
	__m128 vp = _mm_fmadd_ps(_mm_set1_ps(1.5f), vd, _mm_set1_ps(-2.5f));
	__m128 vn, vmask, vc;
	vp = _mm_fmadd_ps(vp, vd2, _mm_set1_ps(1.0f));
	vn = _mm_sub_ps(_mm_set1_ps(5.0f), vd);
	vn = _mm_fmadd_ps(vn, vd, _mm_set1_ps(-8.0f));
	vn = _mm_fmadd_ps(vn, vd, _mm_set1_ps(4.0f));
	vn = _mm_mul_ps(vn, _mm_set1_ps(0.5f));
	vmask = _mm_castsi128_ps(_mm_setr_epi32(0, -1, -1, 0));
	vc = _mm_blendv_ps(vn, vp, vmask);
	_mm_storeu_ps(coeff_buf, vc);
}

#endif /* __AVX2__ && __FMA__ */

#endif
