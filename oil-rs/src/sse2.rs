#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

use crate::srgb;

/// Equivalent to C's mm_shuffle(z, y, x, w).
const fn mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

/// SSE2 horizontal upscale for RGB.
/// Mirrors oil_xscale_up_rgb_sse2: per-channel sliding window with vectorized dot products.
#[target_feature(enable = "sse2")]
pub unsafe fn xscale_up_rgb(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = _mm_setzero_ps();
    let mut smp_g = _mm_setzero_ps();
    let mut smp_b = _mm_setzero_ps();
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 3;

        smp_r = push_f_sse2(smp_r, *s2l.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_sse2(smp_g, *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_sse2(smp_b, *s2l.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);

            *out_ptr.add(out_idx)     = _mm_cvtss_f32(t2_r);
            *out_ptr.add(out_idx + 1) = _mm_cvtss_f32(t2_g);
            *out_ptr.add(out_idx + 2) = _mm_cvtss_f32(t2_b);
            *out_ptr.add(out_idx + 3) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_r, t2_r, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 4) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_g, t2_g, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 5) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_b, t2_b, mm_shuffle(1, 1, 1, 1)),
            );

            out_idx += 6;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = _mm_loadu_ps(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);

            out_idx += 3;
            coeff_idx += 4;
        }
    }
}

/// SSE2 vertical upscale for RGB.
/// Mirrors oil_yscale_up_rgb_sse2: 4-tap vertical blend, output through l2s LUT.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_up_rgb(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale_f = (tables.l2s_len - 1) as f32;

    let c0 = _mm_set1_ps(coeffs[0] * scale_f);
    let c1 = _mm_set1_ps(coeffs[1] * scale_f);
    let c2 = _mm_set1_ps(coeffs[2] * scale_f);
    let c3 = _mm_set1_ps(coeffs[3] * scale_f);

    let mut i = 0;
    let mut idx_buf: [i32; 12] = [0i32; 12];
    let idx_ptr = idx_buf.as_mut_ptr() as *mut __m128i;
    let out_ptr = out.as_mut_ptr();

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();

    // Process 12 floats at a time (4 RGB pixels)
    while i + 11 < len {
        let sum0 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        let sum1 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 4))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 4))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 4))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 4))))));
        let sum2 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 8))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 8))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 8))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 8))))));

        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(sum0));
        _mm_storeu_si128(idx_ptr.add(1), _mm_cvttps_epi32(sum1));
        _mm_storeu_si128(idx_ptr.add(2), _mm_cvttps_epi32(sum2));

        *out_ptr.add(i)      = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1)  = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2)  = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3)  = *lut.offset(idx_buf[3] as isize);
        *out_ptr.add(i + 4)  = *lut.offset(idx_buf[4] as isize);
        *out_ptr.add(i + 5)  = *lut.offset(idx_buf[5] as isize);
        *out_ptr.add(i + 6)  = *lut.offset(idx_buf[6] as isize);
        *out_ptr.add(i + 7)  = *lut.offset(idx_buf[7] as isize);
        *out_ptr.add(i + 8)  = *lut.offset(idx_buf[8] as isize);
        *out_ptr.add(i + 9)  = *lut.offset(idx_buf[9] as isize);
        *out_ptr.add(i + 10) = *lut.offset(idx_buf[10] as isize);
        *out_ptr.add(i + 11) = *lut.offset(idx_buf[11] as isize);

        i += 12;
    }

    // Process 4 floats at a time
    while i + 3 < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(sum));
        *out_ptr.add(i)     = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1) = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2) = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3) = *lut.offset(idx_buf[3] as isize);
        i += 4;
    }

    // Scalar tail
    while i < len {
        let val = *coeffs.get_unchecked(0) * scale_f * *l0.add(i)
            + *coeffs.get_unchecked(1) * scale_f * *l1.add(i)
            + *coeffs.get_unchecked(2) * scale_f * *l2.add(i)
            + *coeffs.get_unchecked(3) * scale_f * *l3.add(i);
        *out_ptr.add(i) = *lut.offset(val as isize);
        i += 1;
    }
}

/// SSE2 downscale for RGB: horizontal x-filtering + y-accumulation.
/// Mirrors oil_scale_down_rgb_sse2.
#[target_feature(enable = "sse2")]
pub unsafe fn scale_down_rgb(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());

    let mut sum_r = _mm_setzero_ps();
    let mut sum_g = _mm_setzero_ps();
    let mut sum_b = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 16 {
            // 4-way unroll for large borders (extreme downscale)
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();
            let mut sum_r3 = _mm_setzero_ps();
            let mut sum_g3 = _mm_setzero_ps();
            let mut sum_b3 = _mm_setzero_ps();
            let mut sum_r4 = _mm_setzero_ps();
            let mut sum_g4 = _mm_setzero_ps();
            let mut sum_b4 = _mm_setzero_ps();

            let mut j = 0;
            while j + 3 < border {
                let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx0, s), sum_r);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx0, s), sum_g);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx0, s), sum_b);

                let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_r2);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_g2);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_b2);

                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_r3 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_r3);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 7) as usize));
                sum_g3 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_g3);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 8) as usize));
                sum_b3 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_b3);

                let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 9) as usize));
                sum_r4 = _mm_add_ps(_mm_mul_ps(cx3, s), sum_r4);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 10) as usize));
                sum_g4 = _mm_add_ps(_mm_mul_ps(cx3, s), sum_g4);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 11) as usize));
                sum_b4 = _mm_add_ps(_mm_mul_ps(cx3, s), sum_b4);

                in_idx += 12;
                cx_idx += 16;
                j += 4;
            }
            sum_r = _mm_add_ps(_mm_add_ps(sum_r, sum_r2), _mm_add_ps(sum_r3, sum_r4));
            sum_g = _mm_add_ps(_mm_add_ps(sum_g, sum_g2), _mm_add_ps(sum_g3, sum_g4));
            sum_b = _mm_add_ps(_mm_add_ps(sum_b, sum_b2), _mm_add_ps(sum_b3, sum_b4));
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);
                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        } else if border >= 4 {
            // 2-way unroll for moderate borders
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 3) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_r2);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_g2);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_b2);

                in_idx += 6;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }

            sum_r = _mm_add_ps(sum_r, sum_r2);
            sum_g = _mm_add_ps(sum_g, sum_g2);
            sum_b = _mm_add_ps(sum_b, sum_b2);
        } else if border == 1 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

            let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
            sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

            let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
            sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

            let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
            sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

            in_idx += 3;
            cx_idx += 4;
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

                in_idx += 3;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_r, sum_r, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // G channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_g, sum_g, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // B channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_b, sum_b, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // shift_left for each channel
        sum_r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_r), 4));
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_b = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_b), 4));
    }
}

/// SSE2 output for downscaled linear RGB: convert sums through l2s LUT.
/// Mirrors oil_yscale_out_linear_sse2.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_out_rgb(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    let mut s_idx = 0;

    // Process 8 output values at a time
    while i + 7 < sl_len {
        let sp = s_ptr.add(s_idx) as *mut __m128i;

        // First batch of 4
        let v0 = _mm_loadu_si128(sp);
        let v1 = _mm_loadu_si128(sp.add(1));
        let v2 = _mm_loadu_si128(sp.add(2));
        let v3 = _mm_loadu_si128(sp.add(3));

        let f0 = _mm_castsi128_ps(v0);
        let f1 = _mm_castsi128_ps(v1);
        let f2 = _mm_castsi128_ps(v2);
        let f3 = _mm_castsi128_ps(v3);
        let ab = _mm_shuffle_ps(f0, f1, mm_shuffle(0, 0, 0, 0));
        let cd = _mm_shuffle_ps(f2, f3, mm_shuffle(0, 0, 0, 0));
        let vals = _mm_shuffle_ps(ab, cd, mm_shuffle(2, 0, 2, 0));
        let idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

        // Second batch of 4
        let v4 = _mm_loadu_si128(sp.add(4));
        let v5 = _mm_loadu_si128(sp.add(5));
        let v6 = _mm_loadu_si128(sp.add(6));
        let v7 = _mm_loadu_si128(sp.add(7));

        let g0 = _mm_castsi128_ps(v4);
        let g1 = _mm_castsi128_ps(v5);
        let g2 = _mm_castsi128_ps(v6);
        let g3 = _mm_castsi128_ps(v7);
        let ab2 = _mm_shuffle_ps(g0, g1, mm_shuffle(0, 0, 0, 0));
        let cd2 = _mm_shuffle_ps(g2, g3, mm_shuffle(0, 0, 0, 0));
        let vals2 = _mm_shuffle_ps(ab2, cd2, mm_shuffle(2, 0, 2, 0));
        let idx2 = _mm_cvttps_epi32(_mm_mul_ps(vals2, scale));

        // Interleave LUT lookups from both batches for ILP
        *out_ptr.add(i)     = *lut.offset(_mm_cvtsi128_si32(idx) as isize);
        *out_ptr.add(i + 4) = *lut.offset(_mm_cvtsi128_si32(idx2) as isize);
        *out_ptr.add(i + 1) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 4)) as isize);
        *out_ptr.add(i + 5) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx2, 4)) as isize);
        *out_ptr.add(i + 2) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 8)) as isize);
        *out_ptr.add(i + 6) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx2, 8)) as isize);
        *out_ptr.add(i + 3) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 12)) as isize);
        *out_ptr.add(i + 7) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx2, 12)) as isize);

        // Shift all 8 accumulators left
        _mm_storeu_si128(sp, _mm_srli_si128(v0, 4));
        _mm_storeu_si128(sp.add(1), _mm_srli_si128(v1, 4));
        _mm_storeu_si128(sp.add(2), _mm_srli_si128(v2, 4));
        _mm_storeu_si128(sp.add(3), _mm_srli_si128(v3, 4));
        _mm_storeu_si128(sp.add(4), _mm_srli_si128(v4, 4));
        _mm_storeu_si128(sp.add(5), _mm_srli_si128(v5, 4));
        _mm_storeu_si128(sp.add(6), _mm_srli_si128(v6, 4));
        _mm_storeu_si128(sp.add(7), _mm_srli_si128(v7, 4));

        s_idx += 32;
        i += 8;
    }

    // Process 4 output values at a time
    while i + 3 < sl_len {
        let sp = s_ptr.add(s_idx) as *mut __m128i;
        let v0 = _mm_loadu_si128(sp);
        let v1 = _mm_loadu_si128(sp.add(1));
        let v2 = _mm_loadu_si128(sp.add(2));
        let v3 = _mm_loadu_si128(sp.add(3));

        let f0 = _mm_castsi128_ps(v0);
        let f1 = _mm_castsi128_ps(v1);
        let f2 = _mm_castsi128_ps(v2);
        let f3 = _mm_castsi128_ps(v3);
        let ab = _mm_shuffle_ps(f0, f1, mm_shuffle(0, 0, 0, 0));
        let cd = _mm_shuffle_ps(f2, f3, mm_shuffle(0, 0, 0, 0));
        let vals = _mm_shuffle_ps(ab, cd, mm_shuffle(2, 0, 2, 0));

        let idx = _mm_cvttps_epi32(_mm_mul_ps(vals, scale));

        *out_ptr.add(i)     = *lut.offset(_mm_cvtsi128_si32(idx) as isize);
        *out_ptr.add(i + 1) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 4)) as isize);
        *out_ptr.add(i + 2) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 8)) as isize);
        *out_ptr.add(i + 3) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 12)) as isize);

        _mm_storeu_si128(sp, _mm_srli_si128(v0, 4));
        _mm_storeu_si128(sp.add(1), _mm_srli_si128(v1, 4));
        _mm_storeu_si128(sp.add(2), _mm_srli_si128(v2, 4));
        _mm_storeu_si128(sp.add(3), _mm_srli_si128(v3, 4));

        s_idx += 16;
        i += 4;
    }

    // Scalar tail
    while i < sl_len {
        let val = *s_ptr.add(s_idx);
        *out_ptr.add(i) = *lut.offset((val * (tables.l2s_len - 1) as f32) as isize);
        // shift_left
        *s_ptr.add(s_idx) = *s_ptr.add(s_idx + 1);
        *s_ptr.add(s_idx + 1) = *s_ptr.add(s_idx + 2);
        *s_ptr.add(s_idx + 2) = *s_ptr.add(s_idx + 3);
        *s_ptr.add(s_idx + 3) = 0.0;
        s_idx += 4;
        i += 1;
    }
}

/// SSE2 horizontal upscale for RGBA (premultiplied alpha).
#[target_feature(enable = "sse2")]
pub unsafe fn xscale_up_rgba(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = _mm_setzero_ps();
    let mut smp_g = _mm_setzero_ps();
    let mut smp_b = _mm_setzero_ps();
    let mut smp_a = _mm_setzero_ps();
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;
        let alpha_new = *in_ptr.add(in_base + 3) as f32 / 255.0;

        smp_a = push_f_sse2(smp_a, alpha_new);
        smp_r = push_f_sse2(smp_r, alpha_new * *s2l.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_sse2(smp_g, alpha_new * *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_sse2(smp_b, alpha_new * *s2l.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);
            let t2_a = dot4x2(smp_a, c0, c1);

            // Transpose [r0,r1] [g0,g1] [b0,b1] [a0,a1] -> [r0,g0,b0,a0] [r1,g1,b1,a1]
            let rg = _mm_unpacklo_ps(t2_r, t2_g); // [r0, g0, r1, g1]
            let ba = _mm_unpacklo_ps(t2_b, t2_a); // [b0, a0, b1, a1]
            _mm_storeu_ps(out_ptr.add(out_idx), _mm_movelh_ps(rg, ba));
            _mm_storeu_ps(out_ptr.add(out_idx + 4), _mm_movehl_ps(ba, rg));

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = _mm_loadu_ps(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = dot4(smp_a, coeffs);

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// SSE2 vertical upscale for RGBA (premultiplied alpha).
/// Processes 4 floats (one RGBA pixel) at a time, un-premultiplies, converts to sRGB.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_up_rgba(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let c0 = _mm_set1_ps(coeffs[0]);
    let c1 = _mm_set1_ps(coeffs[1]);
    let c2 = _mm_set1_ps(coeffs[2]);
    let c3 = _mm_set1_ps(coeffs[3]);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;
    let mut idx_buf: [i32; 12] = [0i32; 12];
    let idx_ptr = idx_buf.as_mut_ptr() as *mut __m128i;

    // Process 3 RGBA pixels (12 floats) at a time
    while i + 11 < len {
        // Vertical blend for pixel 0
        let sum0 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        // Vertical blend for pixel 1
        let sum1 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 4))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 4))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 4))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 4))))));
        // Vertical blend for pixel 2
        let sum2 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 8))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 8))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 8))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 8))))));

        // Un-premultiply pixel 0
        let a0_v = _mm_shuffle_ps(sum0, sum0, mm_shuffle(3, 3, 3, 3));
        let a0_v = _mm_min_ps(_mm_max_ps(a0_v, zero), one);
        let a0 = _mm_cvtss_f32(a0_v);
        let mut vals0 = sum0;
        if a0 != 0.0 { vals0 = _mm_mul_ps(vals0, _mm_rcp_ps(a0_v)); }
        let clamped0 = _mm_min_ps(_mm_max_ps(vals0, zero), one);

        // Un-premultiply pixel 1
        let a1_v = _mm_shuffle_ps(sum1, sum1, mm_shuffle(3, 3, 3, 3));
        let a1_v = _mm_min_ps(_mm_max_ps(a1_v, zero), one);
        let a1 = _mm_cvtss_f32(a1_v);
        let mut vals1 = sum1;
        if a1 != 0.0 { vals1 = _mm_mul_ps(vals1, _mm_rcp_ps(a1_v)); }
        let clamped1 = _mm_min_ps(_mm_max_ps(vals1, zero), one);

        // Un-premultiply pixel 2
        let a2_v = _mm_shuffle_ps(sum2, sum2, mm_shuffle(3, 3, 3, 3));
        let a2_v = _mm_min_ps(_mm_max_ps(a2_v, zero), one);
        let a2 = _mm_cvtss_f32(a2_v);
        let mut vals2 = sum2;
        if a2 != 0.0 { vals2 = _mm_mul_ps(vals2, _mm_rcp_ps(a2_v)); }
        let clamped2 = _mm_min_ps(_mm_max_ps(vals2, zero), one);

        // Batch convert to l2s indices
        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(_mm_mul_ps(clamped0, scale)));
        _mm_storeu_si128(idx_ptr.add(1), _mm_cvttps_epi32(_mm_mul_ps(clamped1, scale)));
        _mm_storeu_si128(idx_ptr.add(2), _mm_cvttps_epi32(_mm_mul_ps(clamped2, scale)));

        // Batch LUT lookups for RGB + direct alpha writes
        *out_ptr.add(i)      = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1)  = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2)  = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3)  = (a0 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 4)  = *lut.offset(idx_buf[4] as isize);
        *out_ptr.add(i + 5)  = *lut.offset(idx_buf[5] as isize);
        *out_ptr.add(i + 6)  = *lut.offset(idx_buf[6] as isize);
        *out_ptr.add(i + 7)  = (a1 * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 8)  = *lut.offset(idx_buf[8] as isize);
        *out_ptr.add(i + 9)  = *lut.offset(idx_buf[9] as isize);
        *out_ptr.add(i + 10) = *lut.offset(idx_buf[10] as isize);
        *out_ptr.add(i + 11) = (a2 * 255.0 + 0.5) as u8;

        i += 12;
    }

    // Process remaining pixels one at a time
    while i < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));

        let alpha_v = _mm_shuffle_ps(sum, sum, mm_shuffle(3, 3, 3, 3));
        let alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
        let alpha = _mm_cvtss_f32(alpha_v);

        let mut vals = sum;
        if alpha != 0.0 {
            vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
        }
        let clamped = _mm_min_ps(_mm_max_ps(vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_mul_ps(clamped, scale));

        *out_ptr.add(i)     = *lut.offset(_mm_cvtsi128_si32(idx) as isize);
        *out_ptr.add(i + 1) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 4)) as isize);
        *out_ptr.add(i + 2) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 8)) as isize);
        *out_ptr.add(i + 3) = (alpha * 255.0 + 0.5) as u8;

        i += 4;
    }
}

/// SSE2 downscale for RGBA: horizontal x-filtering with premultiplied alpha + y-accumulation.
#[target_feature(enable = "sse2")]
pub unsafe fn scale_down_rgba(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let i2f = tables.i2f.as_ptr();
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());

    let mut sum_r = _mm_setzero_ps();
    let mut sum_g = _mm_setzero_ps();
    let mut sum_b = _mm_setzero_ps();
    let mut sum_a = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 4 {
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();
            let mut sum_a2 = _mm_setzero_ps();

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);

                sum_a = _mm_add_ps(cx_a, sum_a);

                let cx2_a = _mm_mul_ps(cx2, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 7) as usize)));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_r2);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_g2);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx2_a, s), sum_b2);

                sum_a2 = _mm_add_ps(cx2_a, sum_a2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);

                sum_a = _mm_add_ps(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = _mm_add_ps(sum_r, sum_r2);
            sum_g = _mm_add_ps(sum_g, sum_g2);
            sum_b = _mm_add_ps(sum_b, sum_b2);
            sum_a = _mm_add_ps(sum_a, sum_a2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let cx_a = _mm_mul_ps(cx, _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize)));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx_a, s), sum_b);

                sum_a = _mm_add_ps(cx_a, sum_a);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_r, sum_r, mm_shuffle(0, 0, 0, 0));
        _mm_storeu_ps(sy_ptr.add(sy_idx), _mm_add_ps(_mm_mul_ps(cy, sample), sy));
        sy_idx += 4;

        // G channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_g, sum_g, mm_shuffle(0, 0, 0, 0));
        _mm_storeu_ps(sy_ptr.add(sy_idx), _mm_add_ps(_mm_mul_ps(cy, sample), sy));
        sy_idx += 4;

        // B channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_b, sum_b, mm_shuffle(0, 0, 0, 0));
        _mm_storeu_ps(sy_ptr.add(sy_idx), _mm_add_ps(_mm_mul_ps(cy, sample), sy));
        sy_idx += 4;

        // A channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_a, sum_a, mm_shuffle(0, 0, 0, 0));
        _mm_storeu_ps(sy_ptr.add(sy_idx), _mm_add_ps(_mm_mul_ps(cy, sample), sy));
        sy_idx += 4;

        // shift_left for each channel
        sum_r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_r), 4));
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_b = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_b), 4));
        sum_a = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_a), 4));
    }
}

/// SSE2 output for downscaled RGBA: un-premultiply, convert RGB through l2s LUT, alpha to byte.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_out_rgba(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx) as *mut __m128i;

        // Load 4 accumulators for this pixel: [R0..R3], [G0..G3], [B0..B3], [A0..A3]
        let v0 = _mm_loadu_si128(sp);
        let v1 = _mm_loadu_si128(sp.add(1));
        let v2 = _mm_loadu_si128(sp.add(2));
        let v3 = _mm_loadu_si128(sp.add(3));

        // Gather first element of each accumulator: {R, G, B, A}
        let f0 = _mm_castsi128_ps(v0);
        let f1 = _mm_castsi128_ps(v1);
        let f2 = _mm_castsi128_ps(v2);
        let f3 = _mm_castsi128_ps(v3);
        let ab = _mm_shuffle_ps(f0, f1, mm_shuffle(0, 0, 0, 0));
        let cd = _mm_shuffle_ps(f2, f3, mm_shuffle(0, 0, 0, 0));
        let vals = _mm_shuffle_ps(ab, cd, mm_shuffle(2, 0, 2, 0));

        // Clamp alpha to [0, 1]
        let alpha_v = _mm_shuffle_ps(vals, vals, mm_shuffle(3, 3, 3, 3));
        let alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
        let alpha = _mm_cvtss_f32(alpha_v);

        // Divide RGB by alpha (skip if alpha == 0)
        let mut rgb_vals = vals;
        if alpha != 0.0 {
            rgb_vals = _mm_mul_ps(rgb_vals, _mm_rcp_ps(alpha_v));
        }

        // Clamp RGB to [0, 1] and compute l2s_map indices
        rgb_vals = _mm_min_ps(_mm_max_ps(rgb_vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_mul_ps(rgb_vals, scale));

        *out_ptr.add(o_idx)     = *lut.offset(_mm_cvtsi128_si32(idx) as isize);
        *out_ptr.add(o_idx + 1) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 4)) as isize);
        *out_ptr.add(o_idx + 2) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 8)) as isize);
        *out_ptr.add(o_idx + 3) = (alpha * 255.0 + 0.5) as u8;

        // Shift all 4 accumulators left
        _mm_storeu_si128(sp, _mm_srli_si128(v0, 4));
        _mm_storeu_si128(sp.add(1), _mm_srli_si128(v1, 4));
        _mm_storeu_si128(sp.add(2), _mm_srli_si128(v2, 4));
        _mm_storeu_si128(sp.add(3), _mm_srli_si128(v3, 4));

        s_idx += 16;
        o_idx += 4;
    }
}

// --- RGBX SSE2 ---

/// SSE2 horizontal upscale for RGBX.
/// Like RGBA but 4th component is always 1.0 and RGB is not premultiplied by alpha.
#[target_feature(enable = "sse2")]
pub unsafe fn xscale_up_rgbx(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let mut smp_r = _mm_setzero_ps();
    let mut smp_g = _mm_setzero_ps();
    let mut smp_b = _mm_setzero_ps();
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 4;

        smp_r = push_f_sse2(smp_r, *s2l.add(*in_ptr.add(in_base) as usize));
        smp_g = push_f_sse2(smp_g, *s2l.add(*in_ptr.add(in_base + 1) as usize));
        smp_b = push_f_sse2(smp_b, *s2l.add(*in_ptr.add(in_base + 2) as usize));

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));

            let t2_r = dot4x2(smp_r, c0, c1);
            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_b = dot4x2(smp_b, c0, c1);

            *out_ptr.add(out_idx)     = _mm_cvtss_f32(t2_r);
            *out_ptr.add(out_idx + 1) = _mm_cvtss_f32(t2_g);
            *out_ptr.add(out_idx + 2) = _mm_cvtss_f32(t2_b);
            *out_ptr.add(out_idx + 3) = 1.0;
            *out_ptr.add(out_idx + 4) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_r, t2_r, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 5) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_g, t2_g, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 6) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_b, t2_b, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 7) = 1.0;

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = _mm_loadu_ps(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_r, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 2) = dot4(smp_b, coeffs);
            *out_ptr.add(out_idx + 3) = 1.0;

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// SSE2 vertical upscale for RGBX.
/// No alpha un-premultiply; RGB through l2s LUT, X byte always 255.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_up_rgbx(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale_f = (tables.l2s_len - 1) as f32;

    let c0 = _mm_set1_ps(coeffs[0] * scale_f);
    let c1 = _mm_set1_ps(coeffs[1] * scale_f);
    let c2 = _mm_set1_ps(coeffs[2] * scale_f);
    let c3 = _mm_set1_ps(coeffs[3] * scale_f);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;
    let mut idx_buf: [i32; 12] = [0i32; 12];
    let idx_ptr = idx_buf.as_mut_ptr() as *mut __m128i;

    // Process 12 floats at a time (3 RGBX pixels)
    while i + 11 < len {
        let sum0 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        let sum1 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 4))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 4))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 4))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 4))))));
        let sum2 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 8))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 8))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 8))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 8))))));

        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(sum0));
        _mm_storeu_si128(idx_ptr.add(1), _mm_cvttps_epi32(sum1));
        _mm_storeu_si128(idx_ptr.add(2), _mm_cvttps_epi32(sum2));

        *out_ptr.add(i)      = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1)  = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2)  = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3)  = 255;
        *out_ptr.add(i + 4)  = *lut.offset(idx_buf[4] as isize);
        *out_ptr.add(i + 5)  = *lut.offset(idx_buf[5] as isize);
        *out_ptr.add(i + 6)  = *lut.offset(idx_buf[6] as isize);
        *out_ptr.add(i + 7)  = 255;
        *out_ptr.add(i + 8)  = *lut.offset(idx_buf[8] as isize);
        *out_ptr.add(i + 9)  = *lut.offset(idx_buf[9] as isize);
        *out_ptr.add(i + 10) = *lut.offset(idx_buf[10] as isize);
        *out_ptr.add(i + 11) = 255;

        i += 12;
    }

    // Process 4 floats at a time (1 RGBX pixel)
    while i + 3 < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(sum));
        *out_ptr.add(i)     = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1) = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2) = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3) = 255;
        i += 4;
    }
}

/// SSE2 downscale for RGBX: horizontal x-filtering with X=1.0 + y-accumulation.
#[target_feature(enable = "sse2")]
pub unsafe fn scale_down_rgbx(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let s2l = tables.s2l.as_ptr();
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());

    let mut sum_r = _mm_setzero_ps();
    let mut sum_g = _mm_setzero_ps();
    let mut sum_b = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 16 {
            // 4-way unroll for large borders (extreme downscale)
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();
            let mut sum_r3 = _mm_setzero_ps();
            let mut sum_g3 = _mm_setzero_ps();
            let mut sum_b3 = _mm_setzero_ps();
            let mut sum_r4 = _mm_setzero_ps();
            let mut sum_g4 = _mm_setzero_ps();
            let mut sum_b4 = _mm_setzero_ps();

            let mut j = 0;
            while j + 3 < border {
                let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx0, s), sum_r);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx0, s), sum_g);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx0, s), sum_b);

                let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_r2);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_g2);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx1, s), sum_b2);

                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 8) as usize));
                sum_r3 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_r3);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 9) as usize));
                sum_g3 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_g3);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 10) as usize));
                sum_b3 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_b3);

                let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 12) as usize));
                sum_r4 = _mm_add_ps(_mm_mul_ps(cx3, s), sum_r4);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 13) as usize));
                sum_g4 = _mm_add_ps(_mm_mul_ps(cx3, s), sum_g4);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 14) as usize));
                sum_b4 = _mm_add_ps(_mm_mul_ps(cx3, s), sum_b4);

                in_idx += 16;
                cx_idx += 16;
                j += 4;
            }
            sum_r = _mm_add_ps(_mm_add_ps(sum_r, sum_r2), _mm_add_ps(sum_r3, sum_r4));
            sum_g = _mm_add_ps(_mm_add_ps(sum_g, sum_g2), _mm_add_ps(sum_g3, sum_g4));
            sum_b = _mm_add_ps(_mm_add_ps(sum_b, sum_b2), _mm_add_ps(sum_b3, sum_b4));
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);
                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);
                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        } else if border >= 4 {
            // 2-way unroll for moderate borders
            let mut sum_r2 = _mm_setzero_ps();
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_b2 = _mm_setzero_ps();

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 4) as usize));
                sum_r2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_r2);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 5) as usize));
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_g2);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 6) as usize));
                sum_b2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_b2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_r = _mm_add_ps(sum_r, sum_r2);
            sum_g = _mm_add_ps(sum_g, sum_g2);
            sum_b = _mm_add_ps(sum_b, sum_b2);
        } else if border == 1 {
            let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

            let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
            sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

            let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
            sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

            let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
            sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

            in_idx += 4;
            cx_idx += 4;
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx) as usize));
                sum_r = _mm_add_ps(_mm_mul_ps(cx, s), sum_r);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 1) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx, s), sum_g);

                let s = _mm_set1_ps(*s2l.add(*in_ptr.add(in_idx + 2) as usize));
                sum_b = _mm_add_ps(_mm_mul_ps(cx, s), sum_b);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // Accumulate into y sums: R channel
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_r, sum_r, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // G channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_g, sum_g, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // B channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_b, sum_b, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Skip X channel y-accumulation (X is always 1.0, output always 255)
        sy_idx += 4;

        // shift_left for each channel
        sum_r = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_r), 4));
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_b = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_b), 4));
    }
}

/// SSE2 output for downscaled RGBX: convert RGB through l2s LUT, X byte always 255.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_out_rgbx(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let tables = srgb::tables();
    let lut = tables.l2s_ptr();
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx) as *mut __m128i;

        // Load 3 accumulators for this pixel: [R0..R3], [G0..G3], [B0..B3]
        // X accumulator is unused (output always 255)
        let v0 = _mm_loadu_si128(sp);
        let v1 = _mm_loadu_si128(sp.add(1));
        let v2 = _mm_loadu_si128(sp.add(2));

        // Gather first element of each accumulator: {R, G, B, _}
        let f0 = _mm_castsi128_ps(v0);
        let f1 = _mm_castsi128_ps(v1);
        let f2 = _mm_castsi128_ps(v2);
        let ab = _mm_shuffle_ps(f0, f1, mm_shuffle(0, 0, 0, 0));
        let cd = _mm_shuffle_ps(f2, f2, mm_shuffle(0, 0, 0, 0));
        let vals = _mm_shuffle_ps(ab, cd, mm_shuffle(2, 0, 2, 0));

        // Clamp RGB to [0, 1] and compute l2s_map indices
        let clamped = _mm_min_ps(_mm_max_ps(vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_mul_ps(clamped, scale));

        *out_ptr.add(o_idx)     = *lut.offset(_mm_cvtsi128_si32(idx) as isize);
        *out_ptr.add(o_idx + 1) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 4)) as isize);
        *out_ptr.add(o_idx + 2) = *lut.offset(_mm_cvtsi128_si32(_mm_srli_si128(idx, 8)) as isize);
        *out_ptr.add(o_idx + 3) = 255;

        // Shift R, G, B accumulators left (skip X)
        _mm_storeu_si128(sp, _mm_srli_si128(v0, 4));
        _mm_storeu_si128(sp.add(1), _mm_srli_si128(v1, 4));
        _mm_storeu_si128(sp.add(2), _mm_srli_si128(v2, 4));

        s_idx += 16;
        o_idx += 4;
    }
}

// --- Grayscale (G) SSE2 ---

/// SSE2 horizontal upscale for G (grayscale).
/// Mirrors oil_xscale_up_g_sse2: single sliding window with vectorized dot products.
#[target_feature(enable = "sse2")]
pub unsafe fn xscale_up_g(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let mut smp = _mm_setzero_ps();
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        smp = push_f_sse2(smp, *i2f.add(*in_ptr.add(i) as usize));

        let mut j = *border_ptr.add(i);

        // Process quads of outputs: transpose 4 coefficient vectors, broadcast
        // each sample element, FMA to get 4 dot products, vector store.
        while j >= 4 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));
            let c2 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 8));
            let c3 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 12));

            // Transpose 4x4: [c0,c1,c2,c3] rows -> columns
            let t01_lo = _mm_unpacklo_ps(c0, c1); // [c0_0, c1_0, c0_1, c1_1]
            let t01_hi = _mm_unpackhi_ps(c0, c1); // [c0_2, c1_2, c0_3, c1_3]
            let t23_lo = _mm_unpacklo_ps(c2, c3); // [c2_0, c3_0, c2_1, c3_1]
            let t23_hi = _mm_unpackhi_ps(c2, c3); // [c2_2, c3_2, c2_3, c3_3]
            let row0 = _mm_movelh_ps(t01_lo, t23_lo); // [c0_0, c1_0, c2_0, c3_0]
            let row1 = _mm_movehl_ps(t23_lo, t01_lo); // [c0_1, c1_1, c2_1, c3_1]
            let row2 = _mm_movelh_ps(t01_hi, t23_hi); // [c0_2, c1_2, c2_2, c3_2]
            let row3 = _mm_movehl_ps(t23_hi, t01_hi); // [c0_3, c1_3, c2_3, c3_3]

            // Broadcast each sample element
            let s0 = _mm_shuffle_ps(smp, smp, mm_shuffle(0, 0, 0, 0));
            let s1 = _mm_shuffle_ps(smp, smp, mm_shuffle(1, 1, 1, 1));
            let s2 = _mm_shuffle_ps(smp, smp, mm_shuffle(2, 2, 2, 2));
            let s3 = _mm_shuffle_ps(smp, smp, mm_shuffle(3, 3, 3, 3));

            // result = s0*row0 + s1*row1 + s2*row2 + s3*row3
            let result = _mm_add_ps(_mm_mul_ps(
                s0, row0),
                _mm_add_ps(_mm_mul_ps(
                    s1, row1),
                    _mm_add_ps(_mm_mul_ps(s2, row2), _mm_mul_ps(s3, row3))));
            _mm_storeu_ps(out_ptr.add(out_idx), result);

            out_idx += 4;
            coeff_idx += 16;
            j -= 4;
        }

        // Process pairs of outputs
        while j >= 2 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));
            let t2 = dot4x2(smp, c0, c1);
            *out_ptr.add(out_idx) = _mm_cvtss_f32(t2);
            *out_ptr.add(out_idx + 1) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2, t2, mm_shuffle(1, 1, 1, 1)),
            );
            out_idx += 2;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let coeffs = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            *out_ptr.add(out_idx) = dot4(smp, coeffs);
            out_idx += 1;
            coeff_idx += 4;
        }
    }
}

/// SSE2 vertical upscale for G (grayscale).
/// Mirrors oil_yscale_up_g_cmyk_sse2: 4-tap vertical blend with FMA + SSE2 packing.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_up_g(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let c0 = _mm_set1_ps(coeffs[0]);
    let c1 = _mm_set1_ps(coeffs[1]);
    let c2 = _mm_set1_ps(coeffs[2]);
    let c3 = _mm_set1_ps(coeffs[3]);
    let scale = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;

    // Clamp is omitted in SIMD paths: packs_epi32 (i32→i16 saturate) +
    // packus_epi16 (i16→u8 saturate) naturally clamp to [0, 255].

    // Process 16 pixels at a time
    while i + 15 < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

        let sum2 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 4))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 4))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 4))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 4))))));
        let idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

        let sum3 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 8))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 8))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 8))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 8))))));
        let idx3 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum3, scale), half));

        let sum4 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 12))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 12))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 12))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 12))))));
        let idx4 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum4, scale), half));

        let packed12 = _mm_packs_epi32(idx, idx2);
        let packed34 = _mm_packs_epi32(idx3, idx4);
        let result = _mm_packus_epi16(packed12, packed34);
        _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, result);
        i += 16;
    }

    // Process 8 pixels at a time
    while i + 7 < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));

        let sum2 = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 4))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 4))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 4))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 4))))));
        let idx2 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum2, scale), half));

        let packed = _mm_packs_epi32(idx, idx2);
        let result = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64(out_ptr.add(i) as *mut __m128i, result);
        i += 8;
    }

    // Process 4 pixels at a time
    while i + 3 < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(sum, scale), half));
        let packed = _mm_packs_epi32(idx, idx);
        let result = _mm_packus_epi16(packed, packed);
        *(out_ptr.add(i) as *mut i32) = _mm_cvtsi128_si32(result);
        i += 4;
    }

    // Scalar tail
    while i < len {
        let s = coeffs[0] * *l0.add(i) + coeffs[1] * *l1.add(i)
            + coeffs[2] * *l2.add(i) + coeffs[3] * *l3.add(i);
        let s = if s > 1.0 { 1.0 } else if s < 0.0 { 0.0 } else { s };
        *out_ptr.add(i) = (s * 255.0 + 0.5) as u8;
        i += 1;
    }
}

/// SSE2 downscale for G: horizontal x-filtering + y-accumulation.
/// Mirrors oil_scale_down_g_sse2.
#[target_feature(enable = "sse2")]
pub unsafe fn scale_down_g(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());
    let mut sum = _mm_setzero_ps();
    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 8 {
            let mut sum2 = _mm_setzero_ps();
            let mut sum3 = _mm_setzero_ps();
            let mut sum4 = _mm_setzero_ps();
            let mut j = 0;
            while j + 3 < border {
                let s0 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                let s1 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let s2 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                let s3 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
                let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));
                sum = _mm_add_ps(_mm_mul_ps(cx0, s0), sum);
                sum2 = _mm_add_ps(_mm_mul_ps(cx1, s1), sum2);
                sum3 = _mm_add_ps(_mm_mul_ps(cx2, s2), sum3);
                sum4 = _mm_add_ps(_mm_mul_ps(cx3, s3), sum4);
                in_idx += 4;
                cx_idx += 16;
                j += 4;
            }
            sum = _mm_add_ps(_mm_add_ps(sum, sum2), _mm_add_ps(sum3, sum4));
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
                in_idx += 1;
                cx_idx += 4;
                j += 1;
            }
        } else {
            for _ in 0..border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum = _mm_add_ps(_mm_mul_ps(cx, s), sum);
                in_idx += 1;
                cx_idx += 4;
            }
        }

        let sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample_y = _mm_shuffle_ps(sum, sum, mm_shuffle(0, 0, 0, 0));
        let sy_new = _mm_add_ps(_mm_mul_ps(cy, sample_y), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy_new);
        sy_idx += 4;

        sum = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 4));
    }
}

/// SSE2 vertical output for G downscale.
/// Extracts first element from each 4-element accumulator, clamps, converts to byte.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_out_g(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let scale = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);

    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut i = 0;
    let mut s_idx = 0;

    // Process 4 output values at a time
    while i + 3 < sl_len {
        let sp = s_ptr.add(s_idx) as *mut __m128i;
        let v0 = _mm_loadu_si128(sp);
        let v1 = _mm_loadu_si128(sp.add(1));
        let v2 = _mm_loadu_si128(sp.add(2));
        let v3 = _mm_loadu_si128(sp.add(3));

        // Extract first float from each 4-element accumulator
        let f0 = _mm_castsi128_ps(v0);
        let f1 = _mm_castsi128_ps(v1);
        let f2 = _mm_castsi128_ps(v2);
        let f3 = _mm_castsi128_ps(v3);
        let ab = _mm_shuffle_ps(f0, f1, mm_shuffle(0, 0, 0, 0));
        let cd = _mm_shuffle_ps(f2, f3, mm_shuffle(0, 0, 0, 0));
        let vals = _mm_shuffle_ps(ab, cd, mm_shuffle(2, 0, 2, 0));

        // Clamp to [0, 1], scale to [0, 255], add 0.5, truncate to int, pack to bytes
        let clamped = _mm_min_ps(_mm_max_ps(vals, zero), one);
        let idx = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(clamped, scale), half));
        let packed = _mm_packs_epi32(idx, idx);
        let result = _mm_packus_epi16(packed, packed);
        *(out_ptr.add(i) as *mut i32) = _mm_cvtsi128_si32(result);

        // Shift all 4 accumulators left
        _mm_storeu_si128(sp, _mm_srli_si128(v0, 4));
        _mm_storeu_si128(sp.add(1), _mm_srli_si128(v1, 4));
        _mm_storeu_si128(sp.add(2), _mm_srli_si128(v2, 4));
        _mm_storeu_si128(sp.add(3), _mm_srli_si128(v3, 4));

        s_idx += 16;
        i += 4;
    }

    // Scalar tail
    while i < sl_len {
        let val = *s_ptr.add(s_idx);
        let val = if val > 1.0 { 1.0 } else if val < 0.0 { 0.0 } else { val };
        *out_ptr.add(i) = (val * 255.0 + 0.5) as u8;
        // shift_left
        *s_ptr.add(s_idx) = *s_ptr.add(s_idx + 1);
        *s_ptr.add(s_idx + 1) = *s_ptr.add(s_idx + 2);
        *s_ptr.add(s_idx + 2) = *s_ptr.add(s_idx + 3);
        *s_ptr.add(s_idx + 3) = 0.0;
        s_idx += 4;
        i += 1;
    }
}

// --- CMYK SSE2 ---

/// SSE2 horizontal upscale for CMYK.
/// Interleaved layout: each smpN = [C, M, Y, K] for one tap position.
/// Mirrors oil_xscale_up_cmyk_sse2.
#[target_feature(enable = "sse2")]
pub unsafe fn xscale_up_cmyk(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let inv255 = _mm_set1_ps(1.0 / 255.0);
    let zero_i = _mm_setzero_si128();
    let mut smp0;
    let mut smp1 = _mm_setzero_ps();
    let mut smp2 = _mm_setzero_ps();
    let mut smp3 = _mm_setzero_ps();
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        // Load 4 bytes [C,M,Y,K], unpack to 4 floats, divide by 255
        let px = _mm_cvtsi32_si128(*(in_ptr.add(i * 4) as *const i32));
        let px = _mm_unpacklo_epi8(px, zero_i);
        let px = _mm_unpacklo_epi16(px, zero_i);
        smp0 = smp1;
        smp1 = smp2;
        smp2 = smp3;
        smp3 = _mm_mul_ps(_mm_cvtepi32_ps(px), inv255);

        let mut j = *border_ptr.add(i);

        // Process pairs of outputs
        while j >= 2 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));

            let result0 = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(smp0, _mm_shuffle_ps(c0, c0, mm_shuffle(0, 0, 0, 0))),
                    _mm_mul_ps(smp1, _mm_shuffle_ps(c0, c0, mm_shuffle(1, 1, 1, 1))),
                ),
                _mm_add_ps(
                    _mm_mul_ps(smp2, _mm_shuffle_ps(c0, c0, mm_shuffle(2, 2, 2, 2))),
                    _mm_mul_ps(smp3, _mm_shuffle_ps(c0, c0, mm_shuffle(3, 3, 3, 3))),
                ),
            );

            let result1 = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(smp0, _mm_shuffle_ps(c1, c1, mm_shuffle(0, 0, 0, 0))),
                    _mm_mul_ps(smp1, _mm_shuffle_ps(c1, c1, mm_shuffle(1, 1, 1, 1))),
                ),
                _mm_add_ps(
                    _mm_mul_ps(smp2, _mm_shuffle_ps(c1, c1, mm_shuffle(2, 2, 2, 2))),
                    _mm_mul_ps(smp3, _mm_shuffle_ps(c1, c1, mm_shuffle(3, 3, 3, 3))),
                ),
            );

            _mm_storeu_ps(out_ptr.add(out_idx), result0);
            _mm_storeu_ps(out_ptr.add(out_idx + 4), result1);

            out_idx += 8;
            coeff_idx += 8;
            j -= 2;
        }

        // Process remaining single output
        if j > 0 {
            let c = _mm_loadu_ps(coeff_ptr.add(coeff_idx));

            let result = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(smp0, _mm_shuffle_ps(c, c, mm_shuffle(0, 0, 0, 0))),
                    _mm_mul_ps(smp1, _mm_shuffle_ps(c, c, mm_shuffle(1, 1, 1, 1))),
                ),
                _mm_add_ps(
                    _mm_mul_ps(smp2, _mm_shuffle_ps(c, c, mm_shuffle(2, 2, 2, 2))),
                    _mm_mul_ps(smp3, _mm_shuffle_ps(c, c, mm_shuffle(3, 3, 3, 3))),
                ),
            );

            _mm_storeu_ps(out_ptr.add(out_idx), result);

            out_idx += 4;
            coeff_idx += 4;
        }
    }
}

/// SSE2 downscale for CMYK: horizontal x-filtering with i2f + y-accumulation.
/// Mirrors oil_scale_down_cmyk_sse2.
#[target_feature(enable = "sse2")]
pub unsafe fn scale_down_cmyk(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());

    let mut sum_c = _mm_setzero_ps();
    let mut sum_m = _mm_setzero_ps();
    let mut sum_yc = _mm_setzero_ps();
    let mut sum_k = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 4 {
            let mut sum_c2 = _mm_setzero_ps();
            let mut sum_m2 = _mm_setzero_ps();
            let mut sum_yc2 = _mm_setzero_ps();
            let mut sum_k2 = _mm_setzero_ps();

            let mut j = 0;
            while j + 1 < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_c = _mm_add_ps(_mm_mul_ps(cx, s), sum_c);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_m = _mm_add_ps(_mm_mul_ps(cx, s), sum_m);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_yc = _mm_add_ps(_mm_mul_ps(cx, s), sum_yc);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_k = _mm_add_ps(_mm_mul_ps(cx, s), sum_k);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                sum_c2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_c2);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                sum_m2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_m2);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 6) as usize));
                sum_yc2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_yc2);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 7) as usize));
                sum_k2 = _mm_add_ps(_mm_mul_ps(cx2, s), sum_k2);

                in_idx += 8;
                cx_idx += 8;
                j += 2;
            }

            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_c = _mm_add_ps(_mm_mul_ps(cx, s), sum_c);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_m = _mm_add_ps(_mm_mul_ps(cx, s), sum_m);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_yc = _mm_add_ps(_mm_mul_ps(cx, s), sum_yc);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_k = _mm_add_ps(_mm_mul_ps(cx, s), sum_k);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }

            sum_c = _mm_add_ps(sum_c, sum_c2);
            sum_m = _mm_add_ps(sum_m, sum_m2);
            sum_yc = _mm_add_ps(sum_yc, sum_yc2);
            sum_k = _mm_add_ps(sum_k, sum_k2);
        } else {
            let mut j = 0;
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_c = _mm_add_ps(_mm_mul_ps(cx, s), sum_c);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                sum_m = _mm_add_ps(_mm_mul_ps(cx, s), sum_m);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                sum_yc = _mm_add_ps(_mm_mul_ps(cx, s), sum_yc);

                let s = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                sum_k = _mm_add_ps(_mm_mul_ps(cx, s), sum_k);

                in_idx += 4;
                cx_idx += 4;
                j += 1;
            }
        }

        // C channel
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_c, sum_c, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // M channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_m, sum_m, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Y channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_yc, sum_yc, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // K channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_k, sum_k, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // shift_left for each channel
        sum_c = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_c), 4));
        sum_m = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_m), 4));
        sum_yc = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_yc), 4));
        sum_k = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_k), 4));
    }
}

// --- Grayscale + Alpha (GA) SSE2 ---

/// SSE2 horizontal upscale for GA (grayscale with premultiplied alpha).
/// Two sliding windows: gray (premultiplied) and alpha.
#[target_feature(enable = "sse2")]
pub unsafe fn xscale_up_ga(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let mut smp_g = _mm_setzero_ps();
    let mut smp_a = _mm_setzero_ps();
    let out_ptr = out.as_mut_ptr();
    let coeff_ptr = coeff_buf.as_ptr();
    let border_ptr = border_buf.as_ptr();
    let in_ptr = input.as_ptr();
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 2;
        let alpha = *in_ptr.add(in_base + 1) as f32 / 255.0;
        smp_a = push_f_sse2(smp_a, alpha);
        // Extract alpha we just pushed (position 3) to premultiply gray
        let gray = *in_ptr.add(in_base) as f32 / 255.0;
        smp_g = push_f_sse2(smp_g, alpha * gray);

        let mut j = *border_ptr.add(i);

        while j >= 2 {
            let c0 = _mm_loadu_ps(coeff_ptr.add(coeff_idx));
            let c1 = _mm_loadu_ps(coeff_ptr.add(coeff_idx + 4));

            let t2_g = dot4x2(smp_g, c0, c1);
            let t2_a = dot4x2(smp_a, c0, c1);

            *out_ptr.add(out_idx)     = _mm_cvtss_f32(t2_g);
            *out_ptr.add(out_idx + 1) = _mm_cvtss_f32(t2_a);
            *out_ptr.add(out_idx + 2) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_g, t2_g, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 3) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_a, t2_a, mm_shuffle(1, 1, 1, 1)),
            );

            out_idx += 4;
            coeff_idx += 8;
            j -= 2;
        }

        if j > 0 {
            let coeffs = _mm_loadu_ps(coeff_ptr.add(coeff_idx));

            *out_ptr.add(out_idx)     = dot4(smp_g, coeffs);
            *out_ptr.add(out_idx + 1) = dot4(smp_a, coeffs);

            out_idx += 2;
            coeff_idx += 4;
        }
    }
}

/// SSE2 vertical upscale for GA.
/// Vectorized un-premultiply, clamp, and pack to u8.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_up_ga(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let c0 = _mm_set1_ps(coeffs[0]);
    let c1 = _mm_set1_ps(coeffs[1]);
    let c2 = _mm_set1_ps(coeffs[2]);
    let c3 = _mm_set1_ps(coeffs[3]);
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let v255 = _mm_set1_ps(255.0);
    let half = _mm_set1_ps(0.5);
    // Mask selecting alpha channel positions (1 and 3) in [g, a, g, a]
    let alpha_mask = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();
    let out_ptr = out.as_mut_ptr();

    let mut i = 0;

    // Process 4 pixels (8 floats = 2 x __m128) at a time
    while i + 7 < len {
        // Vertical blend for first 2 pixels using FMA: c0*v0 + c1*v1 + c2*v2 + c3*v3
        let sum_lo = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));
        // Vertical blend for next 2 pixels
        let sum_hi = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i + 4))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i + 4))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i + 4))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i + 4))))));

        // Un-premultiply and pack first pair
        let result_lo = yscale_up_ga_unpremultiply(sum_lo, alpha_mask, zero, one, v255, half);
        // Un-premultiply and pack second pair
        let result_hi = yscale_up_ga_unpremultiply(sum_hi, alpha_mask, zero, one, v255, half);

        // Pack i32 -> i16 -> u8
        let packed16 = _mm_packs_epi32(result_lo, result_hi);
        let packed8 = _mm_packus_epi16(packed16, packed16);

        // Store 8 bytes (4 pixels x 2 channels)
        std::ptr::write_unaligned(
            out_ptr.add(i) as *mut u64,
            _mm_cvtsi128_si64(packed8) as u64,
        );

        i += 8;
    }

    // Process 2 pixels (4 floats) at a time
    while i + 3 < len {
        let sum = _mm_add_ps(_mm_mul_ps(
            c0, _mm_loadu_ps(l0.add(i))),
            _mm_add_ps(_mm_mul_ps(
                c1, _mm_loadu_ps(l1.add(i))),
                _mm_add_ps(_mm_mul_ps(
                    c2, _mm_loadu_ps(l2.add(i))),
                    _mm_mul_ps(c3, _mm_loadu_ps(l3.add(i))))));

        let result = yscale_up_ga_unpremultiply(sum, alpha_mask, zero, one, v255, half);

        // Pack and store 4 bytes (2 pixels)
        let packed16 = _mm_packs_epi32(result, result);
        let packed8 = _mm_packus_epi16(packed16, packed16);
        std::ptr::write_unaligned(
            out_ptr.add(i) as *mut u32,
            _mm_cvtsi128_si32(packed8) as u32,
        );

        i += 4;
    }

    // Process remaining single pixel (2 floats)
    while i + 1 < len {
        let g = coeffs[0] * *l0.add(i) + coeffs[1] * *l1.add(i)
            + coeffs[2] * *l2.add(i) + coeffs[3] * *l3.add(i);
        let a = coeffs[0] * *l0.add(i + 1) + coeffs[1] * *l1.add(i + 1)
            + coeffs[2] * *l2.add(i + 1) + coeffs[3] * *l3.add(i + 1);
        let alpha = if a < 0.0 { 0.0 } else if a > 1.0 { 1.0 } else { a };
        let mut gray = g;
        if alpha != 0.0 {
            gray /= alpha;
        }
        let gray = if gray < 0.0 { 0.0 } else if gray > 1.0 { 1.0 } else { gray };
        *out_ptr.add(i) = (gray * 255.0 + 0.5) as u8;
        *out_ptr.add(i + 1) = (alpha * 255.0 + 0.5) as u8;
        i += 2;
    }
}

/// Vectorized un-premultiply, clamp, scale, and convert to i32 for a [g, a, g, a] vector.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn yscale_up_ga_unpremultiply(
    sum: __m128,
    alpha_mask: __m128,
    zero: __m128,
    one: __m128,
    v255: __m128,
    half: __m128,
) -> __m128i {
    // Spread alpha to both channels: [g0, a0, g1, a1] -> [a0, a0, a1, a1]
    let alpha_spread = _mm_shuffle_ps(sum, sum, mm_shuffle(3, 3, 1, 1));
    let alpha_clamped = _mm_min_ps(_mm_max_ps(alpha_spread, zero), one);

    // Safe division: where alpha == 0, substitute 1.0 to avoid inf/nan
    let nonzero = _mm_cmpneq_ps(alpha_clamped, zero);
    let safe_alpha = _mm_or_ps(
        _mm_and_ps(nonzero, alpha_clamped),
        _mm_andnot_ps(nonzero, one),
    );
    let divided = _mm_div_ps(sum, safe_alpha);
    let clamped = _mm_min_ps(_mm_max_ps(divided, zero), one);

    // Merge: gray channels from divided, alpha channels from original clamped
    let result = _mm_or_ps(
        _mm_andnot_ps(alpha_mask, clamped),
        _mm_and_ps(alpha_mask, alpha_clamped),
    );

    // Scale to [0, 255] with rounding and convert to i32
    _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(result, v255), half))
}

/// SSE2 downscale for GA: horizontal x-filtering with premultiplied alpha + y-accumulation.
#[target_feature(enable = "sse2")]
pub unsafe fn scale_down_ga(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let i2f = tables.i2f.as_ptr();
    let cy = _mm_loadu_ps(coeffs_y.as_ptr());

    let mut sum_g = _mm_setzero_ps();
    let mut sum_a = _mm_setzero_ps();

    let in_ptr = input.as_ptr();
    let cx_ptr = coeffs_x.as_ptr();
    let sy_ptr = sums_y.as_mut_ptr();
    let border_ptr = border_buf.as_ptr();

    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        let border = *border_ptr.add(i);

        if border >= 8 {
            let mut sum_g2 = _mm_setzero_ps();
            let mut sum_g3 = _mm_setzero_ps();
            let mut sum_g4 = _mm_setzero_ps();
            let mut sum_a2 = _mm_setzero_ps();
            let mut sum_a3 = _mm_setzero_ps();
            let mut sum_a4 = _mm_setzero_ps();
            let mut j = 0;
            while j + 3 < border {
                let cx0 = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let cx1 = _mm_loadu_ps(cx_ptr.add(cx_idx + 4));
                let cx2 = _mm_loadu_ps(cx_ptr.add(cx_idx + 8));
                let cx3 = _mm_loadu_ps(cx_ptr.add(cx_idx + 12));

                let a0 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let a1 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 3) as usize));
                let a2 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 5) as usize));
                let a3 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 7) as usize));

                let cx_a0 = _mm_mul_ps(cx0, a0);
                let cx_a1 = _mm_mul_ps(cx1, a1);
                let cx_a2 = _mm_mul_ps(cx2, a2);
                let cx_a3 = _mm_mul_ps(cx3, a3);

                let g0 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                let g1 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 2) as usize));
                let g2 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 4) as usize));
                let g3 = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 6) as usize));

                sum_g  = _mm_add_ps(_mm_mul_ps(cx_a0, g0), sum_g);
                sum_g2 = _mm_add_ps(_mm_mul_ps(cx_a1, g1), sum_g2);
                sum_g3 = _mm_add_ps(_mm_mul_ps(cx_a2, g2), sum_g3);
                sum_g4 = _mm_add_ps(_mm_mul_ps(cx_a3, g3), sum_g4);

                sum_a  = _mm_add_ps(cx_a0, sum_a);
                sum_a2 = _mm_add_ps(cx_a1, sum_a2);
                sum_a3 = _mm_add_ps(cx_a2, sum_a3);
                sum_a4 = _mm_add_ps(cx_a3, sum_a4);

                in_idx += 8;
                cx_idx += 16;
                j += 4;
            }
            sum_g = _mm_add_ps(_mm_add_ps(sum_g, sum_g2), _mm_add_ps(sum_g3, sum_g4));
            sum_a = _mm_add_ps(_mm_add_ps(sum_a, sum_a2), _mm_add_ps(sum_a3, sum_a4));
            while j < border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let a = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let cx_a = _mm_mul_ps(cx, a);
                let g = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, g), sum_g);
                sum_a = _mm_add_ps(cx_a, sum_a);
                in_idx += 2;
                cx_idx += 4;
                j += 1;
            }
        } else {
            for _ in 0..border {
                let cx = _mm_loadu_ps(cx_ptr.add(cx_idx));
                let a = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx + 1) as usize));
                let cx_a = _mm_mul_ps(cx, a);
                let g = _mm_set1_ps(*i2f.add(*in_ptr.add(in_idx) as usize));
                sum_g = _mm_add_ps(_mm_mul_ps(cx_a, g), sum_g);
                sum_a = _mm_add_ps(cx_a, sum_a);
                in_idx += 2;
                cx_idx += 4;
            }
        }

        // Accumulate gray into y sums
        let mut sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_g, sum_g, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // Accumulate alpha into y sums
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_a, sum_a, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
        sy_idx += 4;

        // shift_left for each channel
        sum_g = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_g), 4));
        sum_a = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum_a), 4));
    }
}

/// SSE2 output for downscaled GA: un-premultiply gray, convert to bytes.
#[target_feature(enable = "sse2")]
pub unsafe fn yscale_out_ga(sums: &mut [f32], width: u32, out: &mut [u8]) {
    let s_ptr = sums.as_mut_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut s_idx = 0usize;
    let mut o_idx = 0usize;

    for _ in 0..width {
        let sp = s_ptr.add(s_idx) as *mut __m128i;

        // Load 2 accumulators: [G0..G3], [A0..A3]
        let v0 = _mm_loadu_si128(sp);
        let v1 = _mm_loadu_si128(sp.add(1));

        // Extract first element of each: gray, alpha
        let gray_val = _mm_cvtss_f32(_mm_castsi128_ps(v0));
        let alpha_val = _mm_cvtss_f32(_mm_castsi128_ps(v1));

        // Clamp alpha
        let alpha = if alpha_val < 0.0 { 0.0 } else if alpha_val > 1.0 { 1.0 } else { alpha_val };

        // Un-premultiply gray
        let mut gray = gray_val;
        if alpha != 0.0 {
            gray /= alpha;
        }
        let gray = if gray < 0.0 { 0.0 } else if gray > 1.0 { 1.0 } else { gray };

        *out_ptr.add(o_idx) = (gray * 255.0 + 0.5) as u8;
        *out_ptr.add(o_idx + 1) = (alpha * 255.0 + 0.5) as u8;

        // Shift both accumulators left
        _mm_storeu_si128(sp, _mm_srli_si128(v0, 4));
        _mm_storeu_si128(sp.add(1), _mm_srli_si128(v1, 4));

        s_idx += 8;
        o_idx += 2;
    }
}

// --- Helpers ---

/// SSE2 push_f: shift left by one float, insert new value at position 3.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn push_f_sse2(v: __m128, val: f32) -> __m128 {
    let shifted = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 4));
    let newval = _mm_set_ss(val);
    let hi = _mm_shuffle_ps(shifted, newval, mm_shuffle(0, 0, 3, 2));
    _mm_shuffle_ps(shifted, hi, mm_shuffle(2, 0, 1, 0))
}

/// Compute dot product of 4-element vector with one set of coefficients.
/// Returns the scalar result in the lowest float lane.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn dot4(smp: __m128, coeffs: __m128) -> f32 {
    let prod = _mm_mul_ps(smp, coeffs);
    let t1 = _mm_movehl_ps(prod, prod);
    let t2 = _mm_add_ps(prod, t1);
    let t3 = _mm_shuffle_ps(t2, t2, mm_shuffle(1, 1, 1, 1));
    let t4 = _mm_add_ss(t2, t3);
    _mm_cvtss_f32(t4)
}

/// Compute two dot products simultaneously (smp * c0 and smp * c1).
/// Returns [dot0, dot1, ?, ?] in the __m128 result.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn dot4x2(smp: __m128, c0: __m128, c1: __m128) -> __m128 {
    let p0 = _mm_mul_ps(smp, c0);
    let p1 = _mm_mul_ps(smp, c1);
    let lo = _mm_unpacklo_ps(p0, p1);
    let hi = _mm_unpackhi_ps(p0, p1);
    let sum = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(sum, sum);
    _mm_add_ps(sum, t1)
}
