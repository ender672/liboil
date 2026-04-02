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
    let scale = _mm_set1_ps((tables.l2s_len - 1) as f32);

    let c0 = _mm_set1_ps(coeffs[0]);
    let c1 = _mm_set1_ps(coeffs[1]);
    let c2 = _mm_set1_ps(coeffs[2]);
    let c3 = _mm_set1_ps(coeffs[3]);

    let mut i = 0;
    let mut idx_buf: [i32; 8] = [0i32; 8];
    let idx_ptr = idx_buf.as_mut_ptr() as *mut __m128i;
    let out_ptr = out.as_mut_ptr();

    let l0 = lines[0].as_ptr();
    let l1 = lines[1].as_ptr();
    let l2 = lines[2].as_ptr();
    let l3 = lines[3].as_ptr();

    // Process 8 floats at a time
    while i + 7 < len {
        let v0 = _mm_loadu_ps(l0.add(i));
        let v1 = _mm_loadu_ps(l1.add(i));
        let v2 = _mm_loadu_ps(l2.add(i));
        let v3 = _mm_loadu_ps(l3.add(i));
        let sum = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
            _mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)),
        );
        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(_mm_mul_ps(sum, scale)));

        let v0b = _mm_loadu_ps(l0.add(i + 4));
        let v1b = _mm_loadu_ps(l1.add(i + 4));
        let v2b = _mm_loadu_ps(l2.add(i + 4));
        let v3b = _mm_loadu_ps(l3.add(i + 4));
        let sum2 = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(c0, v0b), _mm_mul_ps(c1, v1b)),
            _mm_add_ps(_mm_mul_ps(c2, v2b), _mm_mul_ps(c3, v3b)),
        );
        _mm_storeu_si128(idx_ptr.add(1), _mm_cvttps_epi32(_mm_mul_ps(sum2, scale)));

        *out_ptr.add(i)     = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1) = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2) = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3) = *lut.offset(idx_buf[3] as isize);
        *out_ptr.add(i + 4) = *lut.offset(idx_buf[4] as isize);
        *out_ptr.add(i + 5) = *lut.offset(idx_buf[5] as isize);
        *out_ptr.add(i + 6) = *lut.offset(idx_buf[6] as isize);
        *out_ptr.add(i + 7) = *lut.offset(idx_buf[7] as isize);

        i += 8;
    }

    // Process 4 floats at a time
    while i + 3 < len {
        let v0 = _mm_loadu_ps(l0.add(i));
        let v1 = _mm_loadu_ps(l1.add(i));
        let v2 = _mm_loadu_ps(l2.add(i));
        let v3 = _mm_loadu_ps(l3.add(i));
        let sum = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
            _mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)),
        );
        _mm_storeu_si128(idx_ptr, _mm_cvttps_epi32(_mm_mul_ps(sum, scale)));
        *out_ptr.add(i)     = *lut.offset(idx_buf[0] as isize);
        *out_ptr.add(i + 1) = *lut.offset(idx_buf[1] as isize);
        *out_ptr.add(i + 2) = *lut.offset(idx_buf[2] as isize);
        *out_ptr.add(i + 3) = *lut.offset(idx_buf[3] as isize);
        i += 4;
    }

    // Scalar tail
    while i < len {
        let val = *coeffs.get_unchecked(0) * *l0.add(i)
            + *coeffs.get_unchecked(1) * *l1.add(i)
            + *coeffs.get_unchecked(2) * *l2.add(i)
            + *coeffs.get_unchecked(3) * *l3.add(i);
        *out_ptr.add(i) = *lut.offset((val * (tables.l2s_len - 1) as f32) as isize);
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

        if border >= 4 {
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

            *out_ptr.add(out_idx)     = _mm_cvtss_f32(t2_r);
            *out_ptr.add(out_idx + 1) = _mm_cvtss_f32(t2_g);
            *out_ptr.add(out_idx + 2) = _mm_cvtss_f32(t2_b);
            *out_ptr.add(out_idx + 3) = _mm_cvtss_f32(t2_a);
            *out_ptr.add(out_idx + 4) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_r, t2_r, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 5) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_g, t2_g, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 6) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_b, t2_b, mm_shuffle(1, 1, 1, 1)),
            );
            *out_ptr.add(out_idx + 7) = _mm_cvtss_f32(
                _mm_shuffle_ps(t2_a, t2_a, mm_shuffle(1, 1, 1, 1)),
            );

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
    while i < len {
        // Load one RGBA pixel's worth of data from each line
        let v0 = _mm_loadu_ps(l0.add(i));
        let v1 = _mm_loadu_ps(l1.add(i));
        let v2 = _mm_loadu_ps(l2.add(i));
        let v3 = _mm_loadu_ps(l3.add(i));
        let sum = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(c0, v0), _mm_mul_ps(c1, v1)),
            _mm_add_ps(_mm_mul_ps(c2, v2), _mm_mul_ps(c3, v3)),
        );

        // Clamp alpha to [0, 1]
        let alpha_v = _mm_shuffle_ps(sum, sum, mm_shuffle(3, 3, 3, 3));
        let alpha_v = _mm_min_ps(_mm_max_ps(alpha_v, zero), one);
        let alpha = _mm_cvtss_f32(alpha_v);

        // Divide RGB by alpha (skip if alpha == 0)
        let mut vals = sum;
        if alpha != 0.0 {
            vals = _mm_mul_ps(vals, _mm_rcp_ps(alpha_v));
        }

        // Clamp to [0, 1] and compute l2s indices
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

        // A channel
        sy = _mm_loadu_ps(sy_ptr.add(sy_idx));
        let sample = _mm_shuffle_ps(sum_a, sum_a, mm_shuffle(0, 0, 0, 0));
        sy = _mm_add_ps(_mm_mul_ps(cy, sample), sy);
        _mm_storeu_ps(sy_ptr.add(sy_idx), sy);
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
