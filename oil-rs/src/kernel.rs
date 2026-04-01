const TAPS: usize = 4;

/// Catmull-Rom interpolation kernel.
#[inline]
pub fn catrom(x: f32) -> f32 {
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else {
        (((5.0 - x) * x - 8.0) * x + 4.0) / 2.0
    }
}

/// Calculate the number of taps needed for resampling.
/// Upscale always uses 4 taps. Downscale widens the kernel to prevent aliasing.
pub fn calc_taps(dim_in: u32, dim_out: u32) -> usize {
    if dim_out > dim_in {
        TAPS
    } else {
        let tmp = (TAPS as u32 * dim_in / dim_out) as usize;
        tmp - (tmp & 1)
    }
}

/// Map from discrete output coordinate to continuous source coordinate.
pub fn map(dim_in: u32, dim_out: u32, pos: u32) -> f64 {
    (pos as f64 + 0.5) * (dim_in as f64 / dim_out as f64) - 0.5
}

/// Split mapped coordinate into integer position and fractional remainder.
pub fn split_map(dim_in: u32, dim_out: u32, pos: u32) -> (i32, f32) {
    let smp = map(dim_in, dim_out, pos);
    let smp_i = if smp < 0.0 { -1 } else { smp as i32 };
    let rest = (smp - smp_i as f64) as f32;
    (smp_i, rest)
}

/// Calculate filter coefficients for a given fractional offset.
/// Coefficients are normalized to sum to 1.0.
pub fn calc_coeffs(coeffs: &mut [f32], tx: f32, taps: usize, ltrim: usize, rtrim: usize) {
    let tap_mult = taps as f32 / TAPS as f32;
    let mut pos = 1.0 - tx - (taps / 2) as f32 + ltrim as f32;
    let mut fudge = 0.0f32;

    for i in ltrim..(taps - rtrim) {
        let tmp = catrom(pos.abs() / tap_mult) / tap_mult;
        fudge += tmp;
        coeffs[i] = tmp;
        pos += 1.0;
    }

    let fudge = 1.0 / fudge;
    for i in ltrim..(taps - rtrim) {
        coeffs[i] *= fudge;
    }
}

/// Pre-calculate coefficients and border counters for upscaling.
///
/// coeff_buf: 4 coefficients per output sample (4 * out_dim total).
/// border_buf: number of output samples to produce for each input sample (in_dim entries).
pub fn scale_up_coeffs(
    in_dim: u32,
    out_dim: u32,
    coeff_buf: &mut [f32],
    border_buf: &mut [i32],
) {
    let max_pos = in_dim as i32 - 1;
    let mut coeff_offset = 0usize;

    for i in 0..out_dim {
        let (smp_i, tx) = split_map(in_dim, out_dim, i);
        let start = smp_i - 1;
        let end = smp_i + 2;

        let safe_end = end.min(max_pos) as usize;

        let mut ltrim = 0usize;
        let mut rtrim = 0usize;
        if start < 0 {
            ltrim = (-start) as usize;
        }
        if end > max_pos {
            rtrim = (end - max_pos) as usize;
        }

        border_buf[safe_end] += 1;

        calc_coeffs(
            &mut coeff_buf[coeff_offset + rtrim..],
            tx,
            4,
            ltrim,
            rtrim,
        );

        coeff_offset += 4;
    }
}

/// Pre-calculate coefficients and border counters for downscaling.
///
/// coeff_buf: 4 coefficients per input sample (4 * in_dim total).
/// border_buf: number of input samples to process before next output is finished (out_dim entries).
/// tmp_coeffs: temporary buffer of size >= taps.
pub fn scale_down_coeffs(
    in_dim: u32,
    out_dim: u32,
    coeff_buf: &mut [f32],
    border_buf: &mut [i32],
    tmp_coeffs: &mut [f32],
) {
    let taps = calc_taps(in_dim as u32, out_dim as u32);
    let mut ends = [-1i32; 4];

    for i in 0..out_dim as usize {
        let (smp_i, tx) = split_map(in_dim, out_dim, i as u32);

        let smp_start = smp_i - (taps as i32 / 2 - 1);
        let mut smp_end = smp_i + taps as i32 / 2;
        if smp_end >= in_dim as i32 {
            smp_end = in_dim as i32 - 1;
        }
        ends[i % 4] = smp_end;
        border_buf[i] = smp_end - ends[(i + 3) % 4];

        let mut ltrim = 0usize;
        if smp_start < 0 {
            ltrim = (-smp_start) as usize;
        }
        let rtrim = (smp_start + taps as i32 - 1 - smp_end) as usize;

        // Zero tmp_coeffs before use
        for c in tmp_coeffs.iter_mut().take(taps) {
            *c = 0.0;
        }
        calc_coeffs(tmp_coeffs, tx, taps, ltrim, rtrim);

        for j in ltrim..(taps - rtrim) {
            let pos = (smp_start + j as i32) as usize;

            let offset = if pos as i32 > ends[(i + 3) % 4] {
                0
            } else if pos as i32 > ends[(i + 2) % 4] {
                1
            } else if pos as i32 > ends[(i + 1) % 4] {
                2
            } else {
                3
            };

            coeff_buf[pos * 4 + offset] = tmp_coeffs[j];
        }
    }
}
