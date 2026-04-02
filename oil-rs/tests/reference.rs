use oil::ColorSpace;
use oil::OilScale;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

static WORST: Mutex<f64> = Mutex::new(0.0);

fn srgb_to_linear_ref(in_f: f64) -> f64 {
    if in_f <= 0.0404482362771082 {
        in_f / 12.92
    } else {
        ((in_f + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_ref(in_f: f64) -> f64 {
    if in_f <= 0.0 {
        return 0.0;
    }
    if in_f >= 1.0 {
        return 1.0;
    }
    if in_f <= 0.00313066844250063 {
        in_f * 12.92
    } else {
        1.055 * in_f.powf(1.0 / 2.4) - 0.055
    }
}

fn cubic(b: f64, c: f64, x: f64) -> f64 {
    if x < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * x * x * x
            + (-18.0 + 12.0 * b + 6.0 * c) * x * x
            + (6.0 - 2.0 * b))
            / 6.0
    } else if x < 2.0 {
        ((-b - 6.0 * c) * x * x * x
            + (6.0 * b + 30.0 * c) * x * x
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn ref_catrom(x: f64) -> f64 {
    cubic(0.0, 0.5, x)
}

fn ref_calc_coeffs(coeffs: &mut [f64], offset: f64, taps: usize, ltrim: usize, rtrim: usize) {
    assert!(taps - ltrim - rtrim > 0);
    let tap_mult = taps as f64 / 4.0;
    let mut fudge = 0.0;

    for i in 0..taps {
        if i < ltrim || i >= taps - rtrim {
            coeffs[i] = 0.0;
            continue;
        }
        let tap_offset = 1.0 - offset - (taps as f64) / 2.0 + i as f64;
        coeffs[i] = ref_catrom(tap_offset.abs() / tap_mult) / tap_mult;
        fudge += coeffs[i];
    }

    let mut total_check = 0.0;
    for i in 0..taps {
        coeffs[i] /= fudge;
        total_check += coeffs[i];
    }
    assert!(
        (total_check - 1.0).abs() < 0.0000000001,
        "coefficients don't sum to 1.0: {}",
        total_check
    );
}

fn calc_taps_check(dim_in: u32, dim_out: u32) -> usize {
    if dim_in < dim_out {
        4
    } else {
        let tmp = (dim_in as usize * 4) / dim_out as usize;
        tmp - (tmp % 2)
    }
}

fn ref_map(dim_in: u32, dim_out: u32, pos: u32) -> f64 {
    (pos as f64 + 0.5) * (dim_in as f64 / dim_out as f64) - 0.5
}

fn split_map_check(dim_in: u32, dim_out: u32, pos: u32) -> (i32, f64) {
    let smp = ref_map(dim_in, dim_out, pos);
    let smp_i = smp.floor() as i32;
    let ty = smp - smp_i as f64;
    (smp_i, ty)
}

fn clamp_f(val: f64) -> f64 {
    val.clamp(0.0, 1.0)
}

fn preprocess(pixel: &mut [f64], cs: ColorSpace) {
    match cs {
        ColorSpace::G | ColorSpace::CMYK | ColorSpace::Unknown => {}
        ColorSpace::GA => {
            pixel[0] *= pixel[1];
        }
        ColorSpace::RGB => {
            pixel[0] = srgb_to_linear_ref(pixel[0]);
            pixel[1] = srgb_to_linear_ref(pixel[1]);
            pixel[2] = srgb_to_linear_ref(pixel[2]);
        }
        ColorSpace::RGBA => {
            pixel[0] = pixel[3] * srgb_to_linear_ref(pixel[0]);
            pixel[1] = pixel[3] * srgb_to_linear_ref(pixel[1]);
            pixel[2] = pixel[3] * srgb_to_linear_ref(pixel[2]);
        }
        ColorSpace::ARGB => {
            pixel[1] = pixel[0] * srgb_to_linear_ref(pixel[1]);
            pixel[2] = pixel[0] * srgb_to_linear_ref(pixel[2]);
            pixel[3] = pixel[0] * srgb_to_linear_ref(pixel[3]);
        }
        ColorSpace::RGBX => {
            pixel[0] = srgb_to_linear_ref(pixel[0]);
            pixel[1] = srgb_to_linear_ref(pixel[1]);
            pixel[2] = srgb_to_linear_ref(pixel[2]);
            pixel[3] = 1.0;
        }
    }
}

fn postprocess(pixel: &mut [f64], cs: ColorSpace) {
    match cs {
        ColorSpace::G => {
            pixel[0] = clamp_f(pixel[0]);
        }
        ColorSpace::GA => {
            let alpha = clamp_f(pixel[1]);
            if alpha != 0.0 {
                pixel[0] /= alpha;
            }
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = alpha;
        }
        ColorSpace::RGB => {
            pixel[0] = linear_to_srgb_ref(pixel[0]);
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
        }
        ColorSpace::RGBA => {
            let alpha = clamp_f(pixel[3]);
            if alpha != 0.0 {
                pixel[0] /= alpha;
                pixel[1] /= alpha;
                pixel[2] /= alpha;
            }
            pixel[0] = linear_to_srgb_ref(pixel[0]);
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
            pixel[3] = alpha;
        }
        ColorSpace::ARGB => {
            let alpha = clamp_f(pixel[0]);
            if alpha != 0.0 {
                pixel[1] /= alpha;
                pixel[2] /= alpha;
                pixel[3] /= alpha;
            }
            pixel[0] = alpha;
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
            pixel[3] = linear_to_srgb_ref(pixel[3]);
        }
        ColorSpace::CMYK => {
            pixel[0] = clamp_f(pixel[0]);
            pixel[1] = clamp_f(pixel[1]);
            pixel[2] = clamp_f(pixel[2]);
            pixel[3] = clamp_f(pixel[3]);
        }
        ColorSpace::RGBX => {
            pixel[0] = linear_to_srgb_ref(pixel[0]);
            pixel[1] = linear_to_srgb_ref(pixel[1]);
            pixel[2] = linear_to_srgb_ref(pixel[2]);
            pixel[3] = 1.0;
        }
        ColorSpace::Unknown => {}
    }
}

fn validate_scanline8(oil: &[u8], reference: &[f64], width: usize, cmp: usize) {
    for i in 0..width {
        for j in 0..cmp {
            let pos = i * cmp + j;
            let ref_f = reference[pos] * 255.0;
            let ref_i = ref_f.round() as i32;
            let error = (oil[pos] as f64 - ref_f).abs() - 0.5;

            {
                let mut worst = WORST.lock().unwrap();
                if error > *worst {
                    *worst = error;
                }
            }

            if error > 0.06 {
                panic!(
                    "[{}:{}] expected: {}, got {} ({:.9})",
                    i, j, ref_i, oil[pos], ref_f
                );
            }
        }
    }
}

fn ref_xscale(input: &[f64], in_width: usize, output: &mut [f64], out_width: usize, cmp: usize) {
    let taps = calc_taps_check(in_width as u32, out_width as u32);
    let mut coeffs = vec![0.0f64; taps];
    let max_pos = in_width as i32 - 1;

    for i in 0..out_width {
        let (smp_i, tx) = split_map_check(in_width as u32, out_width as u32, i as u32);
        let start = smp_i - (taps as i32 / 2 - 1);

        let start_safe = start.max(0);
        let ltrim = (start_safe - start) as usize;

        let mut taps_safe = taps - ltrim;
        if start_safe + taps_safe as i32 > max_pos {
            taps_safe = (max_pos - start_safe + 1) as usize;
        }
        let rtrim = ((start + taps as i32) - (start_safe + taps_safe as i32)) as usize;

        ref_calc_coeffs(&mut coeffs, tx, taps, ltrim, rtrim);

        for j in 0..cmp {
            output[i * cmp + j] = 0.0;
            for k in 0..taps_safe {
                let in_pos = (start_safe + k as i32) as usize;
                output[i * cmp + j] += coeffs[ltrim + k] * input[in_pos * cmp + j];
            }
        }
    }
}

fn ref_yscale(
    input: &[Vec<f64>],
    width: usize,
    in_height: usize,
    output: &mut [Vec<f64>],
    out_height: usize,
    cmp: usize,
) {
    let mut transposed = vec![0.0f64; in_height * cmp];
    let mut trans_scaled = vec![0.0f64; out_height * cmp];

    for i in 0..width {
        // Transpose column i
        for row in 0..in_height {
            for j in 0..cmp {
                transposed[row * cmp + j] = input[row][i * cmp + j];
            }
        }
        // Scale
        ref_xscale(&transposed, in_height, &mut trans_scaled, out_height, cmp);
        // Transpose back
        for row in 0..out_height {
            for j in 0..cmp {
                output[row][i * cmp + j] = trans_scaled[row * cmp + j];
            }
        }
    }
}

fn ref_scale(
    input: &[Vec<u8>],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
    cs: ColorSpace,
) -> Vec<Vec<f64>> {
    let cmp = cs.components();
    let stride = cmp * in_width;

    // Horizontal scaling
    let mut intermediate: Vec<Vec<f64>> = Vec::with_capacity(in_height);
    let mut pre_line = vec![0.0f64; stride];

    for row in 0..in_height {
        // Convert chars to floats
        for j in 0..stride {
            pre_line[j] = input[row][j] as f64 / 255.0;
        }

        // Preprocess each pixel
        for j in 0..in_width {
            preprocess(&mut pre_line[j * cmp..(j + 1) * cmp], cs);
        }

        // Horizontal scale
        let mut out_row = vec![0.0f64; out_width * cmp];
        ref_xscale(&pre_line, in_width, &mut out_row, out_width, cmp);
        intermediate.push(out_row);
    }

    // Vertical scaling
    let mut output: Vec<Vec<f64>> = (0..out_height)
        .map(|_| vec![0.0f64; out_width * cmp])
        .collect();
    ref_yscale(&intermediate, out_width, in_height, &mut output, out_height, cmp);

    // Postprocess
    for row in 0..out_height {
        for j in 0..out_width {
            postprocess(&mut output[row][j * cmp..(j + 1) * cmp], cs);
        }
    }

    output
}

fn do_oil_scale(
    input: &[Vec<u8>],
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    cs: ColorSpace,
) -> Vec<Vec<u8>> {
    let cmp = cs.components();
    let mut os = OilScale::new(in_height, out_height, in_width, out_width, cs).unwrap();
    let mut output: Vec<Vec<u8>> = (0..out_height)
        .map(|_| vec![0u8; out_width as usize * cmp])
        .collect();

    let mut in_line = 0usize;
    for i in 0..out_height as usize {
        while os.slots() > 0 {
            os.push_scanline(&input[in_line]);
            in_line += 1;
        }
        os.read_scanline(&mut output[i]);
    }

    output
}

fn test_scale(
    in_width: u32,
    in_height: u32,
    input: &[Vec<u8>],
    out_width: u32,
    out_height: u32,
    cs: ColorSpace,
) {
    let cmp = cs.components();

    let oil_output = do_oil_scale(input, in_width, in_height, out_width, out_height, cs);

    let ref_output = ref_scale(input, in_width as usize, in_height as usize, out_width as usize, out_height as usize, cs);

    for i in 0..out_height as usize {
        validate_scanline8(&oil_output[i], &ref_output[i], out_width as usize, cmp);
    }
}

fn test_scale_square_rand(rng: &mut StdRng, in_dim: u32, out_dim: u32, cs: ColorSpace) {
    let cmp = cs.components();
    let stride = cmp * in_dim as usize;

    let input: Vec<Vec<u8>> = (0..in_dim)
        .map(|_| {
            let mut row = vec![0u8; stride];
            for b in row.iter_mut() {
                *b = (rng.gen::<u32>() % 256) as u8;
            }
            row
        })
        .collect();

    test_scale(in_dim, in_dim, &input, out_dim, out_dim, cs);
}

fn test_scale_each_cs(rng: &mut StdRng, dim_a: u32, dim_b: u32) {
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::G);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::GA);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RGB);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RGBA);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::RGBX);
    test_scale_square_rand(rng, dim_a, dim_b, ColorSpace::CMYK);
}

fn test_scale_all_permutations(rng: &mut StdRng, dim_a: u32, dim_b: u32) {
    test_scale_each_cs(rng, dim_a, dim_b);
    test_scale_each_cs(rng, dim_b, dim_a);
}

fn test_scale_catrom_extremes(cs: ColorSpace) {
    let cmp = cs.components();
    let mut input: Vec<Vec<u8>> = vec![vec![0u8; 4 * cmp]; 4];

    // Solid white center with black border, replicated across components
    for j in 0..cmp {
        input[1][1 * cmp + j] = 255;
        input[1][2 * cmp + j] = 255;
        input[2][1 * cmp + j] = 255;
        input[2][2 * cmp + j] = 255;
    }

    test_scale(4, 4, &input, 7, 7, cs);
}

#[test]
fn scale_5_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289551);
    test_scale_all_permutations(&mut rng, 5, 1);
}

#[test]
fn scale_8_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289552);
    test_scale_all_permutations(&mut rng, 8, 1);
}

#[test]
fn scale_8_to_3() {
    let mut rng = StdRng::seed_from_u64(1531289553);
    test_scale_all_permutations(&mut rng, 8, 3);
}

#[test]
fn scale_100_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289554);
    test_scale_all_permutations(&mut rng, 100, 1);
}

#[test]
fn scale_100_to_99() {
    let mut rng = StdRng::seed_from_u64(1531289555);
    test_scale_all_permutations(&mut rng, 100, 99);
}

#[test]
fn scale_2_to_1() {
    let mut rng = StdRng::seed_from_u64(1531289556);
    test_scale_all_permutations(&mut rng, 2, 1);
}

#[test]
fn scale_catrom_extremes() {
    test_scale_catrom_extremes(ColorSpace::G);
    test_scale_catrom_extremes(ColorSpace::GA);
    test_scale_catrom_extremes(ColorSpace::RGB);
    test_scale_catrom_extremes(ColorSpace::RGBA);
    test_scale_catrom_extremes(ColorSpace::RGBX);
    test_scale_catrom_extremes(ColorSpace::CMYK);
}
