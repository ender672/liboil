use crate::colorspace::ColorSpace;
use crate::kernel;
use crate::srgb;
#[cfg(target_arch = "x86_64")]
use crate::sse2;

const MAX_DIMENSION: u32 = 1_000_000;
const TAPS: usize = 4;

#[derive(Debug)]
pub enum OilError {
    InvalidArgument,
    AllocationFailed,
}

impl std::fmt::Display for OilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OilError::InvalidArgument => write!(f, "invalid argument"),
            OilError::AllocationFailed => write!(f, "allocation failed"),
        }
    }
}

impl std::error::Error for OilError {}

pub struct OilScale {
    in_height: u32,
    out_height: u32,
    in_width: u32,
    out_width: u32,
    cs: ColorSpace,
    in_pos: u32,
    out_pos: u32,
    coeffs_y: Vec<f32>,
    coeffs_x: Vec<f32>,
    borders_x: Vec<i32>,
    borders_y: Vec<i32>,
    sums_y: Vec<f32>,
    rb: Vec<f32>,
    tmp_coeffs: Vec<f32>,
    is_upscale: bool,
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn clampf(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn f2i(x: f32) -> u8 {
    (x + 0.5) as u8
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn add_sample_to_sum(sample: f32, coeffs: &[f32], sum: &mut [f32]) {
    for i in 0..4 {
        sum[i] += sample * coeffs[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn push_f(f: &mut [f32; 4], val: f32) {
    f[0] = f[1];
    f[1] = f[2];
    f[2] = f[3];
    f[3] = val;
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn shift_left(f: &mut [f32]) {
    f[0] = f[1];
    f[1] = f[2];
    f[2] = f[3];
    f[3] = 0.0;
}

#[cfg(not(target_arch = "x86_64"))]
fn xscale_up_rgb(
    input: &[u8],
    width_in: u32,
    out: &mut [f32],
    coeff_buf: &[f32],
    border_buf: &[i32],
) {
    let tables = srgb::tables();
    let mut smp = [[0.0f32; 4]; 3];
    let mut out_idx = 0usize;
    let mut coeff_idx = 0usize;

    for i in 0..width_in as usize {
        let in_base = i * 3;
        for j in 0..3 {
            push_f(&mut smp[j], tables.s2l[input[in_base + j] as usize]);
        }
        for _ in 0..border_buf[i] {
            let coeffs = &coeff_buf[coeff_idx..coeff_idx + 4];
            for j in 0..3 {
                out[out_idx + j] = smp[j][0] * coeffs[0]
                    + smp[j][1] * coeffs[1]
                    + smp[j][2] * coeffs[2]
                    + smp[j][3] * coeffs[3];
            }
            out_idx += 3;
            coeff_idx += 4;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn yscale_up_rgb(
    lines: [&[f32]; 4],
    len: usize,
    coeffs: &[f32],
    out: &mut [u8],
) {
    let tables = srgb::tables();
    for i in 0..len {
        let sum = coeffs[0] * lines[0][i]
            + coeffs[1] * lines[1][i]
            + coeffs[2] * lines[2][i]
            + coeffs[3] * lines[3][i];
        out[i] = tables.linear_to_srgb(sum);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn scale_down_rgb(
    input: &[u8],
    sums_y: &mut [f32],
    out_width: u32,
    coeffs_x: &[f32],
    border_buf: &[i32],
    coeffs_y: &[f32],
) {
    let tables = srgb::tables();
    let mut sum = [[0.0f32; 4]; 3];
    let mut in_idx = 0usize;
    let mut cx_idx = 0usize;
    let mut sy_idx = 0usize;

    for i in 0..out_width as usize {
        for _ in 0..border_buf[i] {
            let cx = &coeffs_x[cx_idx..cx_idx + 4];
            for k in 0..3 {
                add_sample_to_sum(tables.s2l[input[in_idx + k] as usize], cx, &mut sum[k]);
            }
            in_idx += 3;
            cx_idx += 4;
        }

        for j in 0..3 {
            add_sample_to_sum(sum[j][0], coeffs_y, &mut sums_y[sy_idx..sy_idx + 4]);
            shift_left(&mut sum[j]);
            sy_idx += 4;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn yscale_out_rgb(sums: &mut [f32], sl_len: usize, out: &mut [u8]) {
    let tables = srgb::tables();
    let mut s_idx = 0usize;
    for i in 0..sl_len {
        out[i] = tables.linear_to_srgb(sums[s_idx]);
        shift_left(&mut sums[s_idx..s_idx + 4]);
        s_idx += 4;
    }
}

impl OilScale {
    pub fn new(
        in_height: u32,
        out_height: u32,
        in_width: u32,
        out_width: u32,
        cs: ColorSpace,
    ) -> Result<Self, OilError> {
        if in_height < 1
            || out_height < 1
            || in_width < 1
            || out_width < 1
            || in_height > MAX_DIMENSION
            || out_height > MAX_DIMENSION
            || in_width > MAX_DIMENSION
            || out_width > MAX_DIMENSION
        {
            return Err(OilError::InvalidArgument);
        }

        // Only allow upscaling if both dimensions are being upscaled
        if (out_height > in_height) != (out_width > in_width) {
            return Err(OilError::InvalidArgument);
        }

        // Ensure tables are initialized
        srgb::tables();

        let is_upscale = out_width > in_width;

        let mut os = OilScale {
            in_height,
            out_height,
            in_width,
            out_width,
            cs,
            in_pos: 0,
            out_pos: 0,
            coeffs_y: Vec::new(),
            coeffs_x: Vec::new(),
            borders_x: Vec::new(),
            borders_y: Vec::new(),
            sums_y: Vec::new(),
            rb: Vec::new(),
            tmp_coeffs: Vec::new(),
            is_upscale,
        };

        if is_upscale {
            os.upscale_init();
        } else {
            os.downscale_init();
        }

        Ok(os)
    }

    fn upscale_init(&mut self) {
        let cmp = self.cs.components();
        let coeffs_x_len = TAPS * self.out_width.max(self.in_width) as usize;
        let borders_x_len = self.in_width as usize;
        let coeffs_y_len = TAPS * self.out_height.max(self.in_height) as usize;
        let borders_y_len = self.in_height as usize;
        let rb_len = self.out_width as usize * cmp * TAPS;

        self.coeffs_x = vec![0.0; coeffs_x_len];
        self.borders_x = vec![0; borders_x_len];
        self.coeffs_y = vec![0.0; coeffs_y_len];
        self.borders_y = vec![0; borders_y_len];
        self.rb = vec![0.0; rb_len];

        kernel::scale_up_coeffs(
            self.in_width,
            self.out_width,
            &mut self.coeffs_x,
            &mut self.borders_x,
        );
        kernel::scale_up_coeffs(
            self.in_height,
            self.out_height,
            &mut self.coeffs_y,
            &mut self.borders_y,
        );
    }

    fn downscale_init(&mut self) {
        let cmp = self.cs.components();
        let taps_x = kernel::calc_taps(self.in_width, self.out_width);
        let taps_y = kernel::calc_taps(self.in_height, self.out_height);

        let coeffs_x_len = TAPS * self.in_width.max(self.out_width) as usize;
        let borders_x_len = self.out_width as usize;
        let coeffs_y_len = TAPS * self.in_height.max(self.out_height) as usize;
        let borders_y_len = self.out_height as usize;
        let sums_len = self.out_width as usize * cmp * TAPS;
        let tmp_len = taps_x.max(taps_y);

        self.coeffs_x = vec![0.0; coeffs_x_len];
        self.borders_x = vec![0; borders_x_len];
        self.coeffs_y = vec![0.0; coeffs_y_len];
        self.borders_y = vec![0; borders_y_len];
        self.sums_y = vec![0.0; sums_len];
        self.tmp_coeffs = vec![0.0; tmp_len];

        kernel::scale_down_coeffs(
            self.in_width,
            self.out_width,
            &mut self.coeffs_x,
            &mut self.borders_x,
            &mut self.tmp_coeffs,
        );
        kernel::scale_down_coeffs(
            self.in_height,
            self.out_height,
            &mut self.coeffs_y,
            &mut self.borders_y,
            &mut self.tmp_coeffs,
        );
    }

    /// Return the number of input scanlines needed before the next output
    /// scanline can be produced.
    pub fn slots(&self) -> usize {
        if !self.is_upscale {
            self.borders_y[self.out_pos as usize] as usize
        } else if self.in_pos > 0 {
            if self.borders_y[self.in_pos as usize - 1] == 0 {
                1
            } else {
                0
            }
        } else if self.borders_y[0] == 0 {
            2
        } else {
            1
        }
    }

    /// Ingest one input scanline.
    pub fn push_scanline(&mut self, input: &[u8]) {
        if self.is_upscale {
            self.up_scale_in(input);
        } else {
            self.down_scale_in(input);
        }
    }

    /// Produce the next scaled output scanline.
    pub fn read_scanline(&mut self, output: &mut [u8]) {
        if self.is_upscale {
            self.up_scale_out(output);
        } else {
            self.down_scale_out(output);
        }
        self.out_pos += 1;
    }

    /// Reset row counters so the scaler can be reused.
    pub fn reset(&mut self) {
        self.in_pos = 0;
        self.out_pos = 0;
    }

    fn get_rb_line(&self, line: u32) -> usize {
        let sl_len = self.cs.components() * self.out_width as usize;
        line as usize * sl_len
    }

    fn up_scale_in(&mut self, input: &[u8]) {
        let rb_offset = self.get_rb_line(self.in_pos % 4);
        let sl_len = self.cs.components() * self.out_width as usize;

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    sse2::xscale_up_rgb(
                        input,
                        self.in_width,
                        &mut self.rb[rb_offset..rb_offset + sl_len],
                        &self.coeffs_x,
                        &self.borders_x,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                xscale_up_rgb(
                    input,
                    self.in_width,
                    &mut self.rb[rb_offset..rb_offset + sl_len],
                    &self.coeffs_x,
                    &self.borders_x,
                );
            }
            _ => unimplemented!("colorspace {:?} not yet supported", self.cs),
        }

        self.in_pos += 1;
    }

    fn up_scale_out(&mut self, output: &mut [u8]) {
        let cmp = self.cs.components();
        let sl_len = cmp * self.out_width as usize;

        let offsets: [usize; 4] = [
            self.get_rb_line((self.in_pos + 0) % 4),
            self.get_rb_line((self.in_pos + 1) % 4),
            self.get_rb_line((self.in_pos + 2) % 4),
            self.get_rb_line((self.in_pos + 3) % 4),
        ];

        let lines: [&[f32]; 4] = [
            &self.rb[offsets[0]..offsets[0] + sl_len],
            &self.rb[offsets[1]..offsets[1] + sl_len],
            &self.rb[offsets[2]..offsets[2] + sl_len],
            &self.rb[offsets[3]..offsets[3] + sl_len],
        ];

        let coeff_start = self.out_pos as usize * 4;
        let coeffs = &self.coeffs_y[coeff_start..coeff_start + 4];

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    sse2::yscale_up_rgb(lines, sl_len, coeffs, output);
                }
                #[cfg(not(target_arch = "x86_64"))]
                yscale_up_rgb(lines, sl_len, coeffs, output);
            }
            _ => unimplemented!("colorspace {:?} not yet supported", self.cs),
        }

        self.borders_y[self.in_pos as usize - 1] -= 1;
    }

    fn down_scale_in(&mut self, input: &[u8]) {
        let coeffs_y_start = self.in_pos as usize * 4;
        let coeffs_y = [
            self.coeffs_y[coeffs_y_start],
            self.coeffs_y[coeffs_y_start + 1],
            self.coeffs_y[coeffs_y_start + 2],
            self.coeffs_y[coeffs_y_start + 3],
        ];

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    sse2::scale_down_rgb(
                        input,
                        &mut self.sums_y,
                        self.out_width,
                        &self.coeffs_x,
                        &self.borders_x,
                        &coeffs_y,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                scale_down_rgb(
                    input,
                    &mut self.sums_y,
                    self.out_width,
                    &self.coeffs_x,
                    &self.borders_x,
                    &coeffs_y,
                );
            }
            _ => unimplemented!("colorspace {:?} not yet supported", self.cs),
        }

        self.borders_y[self.out_pos as usize] -= 1;
        self.in_pos += 1;
    }

    fn down_scale_out(&mut self, output: &mut [u8]) {
        let cmp = self.cs.components();
        let sl_len = self.out_width as usize * cmp;

        match self.cs {
            ColorSpace::RGB => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    sse2::yscale_out_rgb(&mut self.sums_y, sl_len, output);
                }
                #[cfg(not(target_arch = "x86_64"))]
                yscale_out_rgb(&mut self.sums_y, sl_len, output);
            }
            _ => unimplemented!("colorspace {:?} not yet supported", self.cs),
        }
    }
}
