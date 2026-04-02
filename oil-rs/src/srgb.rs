use std::sync::OnceLock;

const L2S_ALL_LEN: usize = 32768;

pub struct SrgbTables {
    /// sRGB byte -> linear float (gamma decompression)
    pub s2l: [f32; 256],
    /// byte -> float identity mapping (no gamma, for greyscale/CMYK)
    pub i2f: [f32; 256],
    /// linear-to-sRGB full table (includes padding for out-of-range values)
    l2s_all: [u8; L2S_ALL_LEN],
    /// Offset into l2s_all where the usable mapping begins
    l2s_offset: usize,
    /// Length of the usable mapping region
    pub l2s_len: usize,
}

impl SrgbTables {
    fn build() -> Self {
        let mut tables = SrgbTables {
            s2l: [0.0; 256],
            i2f: [0.0; 256],
            l2s_all: [0; L2S_ALL_LEN],
            l2s_offset: 0,
            l2s_len: 0,
        };

        // build s2l: sRGB byte -> linear float
        for input in 0..=255u16 {
            let in_f = input as f64 / 255.0;
            let val = if in_f <= 0.040448236277 {
                in_f / 12.92
            } else {
                ((in_f + 0.055) / 1.055).powf(2.4)
            };
            tables.s2l[input as usize] = val as f32;
        }

        // build i2f: identity byte -> float
        for i in 0..=255u16 {
            tables.i2f[i as usize] = i as f32 / 255.0;
        }

        // build l2s: linear float -> sRGB byte
        let padding = L2S_ALL_LEN * 17 / 98;
        tables.l2s_len = L2S_ALL_LEN - 2 * padding;
        tables.l2s_offset = padding;

        for i in 0..tables.l2s_len {
            let srgb_f = (i as f64 + 0.5) / (tables.l2s_len - 1) as f64;
            let val = if srgb_f <= 0.00313 {
                srgb_f * 12.92
            } else {
                1.055 * srgb_f.powf(1.0 / 2.4) - 0.055
            };
            tables.l2s_all[padding + i] = (val * 255.0).round() as u8;
        }

        // Padding above: clamp to 255
        for i in 0..padding {
            tables.l2s_all[padding + tables.l2s_len + i] = 255;
        }

        tables
    }

    /// Map a linear RGB float to an sRGB byte using the lookup table.
    #[inline]
    pub fn linear_to_srgb(&self, val: f32) -> u8 {
        let idx = (val * (self.l2s_len - 1) as f32) as i32;
        self.l2s_all[(self.l2s_offset as i32 + idx) as usize]
    }

    /// Return a pointer into l2s_all at the l2s_offset position.
    /// This mirrors the C `l2s_map` pointer: indices can be negative (for
    /// Catmull-Rom overshoot) and positive up to l2s_len, relying on padding
    /// in both directions.
    #[inline]
    pub fn l2s_ptr(&self) -> *const u8 {
        unsafe { self.l2s_all.as_ptr().add(self.l2s_offset) }
    }
}

static TABLES: OnceLock<SrgbTables> = OnceLock::new();

pub fn tables() -> &'static SrgbTables {
    TABLES.get_or_init(SrgbTables::build)
}
