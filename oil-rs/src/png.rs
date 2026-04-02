use std::path::Path;

use crate::colorspace::ColorSpace;
use crate::scale::{OilError, OilScale};

/// Decode an RGB PNG from `input`, resize to `out_width` x `out_height`,
/// and re-encode as PNG, returning the result as bytes.
pub fn resize_png(
    input: &[u8],
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>, OilError> {
    let mut decoder = png::Decoder::new(input);
    // Match C implementation: expand palette to RGB, sub-8-bit gray to 8-bit,
    // strip 16-bit to 8-bit, and handle tRNS as alpha.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|_| OilError::InvalidArgument)?;
    let info = reader.info();
    let in_width = info.width;
    let in_height = info.height;

    let (color_type, _) = reader.output_color_type();
    let cs = match color_type {
        png::ColorType::Grayscale => ColorSpace::G,
        png::ColorType::GrayscaleAlpha => ColorSpace::GA,
        png::ColorType::Rgb => ColorSpace::RGB,
        png::ColorType::Rgba => ColorSpace::RGBA,
        _ => return Err(OilError::InvalidArgument),
    };
    let cmp = cs.components();

    let mut scaler = OilScale::new(in_height, out_height, in_width, out_width, cs)?;

    let in_stride = in_width as usize * cmp;
    let out_stride = out_width as usize * cmp;

    // For interlaced PNGs, we must decode the entire image up front.
    // For non-interlaced, we can stream scanline by scanline.
    let interlaced = info.interlaced;
    let mut output = vec![0u8; out_height as usize * out_stride];

    if interlaced {
        // Read entire image into memory
        let mut full_buf = vec![0u8; reader.output_buffer_size()];
        reader.next_frame(&mut full_buf).map_err(|_| OilError::InvalidArgument)?;

        let mut in_line = 0usize;
        for i in 0..out_height as usize {
            while scaler.slots() > 0 {
                let row_start = in_line * in_stride;
                scaler.push_scanline(&full_buf[row_start..row_start + in_stride]);
                in_line += 1;
            }
            scaler.read_scanline(&mut output[i * out_stride..(i + 1) * out_stride]);
        }
    } else {
        // Stream one row at a time
        let mut row_buf = vec![0u8; in_stride];
        for i in 0..out_height as usize {
            while scaler.slots() > 0 {
                let row = reader.next_row()
                    .map_err(|_| OilError::InvalidArgument)?
                    .ok_or(OilError::InvalidArgument)?;
                row_buf.copy_from_slice(row.data());
                scaler.push_scanline(&row_buf);
            }
            scaler.read_scanline(&mut output[i * out_stride..(i + 1) * out_stride]);
        }
    };

    // Encode the output as PNG
    let mut result = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut result, out_width, out_height);
        let out_color = match cs {
            ColorSpace::G => png::ColorType::Grayscale,
            ColorSpace::GA => png::ColorType::GrayscaleAlpha,
            ColorSpace::RGBA => png::ColorType::Rgba,
            _ => png::ColorType::Rgb,
        };
        encoder.set_color(out_color);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|_| OilError::AllocationFailed)?;
        writer.write_image_data(&output).map_err(|_| OilError::AllocationFailed)?;
    }

    Ok(result)
}

/// Resize a PNG file on disk. Reads from `input_path`, writes to `output_path`.
pub fn resize_png_file(
    input_path: &Path,
    out_width: u32,
    out_height: u32,
    output_path: &Path,
) -> Result<(), OilError> {
    let input_data = std::fs::read(input_path).map_err(|_| OilError::InvalidArgument)?;
    let encoded = resize_png(&input_data, out_width, out_height)?;
    std::fs::write(output_path, &encoded).map_err(|_| OilError::AllocationFailed)?;
    Ok(())
}

/// Read the dimensions of a PNG file without decoding pixel data.
pub fn png_dimensions(input: &[u8]) -> Result<(u32, u32), OilError> {
    let decoder = png::Decoder::new(input);
    let reader = decoder.read_info().map_err(|_| OilError::InvalidArgument)?;
    let info = reader.info();
    Ok((info.width, info.height))
}
