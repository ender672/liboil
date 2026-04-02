use crate::colorspace::ColorSpace;
use crate::scale::{OilError, OilScale};

/// Decode a JPEG from bytes, resize it, and re-encode as JPEG.
pub fn resize_jpeg(
    input: &[u8],
    out_width: u32,
    out_height: u32,
    quality: u8,
) -> Result<Vec<u8>, OilError> {
    let mut decoder = jpeg_decoder::Decoder::new(input);
    let pixels = decoder.decode().map_err(|_| OilError::InvalidArgument)?;
    let info = decoder.info().ok_or(OilError::InvalidArgument)?;

    let in_width = info.width as u32;
    let in_height = info.height as u32;

    let cs = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => ColorSpace::G,
        jpeg_decoder::PixelFormat::RGB24 => ColorSpace::RGB,
        jpeg_decoder::PixelFormat::CMYK32 => ColorSpace::CMYK,
        _ => return Err(OilError::InvalidArgument),
    };
    let cmp = cs.components();

    let mut scaler = OilScale::new(in_height, out_height, in_width, out_width, cs)?;

    let in_stride = in_width as usize * cmp;
    let out_stride = out_width as usize * cmp;
    let mut output = vec![0u8; out_height as usize * out_stride];

    let mut in_line = 0usize;
    for i in 0..out_height as usize {
        while scaler.slots() > 0 {
            let row_start = in_line * in_stride;
            scaler.push_scanline(&pixels[row_start..row_start + in_stride]);
            in_line += 1;
        }
        scaler.read_scanline(&mut output[i * out_stride..(i + 1) * out_stride]);
    }

    let enc_color_type = match cs {
        ColorSpace::G => jpeg_encoder::ColorType::Luma,
        ColorSpace::RGB => jpeg_encoder::ColorType::Rgb,
        ColorSpace::CMYK => jpeg_encoder::ColorType::Cmyk,
        _ => return Err(OilError::InvalidArgument),
    };

    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, quality);
    encoder
        .encode(&output, out_width as u16, out_height as u16, enc_color_type)
        .map_err(|_| OilError::AllocationFailed)?;

    Ok(buf)
}

/// Adjust output dimensions to maintain the input aspect ratio, fitting within
/// the given bounding box.
pub fn fix_ratio(
    src_width: u32,
    src_height: u32,
    out_width: &mut u32,
    out_height: &mut u32,
) -> Result<(), OilError> {
    if src_width < 1 || src_height < 1 || *out_width < 1 || *out_height < 1 {
        return Err(OilError::InvalidArgument);
    }

    let width_ratio = *out_width as f64 / src_width as f64;
    let height_ratio = *out_height as f64 / src_height as f64;

    if width_ratio < height_ratio {
        let tmp = (width_ratio * src_height as f64).round();
        *out_height = if tmp < 1.0 { 1 } else { tmp as u32 };
    } else {
        let tmp = (height_ratio * src_width as f64).round();
        *out_width = if tmp < 1.0 { 1 } else { tmp as u32 };
    }

    Ok(())
}
