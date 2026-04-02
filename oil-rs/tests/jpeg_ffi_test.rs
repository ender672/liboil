#![cfg(feature = "ffi")]

use oil::jpeg_ffi;

/// Encode a minimal grayscale JPEG from raw pixel data.
fn make_grayscale_jpeg(width: u16, height: u16, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, 95);
    encoder
        .encode(pixels, width, height, jpeg_encoder::ColorType::Luma)
        .expect("encode grayscale JPEG");
    buf
}

/// Encode a minimal CMYK JPEG from raw pixel data.
fn make_cmyk_jpeg(width: u16, height: u16, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, 95);
    encoder
        .encode(pixels, width, height, jpeg_encoder::ColorType::Cmyk)
        .expect("encode CMYK JPEG");
    buf
}

#[test]
fn ffi_resize_downscale() {
    let input = std::fs::read("../dalai.jpg").expect("read test JPEG");
    let output = jpeg_ffi::resize_jpeg(&input, 100, 100, 94).expect("ffi resize");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let pixels = decoder.decode().expect("decode FFI output");
    let info = decoder.info().unwrap();

    // dalai.jpg is 258x222, so 100x100 should produce valid output
    assert!(info.width > 0 && info.width <= 100);
    assert!(info.height > 0 && info.height <= 100);
    assert_eq!(pixels.len(), info.width as usize * info.height as usize * 3);
}

#[test]
fn ffi_resize_upscale() {
    let input = std::fs::read("../dalai.jpg").expect("read test JPEG");
    let output = jpeg_ffi::resize_jpeg(&input, 500, 500, 94).expect("ffi resize");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let pixels = decoder.decode().expect("decode FFI output");
    let info = decoder.info().unwrap();

    assert!(info.width > 0 && info.width <= 500);
    assert!(info.height > 0 && info.height <= 500);
    assert_eq!(pixels.len(), info.width as usize * info.height as usize * 3);
}

#[test]
fn ffi_resize_earth() {
    let input = std::fs::read("../earth.jpg").expect("read test JPEG");
    let output = jpeg_ffi::resize_jpeg(&input, 300, 150, 85).expect("ffi resize");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let pixels = decoder.decode().expect("decode FFI output");
    let info = decoder.info().unwrap();

    assert!(info.width > 0 && info.width <= 300);
    assert!(info.height > 0 && info.height <= 150);
    assert_eq!(pixels.len(), info.width as usize * info.height as usize * 3);
}

#[test]
fn ffi_resize_grayscale() {
    let (w, h) = (8u16, 8u16);
    let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let input = make_grayscale_jpeg(w, h, &pixels);

    let output = jpeg_ffi::resize_jpeg(&input, 4, 4, 90).expect("ffi resize grayscale");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let out_pixels = decoder.decode().expect("decode output");
    let info = decoder.info().unwrap();

    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::L8);
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);
    assert_eq!(out_pixels.len(), 4 * 4);
}

#[test]
fn ffi_resize_cmyk() {
    let (w, h) = (8u16, 8u16);
    let pixels: Vec<u8> = (0..8 * 8 * 4).map(|i| (i % 256) as u8).collect();
    let input = make_cmyk_jpeg(w, h, &pixels);

    let output = jpeg_ffi::resize_jpeg(&input, 4, 4, 90).expect("ffi resize CMYK");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let out_pixels = decoder.decode().expect("decode output");
    let info = decoder.info().unwrap();

    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::CMYK32);
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);
    assert_eq!(out_pixels.len(), 4 * 4 * 4);
}
