#![cfg(feature = "ffi")]

use oil::jpeg_ffi;

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
