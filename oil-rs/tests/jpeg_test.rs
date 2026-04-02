use oil::jpeg;

/// Encode a minimal grayscale JPEG from raw pixel data.
fn make_grayscale_jpeg(width: u16, height: u16, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, 95);
    encoder
        .encode(pixels, width, height, jpeg_encoder::ColorType::Luma)
        .expect("encode grayscale JPEG");
    buf
}

/// Encode a minimal RGB JPEG from raw pixel data.
fn make_rgb_jpeg(width: u16, height: u16, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    let encoder = jpeg_encoder::Encoder::new(&mut buf, 95);
    encoder
        .encode(pixels, width, height, jpeg_encoder::ColorType::Rgb)
        .expect("encode RGB JPEG");
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
fn resize_grayscale() {
    let (w, h) = (8u16, 8u16);
    let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let input = make_grayscale_jpeg(w, h, &pixels);

    let output = jpeg::resize_jpeg(&input, 4, 4, 90).expect("resize grayscale");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let out_pixels = decoder.decode().expect("decode output");
    let info = decoder.info().unwrap();

    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::L8);
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);
    assert_eq!(out_pixels.len(), 4 * 4 * 1);
}

#[test]
fn resize_rgb() {
    let (w, h) = (8u16, 8u16);
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
    let input = make_rgb_jpeg(w, h, &pixels);

    let output = jpeg::resize_jpeg(&input, 4, 4, 90).expect("resize RGB");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let out_pixels = decoder.decode().expect("decode output");
    let info = decoder.info().unwrap();

    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);
    assert_eq!(out_pixels.len(), 4 * 4 * 3);
}

#[test]
fn resize_cmyk() {
    let (w, h) = (8u16, 8u16);
    let pixels: Vec<u8> = (0..8 * 8 * 4).map(|i| (i % 256) as u8).collect();
    let input = make_cmyk_jpeg(w, h, &pixels);

    let output = jpeg::resize_jpeg(&input, 4, 4, 90).expect("resize CMYK");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let out_pixels = decoder.decode().expect("decode output");
    let info = decoder.info().unwrap();

    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::CMYK32);
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 4);
    assert_eq!(out_pixels.len(), 4 * 4 * 4);
}

#[test]
fn resize_grayscale_upscale() {
    let (w, h) = (4u16, 4u16);
    let pixels: Vec<u8> = vec![128; 16];
    let input = make_grayscale_jpeg(w, h, &pixels);

    let output = jpeg::resize_jpeg(&input, 8, 8, 90).expect("upscale grayscale");

    let mut decoder = jpeg_decoder::Decoder::new(&output[..]);
    let out_pixels = decoder.decode().expect("decode output");
    let info = decoder.info().unwrap();

    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::L8);
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert_eq!(out_pixels.len(), 8 * 8);
}
