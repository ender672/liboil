use oil::png::{resize_png, png_dimensions};

/// Create a minimal PNG in memory with the given dimensions, color type, and pixel data.
fn make_png(width: u32, height: u32, color_type: png::ColorType, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, width, height);
        encoder.set_color(color_type);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(pixels).unwrap();
    }
    buf
}

fn make_rgb_png(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    make_png(width, height, png::ColorType::Rgb, pixels)
}

fn make_rgba_png(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    make_png(width, height, png::ColorType::Rgba, pixels)
}

#[test]
fn png_resize_downscale() {
    let w = 100u32;
    let h = 80u32;
    let pixels = vec![128u8; (w * h * 3) as usize];
    let input = make_rgb_png(w, h, &pixels);

    let output = resize_png(&input, 50, 40).expect("resize down");

    let (ow, oh) = png_dimensions(&output).expect("read output dims");
    assert_eq!(ow, 50);
    assert_eq!(oh, 40);
}

#[test]
fn png_resize_upscale() {
    let w = 20u32;
    let h = 16u32;
    let pixels = vec![200u8; (w * h * 3) as usize];
    let input = make_rgb_png(w, h, &pixels);

    let output = resize_png(&input, 60, 48).expect("resize up");

    let (ow, oh) = png_dimensions(&output).expect("read output dims");
    assert_eq!(ow, 60);
    assert_eq!(oh, 48);
}

#[test]
fn png_roundtrip_identity() {
    // 4x4 solid red image — resize to same size should produce valid output
    let w = 4u32;
    let h = 4u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for chunk in pixels.chunks_exact_mut(3) {
        chunk[0] = 255; // R
        chunk[1] = 0;   // G
        chunk[2] = 0;   // B
    }
    let input = make_rgb_png(w, h, &pixels);

    let output = resize_png(&input, 4, 4).expect("identity resize");

    // Decode and verify pixels are still reddish
    let decoder = png::Decoder::new(&output[..]);
    let mut reader = decoder.read_info().unwrap();
    let mut out_pixels = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut out_pixels).unwrap();

    for chunk in out_pixels.chunks_exact(3) {
        assert!(chunk[0] > 200, "red channel should be high: {}", chunk[0]);
        assert!(chunk[1] < 30, "green channel should be low: {}", chunk[1]);
        assert!(chunk[2] < 30, "blue channel should be low: {}", chunk[2]);
    }
}

#[test]
fn png_dimensions_read() {
    let input = make_rgb_png(123, 456, &vec![0u8; 123 * 456 * 3]);
    let (w, h) = png_dimensions(&input).expect("dimensions");
    assert_eq!(w, 123);
    assert_eq!(h, 456);
}

#[test]
fn png_rgba_resize_downscale() {
    let w = 100u32;
    let h = 80u32;
    let pixels = vec![128u8; (w * h * 4) as usize];
    let input = make_rgba_png(w, h, &pixels);

    let output = resize_png(&input, 50, 40).expect("resize down");

    let (ow, oh) = png_dimensions(&output).expect("read output dims");
    assert_eq!(ow, 50);
    assert_eq!(oh, 40);

    // Verify output is RGBA
    let decoder = png::Decoder::new(&output[..]);
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().color_type, png::ColorType::Rgba);
}

#[test]
fn png_rgba_resize_upscale() {
    let w = 20u32;
    let h = 16u32;
    let pixels = vec![200u8; (w * h * 4) as usize];
    let input = make_rgba_png(w, h, &pixels);

    let output = resize_png(&input, 60, 48).expect("resize up");

    let (ow, oh) = png_dimensions(&output).expect("read output dims");
    assert_eq!(ow, 60);
    assert_eq!(oh, 48);

    let decoder = png::Decoder::new(&output[..]);
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().color_type, png::ColorType::Rgba);
}

#[test]
fn png_rgba_roundtrip_identity() {
    let w = 4u32;
    let h = 4u32;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for chunk in pixels.chunks_exact_mut(4) {
        chunk[0] = 255; // R
        chunk[1] = 0;   // G
        chunk[2] = 0;   // B
        chunk[3] = 255; // A (fully opaque)
    }
    let input = make_rgba_png(w, h, &pixels);

    let output = resize_png(&input, 4, 4).expect("identity resize");

    let decoder = png::Decoder::new(&output[..]);
    let mut reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().color_type, png::ColorType::Rgba);
    let mut out_pixels = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut out_pixels).unwrap();

    for chunk in out_pixels.chunks_exact(4) {
        assert!(chunk[0] > 200, "red channel should be high: {}", chunk[0]);
        assert!(chunk[1] < 30, "green channel should be low: {}", chunk[1]);
        assert!(chunk[2] < 30, "blue channel should be low: {}", chunk[2]);
        assert!(chunk[3] > 200, "alpha channel should be high: {}", chunk[3]);
    }
}
