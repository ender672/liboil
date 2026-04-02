use std::env;
use std::fs;
use std::process;
use std::time::Instant;

use oil::colorspace::ColorSpace;
use oil::jpeg::fix_ratio;
use oil::scale::OilScale;

struct BenchImage {
	pixels: Vec<u8>,
	width: u32,
	height: u32,
}

fn load_rgba_png(path: &str) -> BenchImage {
	let data = fs::read(path).unwrap_or_else(|e| {
		eprintln!("Unable to open file: {}", e);
		process::exit(1);
	});
	let decoder = png::Decoder::new(&data[..]);
	let mut reader = decoder.read_info().unwrap_or_else(|e| {
		eprintln!("PNG decode error: {}", e);
		process::exit(1);
	});
	let info = reader.info();
	if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
		eprintln!("Input PNG must be 8-bit RGBA.");
		process::exit(1);
	}
	let width = info.width;
	let height = info.height;
	let mut pixels = vec![0u8; reader.output_buffer_size()];
	reader.next_frame(&mut pixels).unwrap_or_else(|e| {
		eprintln!("PNG frame decode error: {}", e);
		process::exit(1);
	});
	BenchImage {
		pixels,
		width,
		height,
	}
}

/// Convert RGBA pixel buffer to RGB by stripping alpha.
fn rgba_to_rgb(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
	let num_pixels = width as usize * height as usize;
	let mut rgb = Vec::with_capacity(num_pixels * 3);
	for i in 0..num_pixels {
		rgb.push(rgba[i * 4]);
		rgb.push(rgba[i * 4 + 1]);
		rgb.push(rgba[i * 4 + 2]);
	}
	rgb
}

/// Convert RGBA pixel buffer to grayscale using RGB-to-gray conversion.
/// Uses the same weighting as libpng's png_set_rgb_to_gray (ITU-R BT.709).
fn rgba_to_g(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
	let num_pixels = width as usize * height as usize;
	let mut gray = Vec::with_capacity(num_pixels);
	for i in 0..num_pixels {
		let r = rgba[i * 4] as f32;
		let g = rgba[i * 4 + 1] as f32;
		let b = rgba[i * 4 + 2] as f32;
		gray.push((0.2126 * r + 0.7152 * g + 0.0722 * b + 0.5) as u8);
	}
	gray
}

fn resize(pixels: &[u8], width: u32, height: u32, cs: ColorSpace, out_width: u32, out_height: u32) -> f64 {
	let cmp = cs.components();
	let in_stride = width as usize * cmp;

	let mut outbuf = vec![0u8; out_width as usize * cmp];

	let start = Instant::now();
	let mut scaler = OilScale::new(height, out_height, width, out_width, cs).unwrap();
	let mut in_line = 0usize;
	for _ in 0..out_height {
		while scaler.slots() > 0 {
			let row_start = in_line * in_stride;
			scaler.push_scanline(&pixels[row_start..row_start + in_stride]);
			in_line += 1;
		}
		scaler.read_scanline(&mut outbuf);
	}
	let elapsed = start.elapsed();

	elapsed.as_secs_f64() * 1000.0
}

fn do_bench(pixels: &[u8], width: u32, height: u32, cs: ColorSpace, ratio: f64, iterations: u32) {
	let mut out_width = (width as f64 * ratio).round() as u32;
	let mut out_height = 500_000;

	fix_ratio(width, height, &mut out_width, &mut out_height).unwrap();

	let mut t_min: f64 = f64::MAX;
	for _ in 0..iterations {
		let t = resize(pixels, width, height, cs, out_width, out_height);
		if t < t_min {
			t_min = t;
		}
	}

	println!("    to {:4}x{:4} {:6.2}ms", out_width, out_height, t_min);
}

fn do_bench_sizes(name: &str, pixels: &[u8], width: u32, height: u32, cs: ColorSpace, iterations: u32) {
	println!("{}x{} {}", width, height, name);

	do_bench(pixels, width, height, cs, 0.01, iterations);
	do_bench(pixels, width, height, cs, 0.125, iterations);
	do_bench(pixels, width, height, cs, 0.8, iterations);
	do_bench(pixels, width, height, cs, 2.14, iterations);
}

fn main() {
	let args: Vec<String> = env::args().collect();
	if args.len() < 2 || args.len() > 3 {
		eprintln!("Usage: {} <path.png> [colorspace]", args[0]);
		process::exit(1);
	}

	let iterations: u32 = env::var("OILITERATIONS")
		.ok()
		.and_then(|s| s.parse().ok())
		.filter(|&n| n > 0)
		.unwrap_or(100);
	eprintln!("Iterations: {}", iterations);

	let image = load_rgba_png(&args[1]);

	let spaces: &[(&str, ColorSpace)] = &[
		("G", ColorSpace::G),
		("RGB", ColorSpace::RGB),
		("RGBA", ColorSpace::RGBA),
	];

	// Filter to a specific colorspace if requested
	let filter: Option<&str> = args.get(2).map(|s| s.as_str());

	if let Some(name) = filter {
		let entry = spaces.iter().find(|(n, _)| *n == name);
		match entry {
			Some((n, cs)) => {
				let pixels = match *cs {
					ColorSpace::G => rgba_to_g(&image.pixels, image.width, image.height),
					ColorSpace::RGB => rgba_to_rgb(&image.pixels, image.width, image.height),
					_ => image.pixels.clone(),
				};
				do_bench_sizes(n, &pixels, image.width, image.height, *cs, iterations);
			}
			None => {
				eprintln!("Colorspace not recognized. Options: G, RGB, RGBA");
				process::exit(1);
			}
		}
	} else {
		for (name, cs) in spaces {
			let pixels = match *cs {
				ColorSpace::G => rgba_to_g(&image.pixels, image.width, image.height),
				ColorSpace::RGB => rgba_to_rgb(&image.pixels, image.width, image.height),
				_ => image.pixels.clone(),
			};
			do_bench_sizes(name, &pixels, image.width, image.height, *cs, iterations);
		}
	}
}
