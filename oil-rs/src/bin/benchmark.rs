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

fn load_jpeg(path: &str) -> BenchImage {
	let data = fs::read(path).unwrap_or_else(|e| {
		eprintln!("Unable to open file: {}", e);
		process::exit(1);
	});
	let mut decoder = jpeg_decoder::Decoder::new(&data[..]);
	let pixels = decoder.decode().unwrap_or_else(|e| {
		eprintln!("JPEG decode error: {}", e);
		process::exit(1);
	});
	let info = decoder.info().unwrap();
	if info.pixel_format != jpeg_decoder::PixelFormat::RGB24 {
		eprintln!("Input image must be RGB.");
		process::exit(1);
	}
	BenchImage {
		pixels,
		width: info.width as u32,
		height: info.height as u32,
	}
}

fn resize(image: &BenchImage, out_width: u32, out_height: u32) -> f64 {
	let cs = ColorSpace::RGB;
	let cmp = cs.components();
	let in_stride = image.width as usize * cmp;

	let mut outbuf = vec![0u8; out_width as usize * cmp];

	let start = Instant::now();
	let mut scaler = OilScale::new(image.height, out_height, image.width, out_width, cs).unwrap();
	let mut in_line = 0usize;
	for _ in 0..out_height {
		while scaler.slots() > 0 {
			let row_start = in_line * in_stride;
			scaler.push_scanline(&image.pixels[row_start..row_start + in_stride]);
			in_line += 1;
		}
		scaler.read_scanline(&mut outbuf);
	}
	let elapsed = start.elapsed();

	elapsed.as_secs_f64() * 1000.0
}

fn do_bench(image: &BenchImage, ratio: f64, iterations: u32) {
	let mut out_width = (image.width as f64 * ratio).round() as u32;
	let mut out_height = 500_000;

	fix_ratio(image.width, image.height, &mut out_width, &mut out_height).unwrap();

	let mut t_min: f64 = f64::MAX;
	for _ in 0..iterations {
		let t = resize(image, out_width, out_height);
		if t < t_min {
			t_min = t;
		}
	}

	println!("    to {:4}x{:4} {:6.2}ms", out_width, out_height, t_min);
}

fn main() {
	let args: Vec<String> = env::args().collect();
	if args.len() != 2 {
		eprintln!("Usage: {} <path.jpg>", args[0]);
		process::exit(1);
	}

	let iterations: u32 = env::var("OILITERATIONS")
		.ok()
		.and_then(|s| s.parse().ok())
		.filter(|&n| n > 0)
		.unwrap_or(100);
	eprintln!("Iterations: {}", iterations);

	let image = load_jpeg(&args[1]);
	println!("{}x{} RGB", image.width, image.height);

	do_bench(&image, 0.01, iterations);
	do_bench(&image, 0.125, iterations);
	do_bench(&image, 0.8, iterations);
	do_bench(&image, 2.14, iterations);
}
