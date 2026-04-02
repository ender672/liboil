use std::env;
use std::fs;
use std::path::Path;
use std::process;

enum Format {
    Jpeg,
    Png,
}

fn detect_format(path: &str) -> Option<Format> {
    match Path::new(path).extension().and_then(|e| e.to_str()) {
        Some(ext) => match ext.to_ascii_lowercase().as_str() {
            "jpg" | "jpeg" => Some(Format::Jpeg),
            "png" => Some(Format::Png),
            _ => None,
        },
        None => None,
    }
}

fn default_output(format: &Format) -> &'static str {
    match format {
        Format::Jpeg => "output.jpg",
        Format::Png => "output.png",
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} WIDTH HEIGHT INPUT [OUTPUT]", args[0]);
        process::exit(1);
    }

    let mut width: u32 = args[1].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid width.");
        process::exit(1);
    });
    let mut height: u32 = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Error: Invalid height.");
        process::exit(1);
    });

    let input_path = &args[3];
    let format = detect_format(input_path).unwrap_or_else(|| {
        eprintln!("Error: Unsupported file format. Use .jpg, .jpeg, or .png");
        process::exit(1);
    });

    let output_path = if args.len() > 4 {
        args[4].as_str()
    } else {
        default_output(&format)
    };

    match format {
        Format::Jpeg => resize_jpeg(input_path, &mut width, &mut height, output_path),
        Format::Png => resize_png(input_path, &mut width, &mut height, output_path),
    }

    eprintln!("Resized to {}x{} -> {}", width, height, output_path);
}

fn resize_jpeg(input_path: &str, width: &mut u32, height: &mut u32, output_path: &str) {
    #[cfg(feature = "ffi")]
    let encoded = {
        let path = Path::new(input_path);
        let (src_w, src_h) = oil::jpeg_ffi::jpeg_dimensions_file(path).unwrap_or_else(|e| {
            eprintln!("Unable to read JPEG header: {:?}", e);
            process::exit(1);
        });
        oil::jpeg::fix_ratio(src_w, src_h, width, height).unwrap_or_else(|e| {
            eprintln!("Error adjusting aspect ratio: {:?}", e);
            process::exit(1);
        });
        oil::jpeg_ffi::resize_jpeg_file(path, *width, *height, 94)
    };

    #[cfg(not(feature = "ffi"))]
    let encoded = {
        let input_data = fs::read(input_path).unwrap_or_else(|e| {
            eprintln!("Unable to open source file: {}", e);
            process::exit(1);
        });
        let mut decoder = jpeg_decoder::Decoder::new(&input_data[..]);
        decoder.read_info().unwrap_or_else(|e| {
            eprintln!("Unable to read JPEG header: {}", e);
            process::exit(1);
        });
        let info = decoder.info().unwrap();
        oil::jpeg::fix_ratio(info.width as u32, info.height as u32, width, height)
            .unwrap_or_else(|e| {
                eprintln!("Error adjusting aspect ratio: {:?}", e);
                process::exit(1);
            });
        oil::jpeg::resize_jpeg(&input_data, *width, *height, 94)
    };

    let encoded = encoded.unwrap_or_else(|e| {
        eprintln!("Error resizing image: {:?}", e);
        process::exit(1);
    });

    fs::write(output_path, &encoded).unwrap_or_else(|e| {
        eprintln!("Unable to write output file: {}", e);
        process::exit(1);
    });
}

fn resize_png(input_path: &str, width: &mut u32, height: &mut u32, output_path: &str) {
    let input_data = fs::read(input_path).unwrap_or_else(|e| {
        eprintln!("Unable to open source file: {}", e);
        process::exit(1);
    });

    let (src_w, src_h) = oil::png::png_dimensions(&input_data).unwrap_or_else(|e| {
        eprintln!("Unable to read PNG header: {:?}", e);
        process::exit(1);
    });

    oil::jpeg::fix_ratio(src_w, src_h, width, height).unwrap_or_else(|e| {
        eprintln!("Error adjusting aspect ratio: {:?}", e);
        process::exit(1);
    });

    let encoded = oil::png::resize_png(&input_data, *width, *height).unwrap_or_else(|e| {
        eprintln!("Error resizing image: {:?}", e);
        process::exit(1);
    });

    fs::write(output_path, &encoded).unwrap_or_else(|e| {
        eprintln!("Unable to write output file: {}", e);
        process::exit(1);
    });
}
