use std::env;
use std::fs;
use std::process;

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
    let output_path = if args.len() > 4 {
        &args[4]
    } else {
        "output.jpg"
    };

    #[cfg(feature = "ffi")]
    let encoded = {
        use std::path::Path;
        let path = Path::new(input_path);
        let (src_w, src_h) = oil::jpeg_ffi::jpeg_dimensions_file(path).unwrap_or_else(|e| {
            eprintln!("Unable to read JPEG header: {:?}", e);
            process::exit(1);
        });
        oil::jpeg::fix_ratio(src_w, src_h, &mut width, &mut height).unwrap_or_else(|e| {
            eprintln!("Error adjusting aspect ratio: {:?}", e);
            process::exit(1);
        });
        oil::jpeg_ffi::resize_jpeg_file(path, width, height, 94)
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
        oil::jpeg::fix_ratio(info.width as u32, info.height as u32, &mut width, &mut height)
            .unwrap_or_else(|e| {
                eprintln!("Error adjusting aspect ratio: {:?}", e);
                process::exit(1);
            });
        oil::jpeg::resize_jpeg(&input_data, width, height, 94)
    };

    let encoded = encoded.unwrap_or_else(|e| {
        eprintln!("Error resizing image: {:?}", e);
        process::exit(1);
    });

    fs::write(output_path, &encoded).unwrap_or_else(|e| {
        eprintln!("Unable to write output file: {}", e);
        process::exit(1);
    });

    #[cfg(feature = "ffi")]
    eprintln!("Resized to {}x{} -> {} (ffi)", width, height, output_path);
    #[cfg(not(feature = "ffi"))]
    eprintln!("Resized to {}x{} -> {} (pure-rust)", width, height, output_path);
}
