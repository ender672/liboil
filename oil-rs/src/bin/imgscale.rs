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
    let input_data = fs::read(input_path).unwrap_or_else(|e| {
        eprintln!("Unable to open source file: {}", e);
        process::exit(1);
    });

    // Decode header to get dimensions for aspect ratio fix
    let mut decoder = jpeg_decoder::Decoder::new(&input_data[..]);
    decoder.read_info().unwrap_or_else(|e| {
        eprintln!("Unable to read JPEG header: {}", e);
        process::exit(1);
    });
    let info = decoder.info().unwrap();

    oil::jpeg::fix_ratio(info.width as u32, info.height as u32, &mut width, &mut height)
        .unwrap_or_else(|e| {
            eprintln!("Error adjusting aspect ratio: {}", e);
            process::exit(1);
        });

    let encoded = oil::jpeg::resize_jpeg(&input_data, width, height, 94).unwrap_or_else(|e| {
        eprintln!("Error resizing image: {}", e);
        process::exit(1);
    });

    let output_path = if args.len() > 4 {
        &args[4]
    } else {
        "output.jpg"
    };

    fs::write(output_path, &encoded).unwrap_or_else(|e| {
        eprintln!("Unable to write output file: {}", e);
        process::exit(1);
    });

    eprintln!("Resized to {}x{} -> {}", width, height, output_path);
}
