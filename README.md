liboil
======

A C library for resizing images. It currently does all resizing with a bicubic (catmull-rom) interpolator. This library aims for fast performance, low memory use, and accuracy.

Performance over time, broken down by colorspace and SIMD backend, is tracked at [liboil-bench.netlify.app](https://liboil-bench.netlify.app/).

Purpose
-------

liboil aims to provide excellent general-purpose image thumbnailing.

liboil is not very configurable -- it currently only has one interpolator (catmull-rom). It is not suited for scenarios where you want to customize your settings by hand for each image.

An example use-case is a web server that thumbnails user-uploaded images.

Features
--------

 * Antialiasing - the interpolator is scaled when shrinking images.
 * Color space aware - liboil converts images to linear RGB for processing.
 * Pre-multiplied alpha - avoids artifacts when resizing with transparency.
 * SIMD acceleration - SSE2 and AVX2 on x86_64, NEON on AArch64 (ARM64).

imgscale
--------

The liboil repository includes a command-line tool for resizing JPEG, PNG, and
JPEG XL images. The output format is chosen from the destination file
extension, so you can also transcode between formats while resizing.

For example, to resize in.jpg to fit in a 400x800 box while preserving the aspect ratio: 

    ./imgscale 400 800 in.jpg out.jpg

To resize and transcode a JPEG XL source to a PNG thumbnail:

    ./imgscale 400 800 in.jxl out.png

Example usage as a C library
----------------------------

```C
#include "oil_resample.h"

unsigned char *inbuf, *outbuf;
int i, ret;
struct oil_scale os;

oil_fix_ratio(in_width, in_height, &out_width, &out_height);
inbuf = malloc(in_width * 3);
outbuf = malloc(out_width * 3);

ret = oil_scale_init(&os, in_height, out_height, in_width, out_width, OIL_CS_RGB);
if (ret!=0) {
    fprintf(stderr, "Unable to allocate buffers.");
    exit(1);
}

for(i=0; i<out_height; i++) {
    while (oil_scale_slots(&os)) {
        fill_with_next_scanline_from_source_image(inbuf);
        oil_scale_in(&os, inbuf);
    }
    oil_scale_out(&os, outbuf);
    write_to_next_scanline_of_destination_image(outbuf);
}
```

Example: crop and scale
-----------------------

To produce a thumbnail that fills the output dimensions exactly (CSS "cover"
semantics), crop the largest sub-rect of the input that matches the output
aspect ratio, then scale that crop:

```C
#include "oil_resample.h"

unsigned char *inbuf, *outbuf;
int i, ret;
int out_width = 400, out_height = 300;
double src_x, src_y, src_width, src_height;
int fed_x, fed_y, fed_width, fed_height;
struct oil_scale os;

// Pick the crop rect inside the source image.
oil_compute_cover_rect(in_width, in_height, out_width, out_height,
    OIL_GRAVITY_CENTER, &src_x, &src_y, &src_width, &src_height);

// Expand the crop with filter halo; this is the rect the decoder must produce.
oil_required_input_rect(in_height, in_width, src_y, src_height,
    src_x, src_width, out_height, out_width,
    &fed_y, &fed_height, &fed_x, &fed_width);

inbuf = malloc(fed_width * 3);
outbuf = malloc(out_width * 3);

// in_* args describe the fed buffer; src_* args describe the crop's
// position relative to it.
ret = oil_scale_init_ex(&os, fed_height, out_height, fed_width, out_width,
    src_y - fed_y, src_height, src_x - fed_x, src_width, OIL_CS_RGB);
if (ret!=0) {
    fprintf(stderr, "Unable to allocate buffers.");
    exit(1);
}

for(i=0; i<out_height; i++) {
    while (oil_scale_slots(&os)) {
        fill_with_next_scanline_of_fed_rect(inbuf);
        oil_scale_in(&os, inbuf);
    }
    oil_scale_out(&os, outbuf);
    write_to_next_scanline_of_destination_image(outbuf);
}
```

When decoding JPEG, PNG, or JPEG XL, `oil_libjpeg_init_ex`,
`oil_libpng_init_ex`, and `oil_libjxl_init_ex` take the same `src_*` rect and
handle the decode-side cropping for you (the JPEG wrapper uses libjpeg-turbo's
skip and crop fast paths when available).

libjxl has no incremental pull API, so the JPEG XL wrapper decodes on its own
producer thread and serves finalized scanlines top-to-bottom. Unlike the JPEG
and PNG wrappers, the caller sets up the `JxlDecoder` (parallel runner, drive to
basic info) before init; see `oil_libjxl.h`.

Reference Documentation
-----------------------

Refer to oil_resample.h for reference documentation.

Related Projects
----------------

 * [ender672/oil](https://github.com/ender672/oil) - Ruby port/extension of this library.
 * [ender672/oil-scale](https://github.com/ender672/oil-scale) - Rust port of this library.

Building
--------

Dependencies: libjpeg, libpng, libjxl (libjxl_threads), libm.

On macOS with Homebrew:

    brew install jpeg libpng jpeg-xl

The Makefile auto-detects the architecture and builds the appropriate SIMD backend (SSE2/AVX2 on x86_64, NEON on ARM64).

Per-machine compiler settings go in `local.mk` (gitignored, included by the Makefile). For example, on Apple Silicon:

    CFLAGS += -O3 -mcpu=apple-m1
    LDFLAGS += -L/opt/homebrew/lib
    CFLAGS += -I/opt/homebrew/include

For a generic AArch64 target:

    CFLAGS += -O3 -march=armv8-a

Testing
-------

liboil includes a test binary that compares the output of the resizer to a reference implementation. You can build it with: 

    make test

And run it with: 

    ./test

It is recommended to run it with valgrind as well: 

    valgrind ./test

Benchmarking
------------

    make benchmark
    ./benchmark <path-to-rgba-png> [colorspace]

Set `OILITERATIONS=N` to control iteration count (default 100).
