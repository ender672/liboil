liboil
======

A C library for resizing images. It currently does all resizing with a bicubic (catmull-rom) interpolator. This library aims for fast performance, low memory use, and accuracy.

Purpose
-------

liboil aims to provide excellent general-purpose image thumbnailing and is optimized for low memory and CPU use.

liboil is not very configurable -- it currently only has one interpolator (catmull-rom). It is not suited for scenarios where you want to customize your settings by hand for each image.

An example use-case is a web server that thumbnails user-uploaded images.

Features
--------

 * Antialiasing - the interpolator is scaled when shrinking images.
 * Color space aware - liboil converts images to linear RGB for processing.
 * Pre-multiplied alpha - avoids artifacts when resizing with transparency.
 * SIMD acceleration - SSE2 on x86\_64, NEON on AArch64 (ARM64).

imgscale
--------

The liboil repository include a command-line tool for resizing JPEG and PNG images. This resizer reads the original image from stdin and writes the resized image to stdout.

For example, to resize in.jpg to fit in a 400x800 box while preserving the aspect ratio: 

    ./imgscale 400 800 in.jpg out.jpg

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

Reference Documentation
-----------------------

Refer to oil_resample.h for reference documentation.

Building
--------

Dependencies: libjpeg, libpng, libm.

On macOS with Homebrew:

    brew install jpeg libpng

The Makefile auto-detects the architecture and enables SIMD (SSE2 on x86\_64, NEON on ARM64). To disable SIMD:

    make SIMD=none

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
