# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

liboil is a C library for fast, accurate image resizing using Catmull-Rom (bicubic) interpolation with proper sRGB gamma correction and premultiplied alpha handling. It processes images scanline-by-scanline to minimize memory usage.

## Build Commands

```bash
make test          # build test binary
./test             # run tests (assert-based, compares against long-double reference impl)
make benchmark     # build perf benchmark. The benchmark runs many iterations and prints the best of. One run is always enough.
make clean         # remove all build artifacts
```

Compiler settings are in `local.mk` (gitignored, included by Makefile). On macOS/Homebrew, also add `-I/opt/homebrew/include` and `-L/opt/homebrew/lib` in `local.mk`.

Dependencies: libjpeg, libpng, libm. On macOS: `brew install jpeg libpng`. Optional: SDL2 (`make sdltest`), GTK+3 (`make oilview`).

## Architecture

Three layers, all C with no external build system beyond make:

- **Core resampler** (`oil_resample.h/c`): The scaling engine. `struct oil_scale` holds all state. Callers feed input scanlines with `oil_scale_in()` and read output with `oil_scale_out()`. Supports color spaces G, GA, RGB, RGBA, CMYK. The filter widens its tap count automatically when downsampling to prevent aliasing.

- **SIMD backends**: SSE2 on x86_64 (`oil_resample_sse2.c`), NEON on AArch64 (`oil_resample_neon.c`). Each provides its own `oil_scale_in_*`/`oil_scale_out_*` entry points. Built unconditionally for the detected architecture.

- **JPEG wrapper** (`oil_libjpeg.h/c`): Integrates with `libjpeg`'s `jpeg_decompress_struct` to feed scanlines into the core resampler.

- **PNG wrapper** (`oil_libpng.h/c`): Integrates with `libpng`. Handles both interlaced (Adam7, requires full image buffer) and non-interlaced PNGs.
