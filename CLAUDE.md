# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

liboil is a C library for fast, accurate image resizing using Catmull-Rom (bicubic) interpolation. It processes images scanline-by-scanline via a ring buffer to minimize memory usage, with proper sRGB gamma correction and premultiplied alpha handling.

## Build Commands

```bash
make test          # build test binary
./test             # run tests (assert-based, compares against long-double reference impl)
make imgscale      # build CLI resizer: ./imgscale <width> <height> <input> <output>
make benchmark     # build perf benchmark
make clean         # remove all build artifacts
```

Compiler settings are in `local.mk` (gitignored, included by Makefile). Toggle between `-O3` and `-O0 -g` there for release/debug builds. On macOS/Homebrew, also add `-I/opt/homebrew/include` and `-L/opt/homebrew/lib` in `local.mk`.

Dependencies: libjpeg, libpng, libm. On macOS: `brew install jpeg libpng`. Optional: SDL2 (`make sdltest`), GTK+3 (`make oilview`).

## Architecture

Three layers, all C with no external build system beyond make:

- **Core resampler** (`oil_resample.h/c`): The scaling engine. `struct oil_scale` holds all state. Callers feed input scanlines with `oil_scale_in()` and read output with `oil_scale_out()`. Supports color spaces G, GA, RGB, RGBA, CMYK. Has optional SIMD paths: SSE2 on x86\_64 (`oil_resample_sse2.c`), NEON on AArch64 (`oil_resample_neon.c`). Dispatch is compile-time via `OIL_USE_SSE2`/`OIL_USE_NEON` in `oil_resample_internal.h`. Build with `SIMD=none` to disable. The filter widens its tap count automatically when downsampling to prevent aliasing.

- **JPEG wrapper** (`oil_libjpeg.h/c`): Integrates with `libjpeg`'s `jpeg_decompress_struct` to feed scanlines into the core resampler.

- **PNG wrapper** (`oil_libpng.h/c`): Integrates with `libpng`. Handles both interlaced (Adam7, requires full image buffer) and non-interlaced PNGs.

## Code Conventions

- 8-space tab width, 100-char line ruler (see `.vscode/settings.json`)
- `oil_` prefix for all public API functions
- Error returns: -1 invalid args, -2 memory failure, -3 range error
- Test tolerance: output pixels must be within 0.06 of the long-double reference (out of 255)
