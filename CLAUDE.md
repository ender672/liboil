# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This branch (`mozilla-cpp-port`) is a port of liboil's core downscaling routines to C++ for direct inclusion in the Firefox project.

The upstream liboil is a C library for fast, accurate image resizing using Catmull-Rom (bicubic) interpolation with proper sRGB gamma correction and premultiplied alpha handling. It processes images scanline-by-scanline to minimize memory usage.

## Port Goals

The port involves four steps, in order:

1. **Trim to core downscaling**: Keep only the core resampler and SIMD backends, downscaling paths only, and only the BGRX and BGRA non-gamma color spaces.
2. **Port to C++**: Convert the remaining C code to idiomatic C++.
3. **Adopt Firefox coding style**: Conform to the [Mozilla/Firefox C++ coding style](https://firefox-source-docs.mozilla.org/code-quality/coding-style/coding_style_cpp.html).

## Build Commands

```bash
make test          # build test binary
./test             # run tests (assert-based, compares against long-double reference impl)
make benchmark     # build perf benchmark
make clean         # remove all build artifacts
```

Compiler settings are in `local.mk` (gitignored, included by Makefile). On macOS/Homebrew, also add `-I/opt/homebrew/include` and `-L/opt/homebrew/lib` in `local.mk`.

Dependencies: libjpeg, libpng, libm. On macOS: `brew install jpeg libpng`. Optional: SDL2 (`make sdltest`), GTK+3 (`make oilview`).

Note: Build commands and dependencies will change as the port progresses.

## Architecture

The code that matters for this port:

- **Core resampler** (`oil_resample.h/c`): The scaling engine. `struct oil_scale` holds all state. Callers feed input scanlines with `oil_scale_in()` and read output with `oil_scale_out()`. Supports color spaces G, GA, RGB, RGBA, CMYK. The filter widens its tap count automatically when downsampling to prevent aliasing.

- **SIMD backends**: SSE2 on x86_64 (`oil_resample_sse2.c`), AVX2 on x86_64, NEON on AArch64 (`oil_resample_neon.c`). Each provides its own `oil_scale_in_*`/`oil_scale_out_*` entry points. Built unconditionally for the detected architecture.

