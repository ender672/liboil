# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

liboil is a C library for fast, accurate image resizing using Catmull-Rom (bicubic) interpolation with proper sRGB gamma correction and premultiplied alpha handling. It processes images scanline-by-scanline to minimize memory usage.

## Build Commands

```bash
make test          # build test binary
./test             # run tests (assert-based, compares against long-double reference impl)
make benchmark     # build perf benchmark
make clean         # remove all build artifacts
```

Compiler settings are in `local.mk` (gitignored, included by Makefile). On macOS/Homebrew, also add `-I/opt/homebrew/include` and `-L/opt/homebrew/lib` in `local.mk`.

Dependencies: libjpeg, libpng, libjxl, libm. On macOS: `brew install jpeg libpng jpeg-xl`. Optional: SDL2 (`make sdltest`), GTK+3 (`make oilview`).

## Architecture

All C, no external build system beyond make:

- **Core resampler** (`oil_resample.h/c`): The scaling engine. `struct oil_scale` holds all state. Callers feed input scanlines with `oil_scale_in()` and read output with `oil_scale_out()`. Supports color spaces G, GA, RGB, RGBA, CMYK. The filter widens its tap count automatically when downsampling to prevent aliasing.

- **SIMD backends**: SSE2/AVX2 on x86_64 (`oil_resample_sse2.c`, `oil_resample_avx2.c`), NEON on AArch64 (`oil_resample_neon.c`). Each provides its own `oil_scale_in_*`/`oil_scale_out_*` entry points. Built for the detected architecture.

- **JPEG wrapper** (`oil_libjpeg.h/c`): Integrates with `libjpeg`'s `jpeg_decompress_struct` to feed scanlines into the core resampler.

- **PNG wrapper** (`oil_libpng.h/c`): Integrates with `libpng`. Handles both interlaced (Adam7, requires full image buffer) and non-interlaced PNGs.

- **JPEG XL helpers** (`oil_libjxl.h/c`, `oil_jxl_rowbuf.h/c`, `oil_jxl_threads.h/c`, `oil_jxl_waiter.h`): unlike libjpeg/libpng, `libjxl` is *not* wrapped. Its decoder has no incremental pull API (one `JxlDecoderProcessInput` decodes the whole frame, dispatching partials to workers out of order), so a single wrapper shape can't fit every caller. Instead liboil exposes a kit of composable helpers:
  - `oil_jxl_rowbuf` — out-of-order → in-order reorder buffer (the decode's image-out sink). Its core is portable C11 atomics with **no** platform threading calls; the two blocking points delegate to a caller-supplied `oil_jxl_waiter`.
  - `oil_jxl_run_decode(dec, fmt, rb)` — drives `JxlDecoderProcessInput` to completion into a rowbuf; the caller runs it on a thread it owns.
  - `oil_jxl_threads` — the *optional* pthreads pieces: the cancellable `oil_jxl_runner` (interrupts a decode mid-frame) and the `oil_jxl_condvar_waiter` (the rowbuf's blocking primitive). This is the only jxl unit that uses pthreads.
  - `oil_jxl_resample(...)` — a one-call convenience composed from all of the above; spawns one decode thread and pulls/scales on the calling thread. `imgscale.c` shows the manual (streaming) composition instead.

  In every case the caller owns the `JxlDecoder`: bind a parallel runner (stock `JxlThreadParallelRunner`, `oil_jxl_runner`, or its own), subscribe events, supply the codestream, and drive to `JXL_DEC_BASIC_INFO` before using the helpers.

## Conventions

- Keep commit messages CONCISE: a short subject line, and a body only covering the "why" that isn't obvious. Skip the body for small or self-evident changes.
- The image-format integrations (`oil_libjpeg`, `oil_libpng`, `oil_jxl`) should primarily be a *kit*, not a wrapper: expose composable pieces — setup, a per-row decode primitive into a caller-owned buffer, and the `oil_scale` for the caller to drive (so SIMD/pipelined/streaming callers compose their own loop). An all-in-one convenience path (e.g. `oil_libjpeg_read_scanline`) is fine layered on top, but the kit primitives are the primary interface; consumers should reach for them rather than the wrapper's internals.
