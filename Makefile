CFLAGS ?= -O2
CFLAGS += -Wall -pedantic
-include local.mk
# Resolve quoted includes from the root (headers) and tests/ (jxl_testutil.h)
# regardless of how local.mk sets CFLAGS.
CFLAGS += -I. -Itests

OIL_OBJS = oil_resample.o
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
OIL_OBJS += oil_resample_neon.o
else ifneq ($(filter x86_64,$(shell uname -m)),)
OIL_OBJS += oil_resample_sse2.o oil_resample_avx2.o
endif

all: test test_libjpeg test_libpng test_libjxl test_jxl_cancel test_jxl_regress jxlmembench jxlstarvebench jxlperfbench imgscale benchmark coeffbench
oil_resample_sse2.o: oil_resample_sse2.c oil_resample_internal.h
	$(CC) $(CFLAGS) -msse2 -c -o $@ $<
oil_resample_avx2.o: oil_resample_avx2.c oil_resample_internal.h
	$(CC) $(CFLAGS) -mavx2 -mfma -c -o $@ $<
oil_resample_neon.o: oil_resample_neon.c oil_resample.h oil_resample_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<
oil_libjxl.o: oil_libjxl.c oil_libjxl.h oil_resample.h
	$(CC) $(CFLAGS) -DHWY_SHARED_DEFINE -c -o $@ $<
tests/jxl_testutil.o: tests/jxl_testutil.c tests/jxl_testutil.h
	$(CC) $(CFLAGS) -c -o $@ $<
test: tests/test.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) tests/test.c -o $@ -lm
test_libjpeg: tests/test_libjpeg.c $(OIL_OBJS) oil_libjpeg.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjpeg.o tests/test_libjpeg.c -o $@ $(LDFLAGS) -ljpeg -lm
test_libpng: tests/test_libpng.c $(OIL_OBJS) oil_libpng.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libpng.o tests/test_libpng.c -o $@ $(LDFLAGS) -lpng -lm
test_libjxl: tests/test_libjxl.c $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o tests/test_libjxl.c -o $@ $(LDFLAGS) -ljxl -ljxl_threads -lpthread -lm
test_jxl_cancel: tests/test_jxl_cancel.c $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o tests/test_jxl_cancel.c -o $@ $(LDFLAGS) -ljxl -ljxl_threads -lpthread -lm
test_jxl_regress: tests/test_jxl_regress.c $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o tests/test_jxl_regress.c -o $@ $(LDFLAGS) -ljxl -ljxl_threads -lpthread -lm
jxlmembench: bench/jxlmembench.c $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o bench/jxlmembench.c -o $@ $(LDFLAGS) -ljxl -ljxl_threads -lpthread -lm
jxlperfbench: bench/jxlperfbench.c $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o bench/jxlperfbench.c -o $@ $(LDFLAGS) -ljxl -ljxl_threads -lpthread -lm
jxlstarvebench: bench/jxlstarvebench.c $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjxl.o tests/jxl_testutil.o bench/jxlstarvebench.c -o $@ $(LDFLAGS) -ljxl -ljxl_threads -lpthread -lm
imgscale: $(OIL_OBJS) oil_libjpeg.o oil_libpng.o imgscale.c
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjpeg.o oil_libpng.o imgscale.c -o $@ $(LDFLAGS) -ljpeg -lpng -lm
benchmark: bench/benchmark.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) bench/benchmark.c -o $@ $(LDFLAGS) -lpng -lm
coeffbench: bench/coeffbench.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) bench/coeffbench.c -o $@ -lm
oilview: $(OIL_OBJS) oil_libjpeg.o oil_libpng.o oilview.c
	$(CC) $(CFLAGS) `pkg-config --cflags gtk+-3.0` $(OIL_OBJS) oil_libjpeg.o oil_libpng.o oilview.c -o $@ $(LDFLAGS) `pkg-config --libs gtk+-3.0` -ljpeg -lpng -lm -lX11
sdltest: $(OIL_OBJS) oil_libjpeg.o oil_libpng.o oil_libjxl.o sdltest.c
	$(CC) $(CFLAGS) -DHWY_SHARED_DEFINE $(OIL_OBJS) oil_libjpeg.o oil_libpng.o oil_libjxl.o sdltest.c -o $@ $(LDFLAGS) -lSDL3 -ljpeg -lpng -ljxl -ljxl_threads -lpthread -lm
clean:
	rm -rf test test.dSYM test_libjpeg test_libjpeg.dSYM test_libpng test_libpng.dSYM test_libjxl test_libjxl.dSYM oil_resample.o oil_resample_sse2.o oil_resample_avx2.o oil_resample_neon.o oil_libpng.o oil_libjpeg.o oil_libjxl.o tests/jxl_testutil.o imgscale oilview benchmark coeffbench sdltest jxlmembench jxlstarvebench jxlperfbench test_jxl_cancel test_jxl_regress
