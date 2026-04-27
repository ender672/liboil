CFLAGS ?= -O2
CFLAGS += -Wall -pedantic
-include local.mk

OIL_OBJS = oil_resample.o
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
OIL_OBJS += oil_resample_neon.o
else ifneq ($(filter x86_64,$(shell uname -m)),)
OIL_OBJS += oil_resample_sse2.o oil_resample_avx2.o
endif

all: test imgscale benchmark coeffbench
oil_resample_sse2.o: oil_resample_sse2.c oil_resample_internal.h
	$(CC) $(CFLAGS) -msse2 -c -o $@ $<
oil_resample_avx2.o: oil_resample_avx2.c oil_resample_internal.h
	$(CC) $(CFLAGS) -mavx2 -mfma -c -o $@ $<
oil_resample_neon.o: oil_resample_neon.c oil_resample.h oil_resample_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<
test: test.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) test.c -o $@ -lm
imgscale: $(OIL_OBJS) oil_libjpeg.o oil_libpng.o imgscale.c
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjpeg.o oil_libpng.o imgscale.c -o $@ $(LDFLAGS) -ljpeg -lpng -lm
benchmark: benchmark.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) benchmark.c -o $@ $(LDFLAGS) -lpng -lm
coeffbench: coeffbench.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) coeffbench.c -o $@ -lm
oilview: $(OIL_OBJS) oil_libjpeg.o oil_libpng.o oilview.c
	$(CC) $(CFLAGS) `pkg-config --cflags gtk+-3.0` $(OIL_OBJS) oil_libjpeg.o oil_libpng.o oilview.c -o $@ $(LDFLAGS) `pkg-config --libs gtk+-3.0` -ljpeg -lpng -lm -lX11
sdltest: $(OIL_OBJS) oil_libjpeg.o oil_libpng.o sdltest.c
	$(CC) $(CFLAGS) $(OIL_OBJS) oil_libjpeg.o oil_libpng.o sdltest.c -o $@ $(LDFLAGS) -lSDL3 -ljpeg -lpng -lm
clean:
	rm -rf test test.dSYM oil_resample.o oil_resample_sse2.o oil_resample_avx2.o oil_resample_neon.o oil_libpng.o oil_libjpeg.o imgscale oilview benchmark coeffbench sdltest
