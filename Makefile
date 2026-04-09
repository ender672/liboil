CFLAGS ?= -O2
CFLAGS += -Wall -pedantic
-include local.mk

OIL_OBJS = oil_resample.o
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
OIL_OBJS += oil_resample_neon.o
else ifneq ($(filter x86_64,$(shell uname -m)),)
OIL_OBJS += oil_resample_sse2.o oil_resample_avx2.o
endif

all: test benchmark
oil_resample_sse2.o: oil_resample_sse2.c oil_resample_internal.h
	$(CC) $(CFLAGS) -msse2 -c -o $@ $<
oil_resample_avx2.o: oil_resample_avx2.c oil_resample_internal.h
	$(CC) $(CFLAGS) -mavx2 -mfma -c -o $@ $<
oil_resample_neon.o: oil_resample_neon.c oil_resample.h oil_resample_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<
test: test.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) test.c -o $@ -lm
benchmark: benchmark.c $(OIL_OBJS)
	$(CC) $(CFLAGS) $(OIL_OBJS) benchmark.c -o $@ $(LDFLAGS) -lpng -lm
clean:
	rm -rf test test.dSYM oil_resample.o oil_resample_sse2.o oil_resample_avx2.o oil_resample_neon.o benchmark
