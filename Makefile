CFLAGS += -O3 -Wall -pedantic

all: test imgscale benchmark
test: test.c oil_resample.o
	$(CC) $(CFLAGS) oil_resample.o test.c -o $@ -lm
imgscale: oil_resample.o oil_libjpeg.o oil_libpng.o imgscale.c
	$(CC) $(CFLAGS) oil_resample.o oil_libjpeg.o oil_libpng.o imgscale.c -o $@ $(LDFLAGS) -ljpeg -lpng -lm
benchmark: benchmark.c oil_resample.o
	$(CC) $(CFLAGS) oil_resample.o benchmark.c -o $@ $(LDFLAGS) -lpng -lm
oilview: oil_resample.o oil_libjpeg.o oil_libpng.o oilview.c
	$(CC) $(CFLAGS) `pkg-config --cflags gtk+-3.0` oil_resample.o oil_libjpeg.o oil_libpng.o oilview.c -o $@ $(LDFLAGS) `pkg-config --libs gtk+-3.0` -ljpeg -lpng -lm -lX11
clean:
	rm -rf test test.dSYM oil_resample.o oil_libpng.o oil_libjpeg.o imgscale oilview benchmark
