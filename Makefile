CFLAGS += -O3 -march=native -Wall -pedantic

all: test imgscale
test: test.c oil_resample.c
	$(CC) $(CFLAGS) test.c -o $@ -lm
imgscale: oil_resample.o oil_libjpeg.o oil_libpng.o imgscale.c
	$(CC) $(CFLAGS) oil_resample.o oil_libjpeg.o oil_libpng.o imgscale.c -o $@ $(LDFLAGS) -ljpeg -lpng -lm
clean:
	rm -rf test test.dSYM oil_resample.o imgscale
