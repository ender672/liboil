CFLAGS += -O3 -march=native -Wall -pedantic

all: test jpgscale pngscale
test: test.c oil_resample.c
	$(CC) $(CFLAGS) test.c -o $@ -lm
jpgscale: oil_resample.o jpgscale.c
	$(CC) $(CFLAGS) oil_resample.o jpgscale.c -o $@ $(LDFLAGS) -ljpeg -lm
pngscale: oil_resample.o pngscale.c
	$(CC) $(CFLAGS) oil_resample.o pngscale.c -o $@ $(LDFLAGS) -lpng -lm
clean:
	rm -rf test test.dSYM oil_resample.o jpgscale pngscale
