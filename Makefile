CFLAGS += -O3 -march=native -Wall -pedantic

all: test jpgscale pngscale
test: test.c resample.c
	$(CC) $(CFLAGS) test.c -o $@ -lm
jpgscale: resample.o jpgscale.c
	$(CC) $(CFLAGS) resample.o jpgscale.c -o $@ -ljpeg -lm
pngscale: resample.o pngscale.c
	$(CC) $(CFLAGS) resample.o pngscale.c -o $@ -lpng -lm
clean:
	rm -rf test test.dSYM resample.o jpgscale pngscale
