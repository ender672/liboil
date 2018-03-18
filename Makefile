CFLAGS += -O3 -march=native -Wall -pedantic

all: test jpgscale pngscale tools/linear_to_srgb_table tools/srgb_to_linear_table
test: test.c resample.c
	$(CC) $(CFLAGS) test.c -o $@ -lm
jpgscale: resample.o jpgscale.c
	$(CC) $(CFLAGS) resample.o jpgscale.c -o $@ -ljpeg -lm
pngscale: resample.o pngscale.c
	$(CC) $(CFLAGS) resample.o pngscale.c -o $@ -lpng -lm
tools/linear_to_srgb_table:
	$(CC) $(CFLAGS) tools/linear_to_srgb_table.c -o $@ -lm
tools/srgb_to_linear_table:
	$(CC) $(CFLAGS) tools/srgb_to_linear_table.c -o $@ -lm
clean:
	rm -rf test test.dSYM resample.o jpgscale pngscale
