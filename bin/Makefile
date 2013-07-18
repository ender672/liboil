dev: CFLAGS=-ggdb3
dev: all

a.out: resize.c ../ext/oil.c ../ext/jpeg.c ../ext/png.c ../ext/resample.c
	gcc $(CFLAGS) -Ofast -march=native -Wall -Wextra -Wno-unused-parameter -Wpointer-arith -pedantic -I../ext ../ext/ppm.c ../ext/jpeg.c ../ext/png.c ../ext/resample.c ../ext/oil.c resize.c -lpng -ljpeg

clean:
	rm a.out
	rm resized.jpg

test-yx: all
	time ./a.out --rgbx --nowrite --noprescale ../../hubble.jpg resized.jpg 1900 1900

test-xy: all
	time ./a.out --rgbx --nowrite --xy --noprescale ../../hubble.jpg resized.jpg 1900 1900

test-x: all
	time ./a.out --rgbx --nobox --nowrite --noprescale ../../hubble.jpg resized.jpg 1900 10110

test-y: all
	time ./a.out --rgbx --nobox --nowrite --noprescale ../../hubble.jpg resized.jpg 10110 1900

all: a.out

callgrind: all
	valgrind --tool=callgrind ./a.out --rgbx --nowrite ../../canon.jpg resized.jpg 1900 1900

valgrind: all
	valgrind ./a.out --rgbx ../../hubble.jpg resized.jpg 1900 1900

strace: all
	strace ./a.out ../../hubble.jpg resized.jpg 1900 1900

gdb: all
	gdb --args ./a.out ../../hubble.jpg resized.jpg 1900 1900

show: all
	eog resized.jpg