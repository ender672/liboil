CFLAGS = -ggdb3 -Ofast -march=native -Wall -Wno-unused-function -pedantic
SRC = oil.c jpeg.c png.c ppm.c resample.c

OBJ = $(SRC:.c=.o)
LIB = liboil.a

all: $(LIB)

$(LIB): $(OBJ)
	@rm -f $(AR)
	$(AR) -rcs $@ $(OBJ)

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(LIB) resize resize.o $(OBJ)
	rm -f resized.jpg

# Example binary

resize: resize.o $(LIB)
	$(CC) -o $@ resize.o $(LIB) -ljpeg -lpng

# Tests

hubble.jpg:
	wget "http://www.spacetelescope.org/static/archives/print_posters/large/hst_print_poster_0015.jpg" -O $@

resized.jpg: resize hubble.jpg
	@rm -f $@
	time ./resize --rgbx --in hubble.jpg --out resized.jpg --width 900 --height 900

callgrind: resize hubble.jpg
	valgrind --tool=callgrind ./resize --rgbx --in hubble.jpg --width 1900 --height 1900

valgrind: resize hubble.jpg
	valgrind ./resize --rgbx --in hubble.jpg --out resized.jpg --width 1900 --height 1900

strace: resize hubble.jpg
	strace ./resize --in hubble.jpg --out resized.jpg --width 1900 --height 1900

gdb: resize hubble.jpg
	gdb --args ./resize --in hubble.jpg --out resized.jpg --width 1900 --height 1900

test: resized.jpg
