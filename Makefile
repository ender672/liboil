CFLAGS = -Os -Wall -pedantic
LDLIBS = -lpng -ljpeg -lm

oil: oil.o resample.o yscaler.o
oil.o: oil.c
yscaler.o: yscaler.c yscaler.h
resample.o: resample.c resample.h
test.o: test.c
test: test.o resample.o
clean:
	rm -rf oil oil.o yscaler.o resample.o test test.o
