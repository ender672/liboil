CFLAGS += -Os -Wall -pedantic -I/usr/local/include

resample.o: resample.c resample.h
test: resample.o test.c
	$(CC) $(CFLAGS) resample.o test.c -o $@ -lm
clean:
	rm -f resample.o test test.o
