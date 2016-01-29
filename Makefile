CFLAGS += -Os -Wall -pedantic -I/usr/local/include
LDLIBS += -lm
LDFLAGS += -L/usr/local/lib

TEST_OBJS=test.o resample.o

resample.o: resample.c resample.h
test.o: test.c
test: ${TEST_OBJS}
	$(CC) $(LDFLAGS) ${TEST_OBJS} -o $@ $(LDLIBS)
clean:
	rm -f resample.o test test.o
