CFLAGS += -O3 -march=native -Wall -pedantic

all: liboilresample.so.0 oiltest oilscalejpg oilscalepng
liboilresample.so.0: src/oil_resample.c
	$(CC) -shared -Wl,-soname,$@ $(CFLAGS) $^ -o $@
oiltest: src/test.c src/oil_resample.c
	$(CC) $(CFLAGS) src/test.c -o $@ -lm
oilscalejpg: liboilresample.so.0 src/jpgscale.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -ljpeg -lm
oilscalepng: liboilresample.so.0 src/pngscale.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lpng -lm
clean:
	rm -rf liboilresample.so.0 oiltest oiltest.dSYM oilscalejpg oilscalepng
