CFLAGS += -Os -Wall -pedantic -I/usr/local/include
LDLIBS += -lpng -ljpeg -lgif -lm
LDFLAGS += -L/usr/local/lib

oil: oil.o resample.o yscaler.o quant.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)
oil.o: oil.c quant.h resample.h yscaler.h
yscaler.o: yscaler.c yscaler.h
resample.o: resample.c resample.h
quant.o: quant.c quant.h
test.o: test.c
test: test.o resample.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)
clean:
	rm -f oil oil.o yscaler.o resample.o test test.o quant.o

# rscope testing
test-rscope: oil
	rscope -gen
	rscope -gen -r
	./oil rpng pl.png | ./oil scalex 555 | ./oil wpng > pl_out.png && rscope -pl pl_out.png pl_out_results.png
	./oil rpng pd.png | ./oil scalex 555 | ./oil wpng > pd_out.png && rscope -pd pd_out.png pd_out_results.png
	./oil rpng plr.png | ./oil scaley 555 | ./oil wpng > plr_out.png && rscope -pl -r plr_out.png plr_out_results.png
	./oil rpng pdr.png | ./oil scaley 555 | ./oil wpng > pdr_out.png && rscope -pd -r pdr_out.png pdr_out_results.png
	rm -f pl.png pd.png pl_out.png pd_out.png plr.png pdr.png plr_out.png pdr_out.png rscope.html rscoper.html

# pngsuite testing
PngSuite-2013jan13.tgz:
	wget http://www.schaik.com/pngsuite/$@
pngsuite: PngSuite-2013jan13.tgz
	mkdir -p $@
	tar xzf PngSuite-2013jan13.tgz -C $@
test-pngsuite: pngsuite oil
	for f in pngsuite/*.png; do echo $$f; ./oil rpng $$f > /dev/null || true; done

# imagetestsuite png testing
imagetestsuite-png-1.01.tar.gz:
	wget https://imagetestsuite.googlecode.com/files/$@ --no-check-certificate
imagetestsuite/png: imagetestsuite-png-1.01.tar.gz
	mkdir -p imagetestsuite
	tar xzf imagetestsuite-png-1.01.tar.gz -C imagetestsuite
test-imagetestsuite-png: imagetestsuite/png oil
	for f in imagetestsuite/png/*.png; do echo $$f; ./oil rpng $$f > /dev/null || true; done

# imagetestsuite jpg testing
imagetestsuite-jpg-1.00.tar.gz:
	wget https://imagetestsuite.googlecode.com/files/$@ --no-check-certificate
imagetestsuite/jpg: imagetestsuite-jpg-1.00.tar.gz
	tar xzf imagetestsuite-jpg-1.00.tar.gz
test-imagetestsuite-jpg: imagetestsuite/jpg oil
	for f in imagetestsuite/jpg/*.jpg; do echo $$f; ./oil rjpeg $$f > /dev/null || true; done

# imagetestsuite gif testing
imagetestsuite-gif-1.00.tar.gz:
	wget https://imagetestsuite.googlecode.com/files/$@ --no-check-certificate
imagetestsuite/gif: imagetestsuite-gif-1.00.tar.gz
	tar xzf imagetestsuite-gif-1.00.tar.gz
test-imagetestsuite-gif: imagetestsuite/gif oil
	for f in imagetestsuite/gif/*.gif; do echo $$f; ./oil rgif $$f > /dev/null || true; done
