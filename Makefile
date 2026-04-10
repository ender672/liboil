CXXFLAGS ?= -O2
CXXFLAGS += -Wall -pedantic -std=c++17
-include local.mk

OIL_OBJS = OilResample.o
ifneq ($(filter aarch64 arm64,$(shell uname -m)),)
OIL_OBJS += OilResampleNeon.o
else ifneq ($(filter x86_64,$(shell uname -m)),)
OIL_OBJS += OilResampleSse2.o OilResampleAvx2.o
endif

all: test benchmark
OilResampleSse2.o: OilResampleSse2.cpp OilResampleInternal.h
	$(CXX) $(CXXFLAGS) -msse2 -c -o $@ $<
OilResampleAvx2.o: OilResampleAvx2.cpp OilResampleInternal.h
	$(CXX) $(CXXFLAGS) -mavx2 -mfma -c -o $@ $<
OilResampleNeon.o: OilResampleNeon.cpp OilResample.h OilResampleInternal.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<
test: Test.cpp $(OIL_OBJS)
	$(CXX) $(CXXFLAGS) $(OIL_OBJS) Test.cpp -o $@ -lm
benchmark: Benchmark.cpp $(OIL_OBJS)
	$(CXX) $(CXXFLAGS) $(OIL_OBJS) Benchmark.cpp -o $@ $(LDFLAGS) -lpng -lm
clean:
	rm -rf test test.dSYM OilResample.o OilResampleSse2.o OilResampleAvx2.o OilResampleNeon.o benchmark
