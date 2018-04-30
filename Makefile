
EXECUTABLE := cudago

CU_FILES   := goai.cu

CU_DEPS    :=

CC_FILES   := go_Decision.cpp

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS=$(OBJDIR)/go_Decision.o  $(OBJDIR)/goai.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) 

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@


nvcc -m64 -O3 -o cudaSearch objs/solve.o objs/readfile.o -arch=compute_37 -code=sm_37
nvcc -m64 -O3 -o cudago objs/go_Decision.o  objs/goai.o -arch=compute_37 -code=sm_37