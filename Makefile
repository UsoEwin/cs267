
CC = g++
NVCC = nvcc

CFLAGS = -std=c++11 -O3
OMPFLAGS = -fopenmp
NVOMPFLAGS = -Xcompiler -fopenmp
# for k80
NVFLAGS = -O3 -arch=compute_37 -code=sm_37
cudasources = goCuda.cpp gpu.cu go.cpp
serialsources = goSerial.cpp serial.cpp go.cpp
serialtargets = serialgo
cudatargets = cudago

serial:
	$(CC) $(CFLAGS) -o $(serialtargets) $(serialsources)
clean:
	rm -f $(serialtargets) $(cudatargets) *.txt *.o
cuda:
	$(NVCC) $(NVFLAGS) -o $(cudatargets) $(cudasources)
hybrid:
	$(NVCC) $(NVFLAGS) $(NVOMPFLAGS) -o $(cudatargets) $(cudasources)
openmp:
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $(serialtargets) $(serialsources)