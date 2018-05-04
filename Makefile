
CC = g++
NVCC = nvcc

CFLAGS = -std=c++11 -fopenmp
# for k80
NVFLAGS = -O3 -arch=compute_37 -code=sm_37
#nvcc -O3  -o cudago goCuda.cpp gpu.cu go.cpp -arch=compute_37 -code=sm_37
#g++ -std=c++11 -fopenmp -O3 goSerial.cpp serial.cpp go.cpp -o go
cudasources = goCuda.cpp gpu.cu go.cpp
serialsources = goSerial.cpp serial.cpp go.cpp
serialtargets = serialgo
cudatargets = cudago

serial:
	$(CC) $(CFLAGS) -o $(serialtargets) $(serialsources)
clean:
	rm -f $(serialtargets) $(cudatargets) *.txt *.o
cuda:
	$(NVCC) $(NVFLAGS) -o $(cudasources)