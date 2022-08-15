objects = main.o hll.o boundary.o reconstruct.o eos.o 

all: $(objects)
	nvcc $(objects) -o test

%.o : %.cu 
	nvcc -x cu -dc $< -o $@
#%.o : %.cu %.cuh
#	nvcc -x cu -I -dc $< -o $@

clean:
	rm -f *.o test
