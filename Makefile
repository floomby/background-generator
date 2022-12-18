CL_LIB := -L "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64"
CL_INC := -I "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include"

CPP_FLAGS := -Wall -pedantic -std=gnu++23 -ggdb -fdiagnostics-color $(CL_INC)
LD_FLAGS := -lOpenCL $(CL_LIB)

target: main.o spng.dll
	g++ $(CPP_FLAGS) -o target main.o $(LD_FLAGS) spng.dll

main.o: main.cpp
	g++ $(CPP_FLAGS) -c main.cpp

spng.dll: libs/spng.c
	gcc -shared -o spng.dll libs/spng.c -lZ

clean:
	rm -f *.o target spng.dll