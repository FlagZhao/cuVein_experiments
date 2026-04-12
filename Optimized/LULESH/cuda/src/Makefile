# MPICH_DIR = /usr/lib/x86_64-linux-gnu/openmpi
MPICH_DIR = /home/yzhao62/opt/gvprof/GVProf/gvprof/spack/opt/spack/linux-ubuntu22.04-zen3/gcc-12.1.0/openmpi-4.1.5-ntt6wucomlskgu43in3af5ewozishqop
# MPI_FLAGS = -DUSE_MPI -I$(MPICH_DIR)/include

MPI_FLAGS =

CUDA_DIR ?= /usr/local/cuda
# NVCC		= $(CUDA_DIR)/bin/nvcc -ccbin /usr/bin/cc
NVCC		= $(CUDA_DIR)/bin/nvcc -ccbin /usr/bin/gcc

FLAGS		= -arch=sm_89 -lineinfo
DFLAGS	= $(MPI_FLAGS) -lineinfo
RFLAGS 	= $(MPI_FLAGS) -O3 -DNDEBUG 

#SILO_INCLUDES := /usr/local/silo-4.8/include
#SILO_LIBS := /usr/local/silo-4.8/lib

LINKFLAGS = -lmpich -L$(MPICH_DIR)/lib -lmpi
#LINKFLAGS += -L$(SILO_LIBS) -lsilo

#INC_SILO:= -I$(SILO_INCLUDES)

all: release 

debug: LINKFLAGS += 

release: 	FLAGS += $(RFLAGS)
debug: 		FLAGS += $(DFLAGS)

release: lulesh
debug: lulesh

lulesh: allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o
	$(NVCC) $(LINKFLAGS) allocator.o lulesh.o lulesh-comms.o lulesh-comms-gpu.o -o lulesh

allocator.o: allocator.cu vector.h
	$(NVCC) $(FLAGS) allocator.cu -I ./ -c -o allocator.o

lulesh.o: lulesh.cu util.h vector.h allocator.h
	$(NVCC) $(FLAGS) lulesh.cu -I ./  $(INC_SILO) -c -o lulesh.o

lulesh-comms.o: lulesh-comms.cu
	$(NVCC) $(FLAGS) lulesh-comms.cu -I ./ -c -o lulesh-comms.o

lulesh-comms-gpu.o: lulesh-comms-gpu.cu
	$(NVCC) $(FLAGS) lulesh-comms-gpu.cu -I ./ -c -o lulesh-comms-gpu.o

clean: 
	rm -rf allocator.o lulesh-comms.o lulesh-comms-gpu.o lulesh.o lulesh xyz.asc regNumList.txt