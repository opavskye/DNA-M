CUDA_INSTALL_PATH ?= /usr/local/cuda

CPP := g++
CC := gcc
NVCC := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I./lib/ 
# ARCH
ARCH = -arch=sm_20
# Libraries
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcurand
# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(ARCH) 
NVCCFLAGS += $(LIB_CUDA) 
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcurand

default: dataReader

dataReader: dataReader.cu sequencer.cu
	$(NVCC) -o dataReader dataReader.cu $(NVCCFLAGS)

memtest: memtest.cu sequencer.cu
	$(NVCC) -o memtest memtest.cu $(NVCCFLAGS)

clean:
	rm -f dataReader memtest *~ *#
