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

default: sequencer

sequencer: sequencerMain.cu sequencer.cu dataTransfer.cu maximums.c printFunctions.cu constants.h
	$(NVCC) -o sequencer sequencerMain.cu $(NVCCFLAGS)

counter: counterMain.cu counter.cu dataTransfer.cu maximums.c printFunctions.cu constants.h
	$(NVCC) -o counter counterMain.cu $(NVCCFLAGS)

clean:
	rm -f sequencer counter memtest *~ *#
