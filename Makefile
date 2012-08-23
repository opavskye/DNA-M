# Copyright 2012 by Erik Opavsky
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

default: sequencer counter promoterMerger

sequencer: sequencerMain.cu sequencer.cu dataTransfer.cu maximums.c printFunctions.cu constants.h
	$(NVCC) -o sequencer sequencerMain.cu $(NVCCFLAGS)

counter: counterMain.cu counter.cu dataTransfer.cu maximums.c printFunctions.cu constants.h
	$(NVCC) -o counter counterMain.cu $(NVCCFLAGS)

promoterMerger: promoterMerger.c
	gcc -o promoterMerger promoterMerger.c

clean:
	rm -f sequencer counter promoterMerger *~ *#
