# Optimized CUDA LBM Makefile.
# Suggested SCITAS modules:
#   module load gcc cuda hdf5
# HDF5_ROOT may be left empty if your environment/compiler already knows HDF5 paths.
#
# Useful builds:
#   make                       # optimized run build
#   make DEBUG=1               # debug build; synchronizes after each kernel
#   make CUDA_ARCH=sm_80       # override GPU architecture

CXX      ?= g++
NVCC     ?= nvcc
LD       := $(NVCC)

CUDA_ARCH ?= sm_70

PKG_CONFIG ?= pkg-config

HDF5_PKG_CFLAGS := $(shell $(PKG_CONFIG) --cflags hdf5 2>/dev/null)
HDF5_PKG_LIBS   := $(shell $(PKG_CONFIG) --libs hdf5 2>/dev/null)

HDF5_INC  := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include,$(HDF5_PKG_CFLAGS))
HDF5_LIBS := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib -lhdf5,$(if $(strip $(HDF5_PKG_LIBS)),$(HDF5_PKG_LIBS),-lhdf5))

COMMON_CXXFLAGS := -Wall -Wextra -std=c++11 $(HDF5_INC)
COMMON_NVCCFLAGS := -std=c++11 -arch=$(CUDA_ARCH) $(HDF5_INC) -Xcompiler "-Wall -Wextra"

ifeq ($(DEBUG),1)
CXXFLAGS  += -g -O0 -DLBM_CUDA_DEBUG $(COMMON_CXXFLAGS)
NVCCFLAGS += -g -G -O0 -DLBM_CUDA_DEBUG $(COMMON_NVCCFLAGS)
else
CXXFLAGS  += -O3 $(COMMON_CXXFLAGS)
NVCCFLAGS += -O3 $(COMMON_NVCCFLAGS)
endif

LDFLAGS   += $(HDF5_LIBS) -lm

OBJS = main.o lbm.o output.o

all: lbm_cuda_opt

lbm_cuda_opt: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

main.o: main.cc lbm.hh output.hh
	$(CXX) $(CXXFLAGS) -c -o $@ $<

output.o: output.cc output.hh lbm.hh
	$(CXX) $(CXXFLAGS) -c -o $@ $<

lbm.o: lbm.cu lbm.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f lbm_cuda_opt *.o *~

distclean: clean
	rm -rf out/ probe.csv probe_* results_cuda_optimized.csv *.gif logs/

.PHONY: all clean distclean
