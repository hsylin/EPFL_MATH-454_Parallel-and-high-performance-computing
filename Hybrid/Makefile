# Makefile for intra-node hybrid MPI + CUDA LBM.
# Suggested SCITAS modules:
#   module purge
#   module load gcc cuda openmpi hdf5
#
# Useful builds:
#   make
#   make DEBUG=1
#   make CUDA_ARCH=sm_70

MPICXX ?= mpicxx
NVCC   ?= nvcc
CUDA_ARCH ?= sm_70
PKG_CONFIG ?= pkg-config

CUDA_HOME_AUTO := $(shell CUDA_NVCC=$$(command -v nvcc 2>/dev/null); if [ -n "$$CUDA_NVCC" ]; then dirname $$(dirname "$$CUDA_NVCC"); fi)
CUDA_HOME ?= $(CUDA_HOME_AUTO)

# OpenMPI-specific. MPICH/Intel MPI may need a different wrapper-info flag.
MPI_COMPILE := $(shell $(MPICXX) --showme:compile 2>/dev/null)

HDF5_PKG_CFLAGS := $(shell $(PKG_CONFIG) --cflags hdf5 2>/dev/null)
HDF5_PKG_LIBS   := $(shell $(PKG_CONFIG) --libs hdf5 2>/dev/null)
HDF5_INC  := $(if $(strip $(HDF5_ROOT)),-I$(HDF5_ROOT)/include,$(HDF5_PKG_CFLAGS))
HDF5_LIBS := $(if $(strip $(HDF5_ROOT)),-L$(HDF5_ROOT)/lib -lhdf5,$(if $(strip $(HDF5_PKG_LIBS)),$(HDF5_PKG_LIBS),-lhdf5))
CUDA_LIBS := $(if $(strip $(CUDA_HOME)),-L$(CUDA_HOME)/lib64,) -lcudart

COMMON_CXXFLAGS := -Wall -Wextra -std=c++11 $(HDF5_INC)
COMMON_NVCCFLAGS := -std=c++11 -arch=$(CUDA_ARCH) $(MPI_COMPILE) $(HDF5_INC) -Xcompiler "-Wall -Wextra"

ifeq ($(DEBUG),1)
CXXFLAGS  += -g -O0 -DLBM_CUDA_DEBUG $(COMMON_CXXFLAGS)
NVCCFLAGS += -g -G -O0 -DLBM_CUDA_DEBUG $(COMMON_NVCCFLAGS)
else
CXXFLAGS  += -O3 $(COMMON_CXXFLAGS)
NVCCFLAGS += -O3 $(COMMON_NVCCFLAGS)
endif

LDFLAGS += $(HDF5_LIBS) $(CUDA_LIBS) -lm
OBJS = main.o lbm.o output.o

all: lbm_hybrid_cuda_mpi

lbm_hybrid_cuda_mpi: $(OBJS)
	$(MPICXX) -o $@ $(OBJS) $(LDFLAGS)

main.o: main.cc lbm.hh output.hh
	$(MPICXX) $(CXXFLAGS) -c -o $@ $<

output.o: output.cc output.hh lbm.hh
	$(MPICXX) $(CXXFLAGS) -c -o $@ $<

lbm.o: lbm.cu lbm.hh
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f lbm_hybrid_cuda_mpi *.o *~

distclean: clean
	rm -rf out/ probe*.csv scaling/logs scaling/out scaling/results_hybrid.csv scaling/hybrid_scaling.md scaling/hybrid_scaling.html

.PHONY: all clean distclean
