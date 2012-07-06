all: dgcuda

dgcuda: 2dadvec.cu 2dadvec_kernels.cu 2dadvec_kernels_wrappers.cu quadrature.h basis.h
	nvcc 2dadvec.cu -o dgcuda

