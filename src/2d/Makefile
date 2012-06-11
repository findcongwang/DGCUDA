all: dgcuda

dgcuda: 2dadvec.cu 2dadvec_kernels.cu quadrature.h basis.h
	nvcc 2dadvec.cu -o dgcuda

