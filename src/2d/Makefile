all: dgcuda
tests: runtests

dgcuda: main.cu 2dadvec.cu 2dadvec_kernels.cu 2dadvec_kernels_wrappers.cu quadrature.h basis.h basis_eval.cu
	nvcc main.cu -o dgcuda

runtests: test_all.cu 2dadvec.cu 2dadvec_kernels.cu 2dadvec_kernels_wrappers.cu quadrature.h basis.h basis_eval.cu
	nvcc test_all.cu -o runtests

