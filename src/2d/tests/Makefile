all: dgcudatest

dgcudatest: test_all.cu ../2dadvec.cu ../2dadvec_kernels.cu ../2dadvec_kernels_wrappers.cu ../quadrature.h ../basis.h
	nvcc test_all.cu -o dgcudatest

