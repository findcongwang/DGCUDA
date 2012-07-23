all: dgeuler

dgeuler: euler.cu 2dadvec_euler.cu 2dadvec_kernels_euler.cu 2dadvec_kernels_wrappers_euler.cu time_integrator_euler.cu ../quadrature.cu ../basis.cu 
	nvcc -arch=sm_13 euler.cu -o dgeuler

