all: dgeuler

dgeuler: main.cu euler.cu euler_kernels.cu euler_kernels_wrappers.cu time_integrator_euler.cu ../quadrature.cu ../basis.cu 
	nvcc -arch=sm_13 main.cu -o dgeuler

