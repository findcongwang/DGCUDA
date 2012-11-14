all: dgeuler

dgeuler: main.cu euler.cu ../conserv.cu ../conserv_kernels.cu ../conserv_kernels_wrappers.cu ../time_integrator.cu ../../quadrature.cu ../../basis.cu 
	nvcc -arch=sm_13 main.cu -o dgeuler

