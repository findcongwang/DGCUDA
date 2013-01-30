CC=nvcc
CFLAGS=-arch=sm_13

all: dgcylinder dgsupersonic dgthreepoint

dgcylinder: cylinder/cylinder.cu ../main.cu ../conserv.cu ../conserv_kernels.cu ../conserv_kernels_wrappers.cu ../time_integrator.cu ../../quadrature.cu ../../basis.cu 
	$(CC) $(CFLAGS) cylinder/cylinder.cu -o dgcylinder

dgsupersonic: supersonic/supersonic.cu ../main.cu ../conserv.cu ../conserv_kernels.cu ../conserv_kernels_wrappers.cu ../time_integrator.cu ../../quadrature.cu ../../basis.cu 
	$(CC) $(CFLAGS) supersonic/supersonic.cu -o dgsupersonic

dgthreepoint: threepoint/threepoint.cu ../main.cu ../conserv.cu ../conserv_kernels.cu ../conserv_kernels_wrappers.cu ../time_integrator.cu ../../quadrature.cu ../../basis.cu 
	$(CC) $(CFLAGS) threepoint/threepoint.cu -o dgthreepoint

