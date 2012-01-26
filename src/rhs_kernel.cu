/*
 * rhs_kernel.cu
 *
 * Computes the right hand side of the 1D linear advection problem.
 *  rhs = -(a * rx) .* (Dr * u) + LIFT * (Fscale .* du)
 *
 * Input:
 *
 * Output:
 */
__global__ void rhs(int N, int *rx_global) {
    int idx = blockIdx.x + threadIdx.x;
    int i;
    double rx[N];

    __shared__ double Dr[N][N];
    __shared__ double LIFT

    /* read in the global data rx */
    for (i = 0; i < N; i++) {
        rx[i] = rx_global[idx*N + i]
    }
}

