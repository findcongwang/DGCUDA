#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "1dadvec_kernels.cu"

void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

/* integrate in time
 *
 * take one time step; calls the kernel functions to compute in parallel.
 */
void timeIntegrate(float *u, float a, int K, float dt, double t, int Np) {
    int size = K * Np;

    int nThreads = 128;

    int nBlocksRHS   = K / nThreads + ((K % nThreads) ? 1 : 0);
    int nBlocksFlux  = (K + 1) / nThreads + (((K + 1) % nThreads) ? 1 : 0);
    int nBlocksRK    = (Np*K) / nThreads + (((Np* K) % nThreads) ? 1 : 0);

    // Stage 1
    // f <- flux(u)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_u, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k1 <- dt*rhs(u)
    rhs<<<nBlocksRHS, nThreads>>>(d_u, d_k1, d_f, d_Dr, d_rx, a, dt, K, Np);
    cudaThreadSynchronize();
    // k* <- u + k1/2
    rk4_tempstorage<<<nBlocksRK, nThreads>>>(d_u, d_kstar, d_k1, 0.5, dt, Np, K);
    cudaThreadSynchronize();

    // Stage 2
    // f <- flux(k*)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_kstar, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k2 <- dt*rhs(k*)
    rhs<<<nBlocksRHS, nThreads>>>(d_kstar, d_k2, d_f, d_Dr, d_rx, a, dt, K, Np);
    cudaThreadSynchronize();
    // k* <- u + k2/2
    rk4_tempstorage<<<nBlocksRK, nThreads>>>(d_u, d_kstar, d_k2, 0.5, dt, Np, K);
    cudaThreadSynchronize();

    // Stage 3
    // f <- flux(k*)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_kstar, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k3 <- dt*rhs(k*)
    rhs<<<nBlocksRHS, nThreads>>>(d_kstar, d_k3, d_f, d_Dr, d_rx, a, dt, K, Np);
    cudaThreadSynchronize();
    // k* <- u + k3
    rk4_tempstorage<<<nBlocksRK, nThreads>>>(d_u, d_kstar, d_k3, 1, dt, Np, K);
    cudaThreadSynchronize();

    // Stage 4
    // f <- flux(k*)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_kstar, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k4 <- dt*rhs(k*)
    rhs<<<nBlocksRHS, nThreads>>>(d_kstar, d_k4, d_f, d_Dr, d_rx, a, dt, K, Np);
    cudaThreadSynchronize();

    checkCudaError("error after rk4");

    rk4<<<nBlocksRK, nThreads>>>(d_u, d_k1, d_k2, d_k3, d_k4, Np, K);
    cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
}

/* returns the Dr matrix
 */
void setDr(float *Dr) {
    //Dr[0] = -0.5;
    //Dr[1] =  0.5;
    //Dr[2] = -0.5;
    //Dr[3] =  0.5;

    // This is for Np == 3
    Dr[0] = -1.50;
    Dr[1] = 2.00;
    Dr[2] = -0.50;
    Dr[3] = -0.50;
    Dr[4] = 0.00;
    Dr[5] = 0.50;
    Dr[6] = 0.50;
    Dr[7] = -2.00;
    Dr[8] = 1.50;
}

/* allocate memory on the GPU
 */
void initGPU(int K, int Np) {
    int size = K * Np;
    cudaDeviceReset();
    checkCudaError("error after reset?");

    // Main variables
    cudaMalloc((void **) &d_u , size * sizeof(float));
    cudaMalloc((void **) &d_Dr, Np * Np * sizeof(float));
    cudaMalloc((void **) &d_f,  (K + 1) * sizeof(float));
    cudaMalloc((void **) &d_rx, size * sizeof(float));
    cudaMalloc((void **) &d_mesh, K * sizeof(float));
    cudaMalloc((void **) &d_x, size * sizeof(float));
    cudaMalloc((void **) &d_r, Np * sizeof(float));

    // Runge-Kutta storage
    cudaMalloc((void **) &d_kstar , size * sizeof(float));
    cudaMalloc((void **) &d_k1 , size * sizeof(float));
    cudaMalloc((void **) &d_k2 , size * sizeof(float));
    cudaMalloc((void **) &d_k3 , size * sizeof(float));
    cudaMalloc((void **) &d_k4 , size * sizeof(float));

    checkCudaError("error in init");
}

int main() {
    int i, size, t, timesteps;
    float *Dr;    // diff matrix
    float *u;     // the computed result
    float *r;     // the GLL points
    
    int Np  = 3;              // polynomial order of the approximation
    int K   = 2*80;           // the mesh size
    float a = 0;              // left boundary
    float b = 2*3.14159;      // right boundary
    float h = (b - a) / K;    // size of cell
    float aspeed = 2*3.14159; // the wave speed

    float CFL = .75;  // CFL number (duh)
    float dt = 0.5* (CFL/aspeed * h); // timestep
    timesteps = 1000; 

    size = Np * K;  // size of u

    Dr    = (float *) malloc(Np * Np * sizeof(float));
    u     = (float *) malloc(K * Np * sizeof(float));
    r     = (float *) malloc(Np * sizeof(float));

    int nThreads    = 128;
    int nBlocksMesh = (K + 1) / nThreads + (((K + 1) % nThreads) ? 1 : 0);
    int nBlocksU    = size / nThreads + ((size % nThreads) ? 1 : 0);

    // Allocate space on the GPU
    initGPU(K, Np);

    // Init the mesh's endpoints
    initMesh<<<nBlocksMesh, nThreads>>>(d_mesh, d_x, h, a, K);
    cudaThreadSynchronize();

    // Copy over r
    r[0] = -1;
    r[1] = 0;
    r[2] = 1;
    cudaMemcpy(d_r, r, Np * sizeof(float), cudaMemcpyHostToDevice);

    // Init the mesh
    initX<<<nBlocksMesh, nThreads>>>(d_mesh, d_x, d_r, h, K, Np);
    cudaThreadSynchronize();

    // Initialize u0
    initU<<<nBlocksU, nThreads>>>(d_u, d_x, K, Np);
    cudaThreadSynchronize();

    // Set the Dr matrix and copy it over
    setDr(Dr);
    cudaMemcpy(d_Dr, Dr, Np * Np * sizeof(float), cudaMemcpyHostToDevice); 

    //  Calculate the rx mapping on the GPU
    int nBlocksRx = (K / nThreads) + ((K % nThreads) ? 1 : 0);
    initRx<<<nBlocksRx,nThreads>>>(d_rx, d_x, d_Dr, K, Np);
    cudaThreadSynchronize();

    float *rx = (float *)malloc(size*sizeof(float));
    cudaMemcpy(rx, d_rx, size*sizeof(float), cudaMemcpyDeviceToHost);
    for (i = 0; i < size; i++ ) {
        printf("%f, ", rx[i]);
    }
    free(rx);
    printf("\n");


    checkCudaError("error after initialization");
    // File for output
    FILE *data;
    data = fopen("data.txt", "w");

    // Run the integrator 
    for (t = 0; t < timesteps; t++) {
        timeIntegrate(u, aspeed, K, dt, dt*t, Np);
        for (i = 0; i < size; i++) {
            fprintf(data," %f ", u[i]);
        }
        fprintf(data, "\n");
    }
    fclose(data);

    // Free host data
    free(u);
    free(r);

    // Free GPU data
    cudaFree(d_u);
    cudaFree(d_Dr);
    cudaFree(d_f);
    cudaFree(d_rx);
    cudaFree(d_mesh);
    cudaFree(d_x);
    cudaFree(d_r);

    cudaFree(d_kstar);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);
}
