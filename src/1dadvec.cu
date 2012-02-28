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
void timeIntegrate(float *u, float a, int K, float dt, float dx, double t, int Np) {
    int size = K * (Np + 1);

    int nThreads = 128;

    int nBlocksRHS   = K / nThreads + ((K % nThreads) ? 1 : 0);
    int nBlocksFlux  = (K + 1) / nThreads + (((K + 1) % nThreads) ? 1 : 0);
    int nBlocksRK    = ((Np + 1)*K) / nThreads + ((((Np + 1)* K) % nThreads) ? 1 : 0);

    checkCudaError("error before rk4");
    // Stage 1
    // f <- flux(u)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_u, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k1 <- dt*rhs(u)
    rhs<<<nBlocksRHS, nThreads>>>(d_u, d_k1, d_f, d_w, d_r, a, dt, dx, K, Np);
    cudaThreadSynchronize();
    // k* <- u + k1/2
    rk4_tempstorage<<<nBlocksRK, nThreads>>>(d_u, d_kstar, d_k1, 0.5, dt, Np, K);
    cudaThreadSynchronize();

    // Stage 2
    // f <- flux(k*)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_kstar, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k2 <- dt*rhs(k*)
    rhs<<<nBlocksRHS, nThreads>>>(d_kstar, d_k2, d_f, d_w, d_r, a, dt, dx, K, Np);
    cudaThreadSynchronize();
    // k* <- u + k2/2
    rk4_tempstorage<<<nBlocksRK, nThreads>>>(d_u, d_kstar, d_k2, 0.5, dt, Np, K);
    cudaThreadSynchronize();

    // Stage 3
    // f <- flux(k*)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_kstar, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k3 <- dt*rhs(k*)
    rhs<<<nBlocksRHS, nThreads>>>(d_kstar, d_k3, d_f, d_w, d_r, a, dt, dx, K, Np);
    cudaThreadSynchronize();
    // k* <- u + k3
    rk4_tempstorage<<<nBlocksRK, nThreads>>>(d_u, d_kstar, d_k3, 1.0, dt, Np, K);
    cudaThreadSynchronize();

    // Stage 4
    // f <- flux(k*)
    calcFlux<<<nBlocksFlux, nThreads>>>(d_kstar, d_f, a, t, K, Np);
    cudaThreadSynchronize();
    // k4 <- dt*rhs(k*)
    rhs<<<nBlocksRHS, nThreads>>>(d_kstar, d_k4, d_f, d_w, d_r, a, dt, dx, K, Np);
    cudaThreadSynchronize();

    checkCudaError("error after rk4");

    rk4<<<nBlocksRK, nThreads>>>(d_u, d_k1, d_k2, d_k3, d_k4, Np, K);

    cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
}

/* allocate memory on the GPU
 */
void initGPU(int K, int Np) {
    int size = K * (Np + 1);
    cudaDeviceReset();
    checkCudaError("error after reset?");

    // Main variables
    cudaMalloc((void **) &d_u , size * sizeof(float));
    cudaMalloc((void **) &d_f,  (K + 1) * sizeof(float));
    cudaMalloc((void **) &d_rx, size * sizeof(float));
    cudaMalloc((void **) &d_mesh, K * sizeof(float));
    cudaMalloc((void **) &d_r, (Np + 1) * sizeof(float));
    cudaMalloc((void **) &d_w, (Np + 1) * sizeof(float));

    // Runge-Kutta storage
    cudaMalloc((void **) &d_kstar , size * sizeof(float));
    cudaMalloc((void **) &d_k1 , size * sizeof(float));
    cudaMalloc((void **) &d_k2 , size * sizeof(float));
    cudaMalloc((void **) &d_k3 , size * sizeof(float));
    cudaMalloc((void **) &d_k4 , size * sizeof(float));

    checkCudaError("error in init");
}

void setIntegrationPoints(int Np, float *w, float *r) {
    switch (Np) {
        case 0:
            r[0] = 0.;
            w[0] = 2.;
            break;

        case 1:
            r[0] = -1./sqrt(3);
            r[1] =  1./sqrt(3);
            w[0] =  1.;
            w[1] =  1.;
            break;

        case 2:
            r[0] =  0.;
            r[1] = -sqrt(3./5);
            r[2] =  sqrt(3./5);
            w[0] =  8./9;
            w[1] =  5./9;
            w[2] =  5./9;
            break;

        case 3:
            r[0] = -sqrt((3.-2.*sqrt(6./5))/7.);
            r[1] =  sqrt((3.-2.*sqrt(6./5))/7.);
            r[2] = -sqrt((3.+2.*sqrt(6./5))/7.);
            r[3] =  sqrt((3.+2.*sqrt(6./5))/7.);
            w[0] =  (18.+sqrt(30.))/36.;
            w[1] =  (18.+sqrt(30.))/36.;
            w[2] =  (18.-sqrt(30.))/36.;
            w[3] =  (18.-sqrt(30.))/36.;
            break;

        case 4:
            r[0] =  0.;
            r[1] = -sqrt(5.-2.*sqrt(10./7))/3.;
            r[2] =  sqrt(5.-2.*sqrt(10./7))/3.;
            r[3] = -sqrt(5.+2.*sqrt(10./7))/3.;
            r[4] =  sqrt(5.+2.*sqrt(10./7))/3.;
            w[0] =  128./225;
            w[1] =  (322.+13.*sqrt(70.))/900.;
            w[2] =  (322.+13.*sqrt(70.))/900.;
            w[3] =  (322.-13.*sqrt(70.))/900.;
            w[4] =  (322.-13.*sqrt(70.))/900.;
            break;

        case 5:
            r[0] = -0.23861918;
            r[1] =  0.23861918;
            r[2] = -0.66120939;
            r[3] =  0.66120939;
            r[4] = -0.93246951;
            r[5] =  0.93246951;
            w[0] =  0.46791393;
            w[1] =  0.46791393;
            w[2] =  0.36076157;
            w[3] =  0.36076157;
            w[4] =  0.17132449;
            w[5] =  0.17132449;
            break;

        case 6:
            r[0] =  0;
            r[1] = -0.40584515;
            r[2] =  0.40584515;
            r[3] = -0.74153119;
            r[4] =  0.74153119;
            r[5] = -0.94910791;
            r[6] =  0.94910791;
            w[0] =  0.41795918;
            w[1] =  0.38183005;
            w[2] =  0.38183005;
            w[3] =  0.27970539;
            w[4] =  0.27970539;
            w[5] =  0.12948497;
            w[6] =  0.12948497;
            break;

        case 7:
            r[0] = -0.18343464;
            r[1] =  0.18343464;
            r[2] = -0.52553241;
            r[3] =  0.52553241;
            r[4] = -0.79666648;
            r[5] =  0.79666648;
            r[6] = -0.96028986;
            r[7] =  0.96028986;
            w[0] =  0.36268378;
            w[1] =  0.36268378;
            w[2] =  0.31370665;
            w[3] =  0.31370665;
            w[4] =  0.22238103;
            w[5] =  0.22238103;
            w[6] =  0.10122854;
            w[7] =  0.10122854;
            break;

        // This is WRONG
        case 8:
            r[0] = -0.14887434;
            r[1] =  0.14887434;
            r[2] = -0.43339539;
            r[3] =  0.43339539;
            r[4] = -0.67940957;
            r[5] =  0.67940957;
            r[6] = -0.86506337;
            r[7] =  0.86506337;
            r[8] = -0.97390653;
            r[9] =  0.97390653;
            w[0] =  0.29552422;
            w[1] =  0.29552422;
            w[2] =  0.26926672;
            w[3] =  0.26926672;
            w[4] =  0.21908636;
            w[5] =  0.21908636;
            w[6] =  0.14945135;
            w[7] =  0.14945135;
            w[8] =  0.06667134;
            w[9]=  0.06667134;
            break;

        // This might be ok.
        case 9:
            r[0] = -0.14887434;
            r[1] =  0.14887434;
            r[2] = -0.43339539;
            r[3] =  0.43339539;
            r[4] = -0.67940957;
            r[5] =  0.67940957;
            r[6] = -0.86506337;
            r[7] =  0.86506337;
            r[8] = -0.97390653;
            r[9]=  0.97390653;
            w[0] =  0.29552422;
            w[1] =  0.29552422;
            w[2] =  0.26926672;
            w[3] =  0.26926672;
            w[4] =  0.21908636;
            w[5] =  0.21908636;
            w[6] =  0.14945135;
            w[7] =  0.14945135;
            w[8] =  0.06667134;
            w[9]=  0.06667134;
            break;

        // This is also WRONG
        case 10:
            r[0] = -0.14887434;
            r[1] =  0.14887434;
            r[2] = -0.43339539;
            r[3] =  0.43339539;
            r[4] = -0.67940957;
            r[5] =  0.67940957;
            r[6] = -0.86506337;
            r[7] =  0.86506337;
            r[8] = -0.97390653;
            r[9]=  0.97390653;
            w[0] =  0.29552422;
            w[1] =  0.29552422;
            w[2] =  0.26926672;
            w[3] =  0.26926672;
            w[4] =  0.21908636;
            w[5] =  0.21908636;
            w[6] =  0.14945135;
            w[7] =  0.14945135;
            w[8] =  0.06667134;
            w[9]=  0.06667134;
            break;
    }
}

int main() {
    int i, j, size, t, timesteps;
    float *mesh;
    float *u;     // the computed result
    float *r;     // the GLL points
    float *w;     // Gaussian integration weights
    
    int Np  = 7;              // polynomial order of the approximation
    int K   = 2*10;           // the mesh size
    float a = -1.;              // left boundary
    float b = 1.;      // right boundary
    float dx = (b - a) / K;    // size of cell
    float aspeed = 2.*3.14159; // the wave speed

    float CFL = 1. / (2.*Np + 1.);
    float dt  = 0.5 * CFL/aspeed * dx; // timestep
    timesteps = 1000;

    size = (Np + 1) * K;  // size of u

    mesh = (float *) malloc(K * sizeof(float));
    u = (float *) malloc(K * (Np + 1) * sizeof(float));
    r = (float *) malloc((Np + 1) * sizeof(float));
    w = (float *) malloc((Np + 1) * sizeof(float));

    int nThreads    = 128;
    int nBlocksMesh = (K + 1) / nThreads + (((K + 1) % nThreads) ? 1 : 0);
    int nBlocksU    = K / nThreads + ((K % nThreads) ? 1 : 0);

    // Allocate space on the GPU
    initGPU(K, Np);

    // Init the mesh's endpoints
    initMesh<<<nBlocksMesh, nThreads>>>(d_mesh, dx, a, K);
    cudaThreadSynchronize();

    // Copy over r and w
    setIntegrationPoints(Np, w, r);
    cudaMemcpy(d_r, r, (Np + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, (Np + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize u0
    initU<<<nBlocksU, nThreads>>>(d_u, d_mesh, d_w, d_r, dx, K, Np);
    cudaThreadSynchronize();

    checkCudaError("error after initialization");
    // File for output
    FILE *data;
    data = fopen("data.txt", "w");

    cudaMemcpy(mesh, d_mesh, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);

    fprintf(data, "%i\n", Np);
    for (i = 0; i < K; i++) {
        fprintf(data, "%f ", mesh[i]);
    }
    fprintf(data, "\n");

    // Run the integrator 
    for (t = 0; t < timesteps; t++) {
        for (i = 0; i < K; i++) {
            for (j = 0; j < Np+1; j++) {
                fprintf(data," %f ", u[j*K + i]);
            }
        }
        fprintf(data, "\n");
        timeIntegrate(u, aspeed, K, dt, dx, dt*t, Np);
    }
    //fclose(data);

    // Free host data
    free(mesh);
    free(u);
    free(r);
    free(w);

    // Free GPU data
    cudaFree(d_u);
    cudaFree(d_f);
    cudaFree(d_rx);
    cudaFree(d_mesh);
    cudaFree(d_r);

    cudaFree(d_kstar);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);
}
