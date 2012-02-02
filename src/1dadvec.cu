#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Important variables for GPU shit
float *d_u;
float *d_f;
float *d_rx;
float *d_Dr;

// Runge-Kutta time integration storage
float *d_kstar;
float *d_k1;
float *d_k2;
float *d_k3;
float *d_k4;

void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

/* flux calculations for each node 
 *
 *  | endpoint - f0 - f1 -  ... - fm-1 - endpoint |
 *
 * That is, fi is the flux between nodes i and i+1,
 * making f a m-1 length vector.
 * Store results into f
 */
__global__ void calcFlux(float *u, float *f, int Np, float aspeed, float time, int K) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;
    float ul, ur;

    if (idx == 0) {
        f[idx] = 0;//-sin(aspeed * time);
    }
    if (idx > 0) {
        // Flux calculations
        ul = u[idx*Np - 1];
        ur = u[idx*Np];

        // Central flux
        f[idx] = (ul - ur) / 2;
    }
    if (idx == K) {
        f[idx] = 0;
    }
}

/* right hand side calculations
 *
 * Calculates the volume and boundary contributions and adds them to the rhs
 * Gets called during time integration
 * Store results into u
 */
__global__ void rhs(float *u, float *k, float *f, float *Dr, float *rx, float a, int Np, float dt) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;
    int i,j;
    float rhs[2], r_u[2];
    float lflux, rflux;

    for (i = 0; i < Np; i++) {
        r_u[i] = u[Np*idx + i];
    }

    // Initialize rhs to 0
    for (i = 0; i < Np; i++) {
        rhs[i] = 0;
    }

    // Calculate Dr * u
    // I think these are wrong
    //for (i = 0; i < Np; i++) {
        //for (j = 0; j < Np; j++) {
            //rhs[i] += Dr[i*Np + j] * u[Np*idx + j];
        //}
    //}
    rhs[0] = Dr[0] * r_u[0] + Dr[1] * r_u[1];
    rhs[1] = Dr[2] * r_u[0] + Dr[3] * r_u[1];
    // And this is OK
    //for (i = 0; i < Np; i++) {
        //for (j = 0; j < Np; j++) {
            //rhs[i] += Dr[i*Np + j] * r_u[j];
        //}
    //}

    lflux  = f[idx];
    rflux  = f[idx+1];

    // Scale RHS up
    for (i = 0; i < Np; i++) {
        rhs[i] *= -a*rx[Np*idx + i];
    }

    // LIFT
    rhs[0]    -= lflux;
    rhs[Np-1] += rflux;

    // Store result
    for (i = 0; i < Np; i++) {
        k[Np*idx + i] = dt * rhs[i];
    }
}

/* tempstorage for RK4
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
__global__ void rk4_tempstorage(float *u, float *kstar, float*k, float alpha) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;
    kstar[idx] = u[idx] + alpha * k[idx];
}

/* rk4
 *
 * computes the runge-kutta solution 
 * u_n+1 = u_n + k1/6 + k2/3 + k3/3 + k4/6
 */
__global__ void rk4(float *u, float *k1, float *k2, float *k3, float *k4) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;

    u[idx] += k1[idx]/6 + k2[idx]/3 + k3[idx]/3 + k4[idx]/6;
}


/* integrate in time
 *
 * take one time step; calls the kernel functions to compute in parallel.
 */
void timeIntegrate(float *u, float a, int K, float dt, int Np, double t) {
    int size = K * Np;

    dim3 nBlocks      = dim3(1);
    dim3 nThreadsRHS  = dim3(K);
    dim3 nThreadsFlux = dim3(K+1);
    dim3 nThreadsRK   = dim3(size);

    // Stage 1
    // f <- flux(u)
    calcFlux<<<nBlocks, nThreadsFlux>>>(d_u, d_f, Np, a, t, K);
    cudaThreadSynchronize();
    // k1 <- dt*rhs(u)
    rhs<<<nBlocks, nThreadsRHS>>>(d_u, d_k1, d_f, d_Dr, d_rx, a, Np, dt);
    cudaThreadSynchronize();
    // k* <- u + k1/2
    rk4_tempstorage<<<nBlocks, nThreadsRK>>>(d_u, d_kstar, d_k1, 0.5);
    cudaThreadSynchronize();

    // Stage 2
    // f <- flux(k*)
    calcFlux<<<nBlocks, nThreadsFlux>>>(d_kstar, d_f, Np, a, t, K);
    cudaThreadSynchronize();
    // k2 <- dt*rhs(k*)
    rhs<<<nBlocks, nThreadsRHS>>>(d_kstar, d_k2, d_f, d_Dr, d_rx, a, Np, dt);
    cudaThreadSynchronize();
    // k* <- u + k2/2
    rk4_tempstorage<<<nBlocks, nThreadsRK>>>(d_u, d_kstar, d_k2, 0.5);
    cudaThreadSynchronize();

    // Stage 3
    // f <- flux(k*)
    calcFlux<<<nBlocks, nThreadsFlux>>>(d_kstar, d_f, Np, a, t, K);
    cudaThreadSynchronize();
    // k3 <- dt*rhs(k*)
    rhs<<<nBlocks, nThreadsRHS>>>(d_kstar, d_k3, d_f, d_Dr, d_rx, a, Np, dt);
    cudaThreadSynchronize();
    // k* <- u + k3
    rk4_tempstorage<<<nBlocks, nThreadsRK>>>(d_u, d_kstar, d_k3, 1);
    cudaThreadSynchronize();

    // Stage 4
    // f <- flux(k*)
    calcFlux<<<nBlocks, nThreadsFlux>>>(d_kstar, d_f, Np, a, t, K);
    cudaThreadSynchronize();
    // k4 <- dt*rhs(k*)
    rhs<<<nBlocks, nThreadsRHS>>>(d_kstar, d_k4, d_f, d_Dr, d_rx, a, Np, dt);
    cudaThreadSynchronize();

    checkCudaError("error after rk4");

    rk4<<<nBlocks, nThreadsRK>>>(d_u, d_k1, d_k2, d_k3, d_k4);
    cudaMemcpy(u, d_u, size * sizeof(float), cudaMemcpyDeviceToHost);
}

/* returns the Dr matrix
 */
void setDr(float *Dr, int Np) {
    Dr[0] = -0.5;
    Dr[1] =  0.5;
    Dr[2] = -0.5;
    Dr[3] =  0.5;
}

/* allocate memory on the GPU
 */
void initGPU(int Np, int K) {
    int size = K * Np;
    cudaDeviceReset();
    checkCudaError("error after reset?");
    // Main variables
    cudaMalloc((void **) &d_u , size * sizeof(float));
    cudaMalloc((void **) &d_Dr, Np * Np * sizeof(float));
    cudaMalloc((void **) &d_f,  (K + 1) * sizeof(float));
    cudaMalloc((void **) &d_rx, size * sizeof(float));

    // Runge-Kutta storage
    cudaMalloc((void **) &d_kstar , size * sizeof(float));
    cudaMalloc((void **) &d_k1 , size * sizeof(float));
    cudaMalloc((void **) &d_k2 , size * sizeof(float));
    cudaMalloc((void **) &d_k3 , size * sizeof(float));
    cudaMalloc((void **) &d_k4 , size * sizeof(float));
    checkCudaError("error in init");
}

int main() {
    int i, j, k, size, t, timesteps;
    float *Dr;    // diff matrix
    float *u;     // the computed result
    float *f;     // the flux
    float *mesh;  // the mesh's endpoints
    float *x;     // the mesh
    float *rx;    // the mapping
    float aspeed = 2*3.14159;      // the wave speed
    
    float K = 2*50; // the mesh size
    float a = 0;  // left boundary
    float b = 1;  // right boundary
    float h = (b - a) / K; // size of cell

    float CFL = .75;
    float dt = 0.5* (CFL/aspeed * h);
    timesteps = 1000;

    int Np = 2;  // polynomial order

    size = Np * K;  // size of u

    Dr    = (float *) malloc(Np * Np * sizeof(float));
    u     = (float *) malloc(K * Np * sizeof(float));
    mesh  = (float *) malloc((K + 1) * sizeof(float));
    x     = (float *) malloc(K * Np * sizeof(float));
    rx    = (float *) malloc(K * Np * sizeof(float));
    f     = (float *) malloc((K + 1) * sizeof(float));

    // Init the mesh's endpoints
    for (i = 0; i < K + 1; i++) {
        mesh[i]   = a + h * i;
    }

    // Init the mesh
    for (i = 0; i < K + 1; i++) {
        x[Np*i] = mesh[i];
        x[Np*i + 1] = mesh[i+ 1];
    }

    //printf("x = ");
    //for (i = 0; i < Np * K; i++) {
        //printf("%f, ", x[i]);
    //}
    //printf("\n");

    // Initialize u0
    for (i = 0; i < Np*K; i++) {
        u[i] = sin(x[i]);
        printf("%f ", u[i]);
    }
    printf("\n");

    // set Dr
    setDr(Dr, Np);

    // compute rx
    //for (i = 0; i < Np; i++) {
        //for (j = 0; j < Np; j++) {
            //rhs[i] += Dr[i*Np + j] * u[Np*idx + j];
        //}
    //}
    for (i = 0; i < Np; i++) {
        for (j = 0; j < K; j++) {
            for (k = 0; k < Np; k++) {        
                // rx[i,j] += Dr[i,k] * x[k,j]
                rx[Np*i + j] += Dr[i*Np + k] * x[k*Np + j];
                //Dr[j*p + k] * x[i*p + k
                ////rx[i] = 1 / 
            }
        }
    }

    //printf("rx = ");
    for (i = 0; i < Np * K; i++) {
        rx[i] = 20;
        //printf("%f, ", rx[i]);
    }
    //printf("\n");

    initGPU(Np, K);
    cudaMemcpy(d_u,  u,  size * sizeof(float) , cudaMemcpyHostToDevice); 
    cudaMemcpy(d_Dr, Dr, Np * Np * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_rx, rx, size * sizeof(float) , cudaMemcpyHostToDevice); 
    checkCudaError("error in memcpy");

    for (t = 0; t < timesteps; t++) {
        //printf("u%i =", t);
        timeIntegrate(u, aspeed, K, dt, Np, dt*t);
        for (i = 0; i < size; i++) {
            printf(" %f ", u[i]);
        }
        printf("\n");
    }

    free(u);
    free(x);
    free(f);
    free(Dr);
    free(mesh);
    free(rx);

    cudaFree(d_u);
    cudaFree(d_Dr);
    cudaFree(d_f);
    cudaFree(d_rx);

    cudaFree(d_kstar);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);
}
