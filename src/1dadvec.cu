#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Important variables for GPU shit
float *d_u;
float *d_f;
float *d_rx;
float *d_Dr;

// Runge-Kutta time integration storage
float *k1;
float *k2;
float *k3;
float *k4;

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
__global__ void calcFlux(float *u, float *f, int Np) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;
    float ul, ur;

    // Flux calculations
    ul = u[(idx + 1) * Np - 1];
    ur = u[(idx + 1) * Np];

    // Central flux
    f[idx] = (ul + ur) / 2;
}

/* right hand side calculations
 *
 * Calculates the volume and boundary contributions and adds them to the rhs
 * Gets called during time integration
 * Store results into u
 */
__global__ void rhs(float *u, float *f, float *Dr, float *rx, float a, int Np) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;
    int i,j;
    float rhs[2], flux;

    // Initialize rhs to 0
    for (i = 0; i < Np; i++) {
        rhs[i] = 0;
    }

    // Calculate Dr * u
    for (i = 0; i < Np; i++) {
        for (j = 0; j < Np; j++) {
            rhs[i] += Dr[i*Np + j] * u[Np*idx + j];
        }
    }

    flux  = f[idx];

    // Scale RHS up
    for (i = 0; i < Np; i++) {
        rhs[i] *= -a*rx[Np*idx + i];
    }

    // LIFT
    rhs[0]   -= flux;
    rhs[Np-1] += flux;

    // Store result
    for (i = 0; i < Np; i++) {
        u[Np*idx + i] = rhs[i];
    }
}

__global__ void rk4(float *u, float *k, float *k_current, float dt, float alpha) {
    int idx = gridDim.x * blockIdx.x + threadIdx.x;

    k_current[idx] = dt*(u[idx] + alpha * k[idx]);
}

void timeIntegrate(float *&u, float a, int K, int h, int Np) {
    int i;
    float *l_k1, *l_k2, *l_k3, *l_k4; 
    int size = K * Np;

    l_k1 = (float *) malloc(size * sizeof(float));
    l_k2 = (float *) malloc(size * sizeof(float));
    l_k3 = (float *) malloc(size * sizeof(float));
    l_k4 = (float *) malloc(size * sizeof(float));

    dim3 nBlocks      = dim3(1);
    dim3 nThreads     = dim3(size);
    dim3 nThreadsFlux = dim3(K-1);

    // Stage 1
    calcFlux<<<nBlocks, nThreadsFlux>>>(d_u, d_f, Np);       cudaThreadSynchronize();
    rhs<<<nBlocks, nThreads>>>(d_u, d_f, d_Dr, d_rx, a, Np); cudaThreadSynchronize();
    rk4<<<nBlocks, nThreads>>>(d_u, k1, d_u, h, 0);          cudaThreadSynchronize();
    checkCudaError("Stage 1");
    // Stage 2
    calcFlux<<<nBlocks, nThreadsFlux>>>(k1, d_f, Np);       cudaThreadSynchronize();
    rhs<<<nBlocks, nThreads>>>(k1, d_f, d_Dr, d_rx, a, Np); cudaThreadSynchronize();
    rk4<<<nBlocks, nThreads>>>(d_u, k2, k1,  h, 0.5);       cudaThreadSynchronize();
    checkCudaError("Stage 2");
    // Stage 3
    calcFlux<<<nBlocks, nThreadsFlux>>>(k2, d_f, Np);       cudaThreadSynchronize();
    rhs<<<nBlocks, nThreads>>>(k2, d_f, d_Dr, d_rx, a, Np); cudaThreadSynchronize();
    rk4<<<nBlocks, nThreads>>>(d_u, k3, k2,  h, 0.5);       cudaThreadSynchronize();
    checkCudaError("Stage 3");
    // Stage 4
    calcFlux<<<nBlocks, nThreadsFlux>>>(k3, d_f, Np);       cudaThreadSynchronize();
    rhs<<<nBlocks, nThreads>>>(k3, d_f, d_Dr, d_rx, a, Np); cudaThreadSynchronize();
    rk4<<<nBlocks, nThreads>>>(d_u, k4, k3,  h, 1);         cudaThreadSynchronize();
    checkCudaError("Stage 4");

    // Copy the data back to compute this timestep; this is horribly inefficient
    cudaMemcpy(l_k1, k1, size * sizeof(float), cudaMemcpyDeviceToHost); 
    checkCudaError("error after k1");
    cudaMemcpy(l_k2, k2, size * sizeof(float), cudaMemcpyDeviceToHost); 
    checkCudaError("error after k2");
    cudaMemcpy(l_k3, k3, size * sizeof(float), cudaMemcpyDeviceToHost); 
    checkCudaError("error after k3");
    cudaMemcpy(l_k4, k4, size * sizeof(float), cudaMemcpyDeviceToHost); 
    checkCudaError("error after k4");

    checkCudaError("error after rk4");

    // result_u = k1/6 + k2/3 + k3/3 + k4/6
    for (i = 0; i < size; i++) {
        u[i] = l_k1[i]/6 + l_k2[i]/3 + l_k3[i]/3 + l_k4[i]/6;
    }

    free(l_k1);
    free(l_k2);
    free(l_k3);
    free(l_k4);
}

/* returns the Dr matrix
 */
void setDr(float *&Dr, int Np) {
    Dr[0] = -0.5;
    Dr[1] =  0.5;
    Dr[2] = -0.5;
    Dr[3] =  0.5;
}

/* allocate memory on the GPU
 */
void initGPU(int h, int Np, int K) {
    int size = K * Np;
    cudaDeviceReset();
    checkCudaError("error after reset?");
    printf("size = %i\n", size * sizeof(float));
    // Main variables
    cudaMalloc((void **) &d_u , size * sizeof(float));
    cudaMalloc((void **) &d_Dr, Np * Np * sizeof(float));
    cudaMalloc((void **) &d_f,  (K - 1) * sizeof(float));
    cudaMalloc((void **) &d_rx, size * sizeof(float));

    // Runge-Kutta storage
    cudaMalloc((void **) &k1 , size * sizeof(float));
    cudaMalloc((void **) &k2 , size * sizeof(float));
    cudaMalloc((void **) &k3 , size * sizeof(float));
    cudaMalloc((void **) &k4 , size * sizeof(float));
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
    float aspeed = 1;      // the wave speed
    
    float K = 10; // the mesh size
    float a = 0;  // left boundary
    float b = 1;  // right boundary
    float h = (b - a) / K; // size of cell
    timesteps = 10;

    int Np = 2;  // polynomial order

    size = Np * K;  // size of u

    Dr    = (float *) malloc(Np * Np * sizeof(float));
    u     = (float *) malloc(K * Np * sizeof(float));
    mesh  = (float *) malloc(K * sizeof(float));
    x     = (float *) malloc(K * Np * sizeof(float));
    rx    = (float *) malloc(K * Np * sizeof(float));
    f     = (float *) malloc((K - 1) * sizeof(float));

    // Init the mesh's endpoints
    for (i = 0; i < K; i++) {
        mesh[i]   = a + h * i;
    }

    // Init the mesh
    for (i = 0; i < K - 1; i++) {
        x[Np*i] = mesh[i];
        x[Np*i + 1] = mesh[i+ 1];
    }

    printf("x = ");
    for (i = 0; i < Np * K; i++) {
        printf("%f, ", x[i]);
    }
    printf("\n");

    // Initialize u0
    printf("u = ");
    for (i = 0; i < K; i++) {
        for (j = 0; j < Np; j++) {
            u[i*Np + j] = sin(x[i]);
            printf("%f, ", u[i]);
        }
    }
    printf("\n");

    // set Dr
    setDr(Dr, Np);

    // compute rx
    for (i = 0; i < K; i++) {
        for (j = 0; j < Np; j++) {
            for (k = 0; k < Np; k++) {        
                rx[Np*i + j] += Dr[j*Np + k] * x[k*Np + i];
                //Dr[j*p + k] * x[i*p + k
                ////rx[i] = 1 / 
            }
        }
    }

    for (i = 0; i < Np * K; i++) {
        rx[i] = 1 / rx[i];
    }

    initGPU(h, Np, K);
    cudaMemcpy(d_u,  u,  size * sizeof(float) , cudaMemcpyHostToDevice); 
    cudaMemcpy(d_Dr, Dr, Np * Np * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_rx, rx, size * sizeof(float) , cudaMemcpyHostToDevice); 
    checkCudaError("error in memcpy");

    for (t = 0; t < timesteps; t++) {
        printf("u%i =", t);
        for (i = 0; i < size; i++) {
            printf("%f, ", u[i]);
        }
        printf("\n");
        timeIntegrate(u, a, K, h, Np);
    }
    free(u);
    free(x);
    free(f);
    free(Dr);
    free(mesh);
    free(rx);
}
