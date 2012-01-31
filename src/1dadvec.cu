#include <cuda.h>
#include <math.h>
#include <stdio.h>

// Important variables for GPU shit
double *d_u;
double *d_f;
double *d_rx;
double *d_Dr;

// Runge-Kutta time integration storage
double *k1;
double *k2;
double *k3;
double *k4;

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
__global__ void calcFlux(double *u, double *f, int p) {
    int idx = threadIdx.x; 
    double ul, ur;

    // Flux calculations
    ul = u[(idx + 1) * p - 1];
    ur = u[(idx + 1) * p];

    // Central flux
    f[idx] = (ul + ur) / 2;
}

/* right hand side calculations
 *
 * Calculates the volume and boundary contributions and adds them to the rhs
 * Gets called during time integration
 * Store results into u
 */
__global__ void rhs(double *u, double *f, double *Dr, double *rx, double a, int p) {
    int idx = threadIdx.x;
    int i,j;
    double rhs[2], flux;

    // Initialize rhs to 0
    for (i = 0; i < p; i++) {
        rhs[i] = 0;
    }

    // Calculate Dr * u
    for (i = 0; i < p; i++) {
        for (j = 0; j < p; j++) {
            rhs[i] += Dr[i*p + j] * u[j];
        }
    }

    flux  = f[idx];

    // Scale RHS up
    for (i = 0; i < p; i++) {
        rhs[i] *= -a*rx[i];
    }

    // LIFT
    rhs[0]   -= flux;
    rhs[p-1] += flux;

    // Store result
    for (i = 0; i < p; i++) {
        u[idx*p + i] = rhs[i];
    }
}

__global__ void rk4(double *u, double *k, double *k_current, double dt, double alpha) {
    int idx = threadIdx.x;

    k_current[idx] = dt*(u[idx] + alpha * k[idx]);
}

void timeIntegrate(double *&u, double a, int h, int p) {
    int i;
    double *l_k1, *l_k2, *l_k3, *l_k4; 
    int size = h * p;

    l_k1 = (double *) malloc(size * sizeof(double));
    l_k2 = (double *) malloc(size * sizeof(double));
    l_k3 = (double *) malloc(size * sizeof(double));
    l_k4 = (double *) malloc(size * sizeof(double));

    dim3 nBlocks      = dim3(1);
    dim3 nThreads     = dim3(size);
    dim3 nThreadsFlux = dim3(h-1);

    // Stage 1
    rhs<<<nBlocks, nThreads>>>(d_u, d_f, d_Dr, d_rx, a, p);
    rk4<<<nBlocks, nThreads>>>(d_u, l_k1, d_u, h, 0);
    calcFlux<<<nBlocks, nThreadsFlux>>>(l_k1, d_f, p);
    // Stage 2
    rhs<<<nBlocks, nThreads>>>(k1, d_f, d_Dr, d_rx, a, p);
    rk4<<<nBlocks, nThreads>>>(d_u, l_k2, k1,  h, 0.5);
    calcFlux<<<nBlocks, nThreadsFlux>>>(l_k2, d_f, p);
    // Stage 3
    rhs<<<nBlocks, nThreads>>>(k2, d_f, d_Dr, d_rx, a, p);
    rk4<<<nBlocks, nThreads>>>(d_u, l_k3, k2,  h, 0.5);
    calcFlux<<<nBlocks, nThreadsFlux>>>(l_k3, d_f, p);
    // Stage 4
    rhs<<<nBlocks, nThreads>>>(k3, d_f, d_Dr, d_rx, a, p);
    rk4<<<nBlocks, nThreads>>>(d_u, l_k4, k3,  h, 1);

    // Copy the data back to compute this timestep; this is horribly inefficient
    cudaMemcpy(l_k1, l_k1, size * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(l_k2, l_k2, size * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(l_k3, l_k3, size * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(l_k4, l_k4, size * sizeof(double), cudaMemcpyHostToDevice); 

    checkCudaError("");

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
void setDr(double *&Dr, int p) {
    Dr[0] = -0.5;
    Dr[1] =  0.5;
    Dr[2] = -0.5;
    Dr[3] =  0.5;
}

/* allocate memory on the GPU
 */
void initGPU(int h, int p) {
    int size = h * p;
    // Main variables
    cudaMalloc((void **) &d_u , size * sizeof(double));
    cudaMalloc((void **) &d_Dr, p * p * sizeof(double));
    cudaMalloc((void **) &d_f, (h - 1) * sizeof(double));

    // Runge-Kutta storage
    cudaMalloc((void **) &k1 , size * sizeof(double));
    cudaMalloc((void **) &k2 , size * sizeof(double));
    cudaMalloc((void **) &k3 , size * sizeof(double));
    cudaMalloc((void **) &k4 , size * sizeof(double));
}

int main() {
    int i, j, size, t, timesteps;
    double *Dr;    // diff matrix
    double *u;     // the computed result
    double *f;     // the flux
    double *mesh;  // the mesh's endpoints
    double *x;     // the mesh
    double aspeed = 1;      // the wave speed
    
    double N = 10; // the mesh size
    double a = 0;  // left boundary
    double b = 1;  // right boundary
    double h = (b - a) / N; // size of cell
    timesteps = 10;

    int p = 2;  // polynomial order

    size = p * N;  // size of u

    Dr = (double *) malloc(p * p * sizeof(double));
    u  = (double *) malloc(N * p * sizeof(double));
    mesh  = (double *) malloc(N * sizeof(double));
    x  = (double *) malloc(N * p * sizeof(double));
    f  = (double *) malloc((N - 1) * sizeof(double));

    // Init the mesh's endpoints
    for (i = 0; i < N; i++) {
        mesh[i]   = a + h * i;
    }

    // Init the mesh
    for (i = 0; i < N - 1; i++) {
        x[p*i] = mesh[i];
        x[p*i + 1] = mesh[i+ 1];
    }

    printf("x = ");
    for (i = 0; i < p * N; i++) {
        printf("%f, ", x[i]);
    }
    printf("\n");

    // Initialize u0
    printf("u = ");
    for (i = 0; i < N; i++) {
        for (j = 0; j < p; j++) {
            u[i*p + j] = sin(x[i]);
            printf("%f, ", u[i]);
        }
    }
    printf("\n");

    setDr(Dr, p);

    initGPU(h, p);
    cudaMemcpy(d_u, u, size * sizeof(double), cudaMemcpyDeviceToHost); 
    cudaMemcpy(d_Dr, Dr, p * p * sizeof(double), cudaMemcpyDeviceToHost); 

    for (t = 0; t < timesteps; t++) {
        for (i = 0; i < size; i++) {
            printf("%f, ", u[i]);
        }
        printf("\n");
        timeIntegrate(u, a, h, p);
    }
    free(u);
    free(x);
    free(f);
    free(Dr);
}
