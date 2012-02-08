#include <cuda.h>

#define NP_MAX 7

// Important variables for GPU shit
float *d_u;
float *d_f;
float *d_rx;
float *d_x;
float *d_Dr;
float *d_mesh;
float *d_r;

// Runge-Kutta time integration storage
float *d_kstar;
float *d_k1;
float *d_k2;
float *d_k3;
float *d_k4;

/* calculate the initial data for U
 *
 * this should calculate Np of the points since we do this on an element by element basis.
 */
__global__ void initU(float *u, float *x, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < Np * K) {
        u[idx] = sin(x[idx]);
    }
}

/* initilialize the mesh nodes
 *
 * ideally, this should be done on the GPU, but meh
 */
__global__ void initMesh(float *mesh, float *x, float h, float a, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < K + 1) {
        mesh[idx] = a + h * idx;
    }
}

/* calculate the mesh using GLL points 
 *
 * ideally, this should be done on the GPU, but meh
 */
__global__ void initX(float *mesh, float *x, float *r, float h, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i;

    if (idx < K + 1) {
        // mesh[idx] holds the begining point for this element.
        for (i = 0; i < Np; i++) {
            x[Np*idx + i] = mesh[idx] + (1 + r[i])/2*h;
        }
    }
}

/* calculate the rx mapping for the gl nodes
 *
 * this should calculate Np of the points since we do this on an element by element basis.
 */
__global__ void initRx(float *rx, float *x, float *Dr, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;

    if (idx < K) {
        // Max order should be NP_MAX
        register float rhs[NP_MAX];
        register float r_x[NP_MAX];

        // Set rhs to zero and read the global x into register x.
        for (i = 0; i < Np; i++) {
            rhs[i] = 0;
            r_x[i] = x[idx*Np+ i];
        }

        // Calculate Dr * register x
        for (i = 0; i < Np; i++) {
            for (j = 0; j < Np; j++) {
                rhs[i] += Dr[i*Np + j] * r_x[j];
            }
        }

        // Store the mapping in rx
        for (i = 0; i < Np; i++) {
            rx[Np*idx + i] = 1./rhs[i];
        }
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
__global__ void calcFlux(float *u, float *f, float aspeed, float time, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float ul, ur;

    if (idx < K+1) {
        if (idx == 0) {
            f[idx] = aspeed*(-sin(aspeed*time) - u[idx]) / 2;
        }
        if (idx > 0) {
            // Flux calculations
            ul = u[idx*Np - 1];
            ur = u[idx*Np];

            // Central flux
            f[idx] = aspeed * (ul - ur) / 2;
        }
        if (idx == K) {
            f[idx] = 0;
        }
    }
}

/* right hand side calculations
 *
 * Calculates the volume and boundary contributions and adds them to the rhs
 * Gets called during time integration
 * Store results into u
 */
__global__ void rhs(float *u, float *k, float *f, float *Dr, float *rx, float a, float dt, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i,j;
    float rhs[3], r_u[3];
    float lflux, rflux;

    if (idx < K) {
        // Read the global u into a register variable and set rhs = 0.
        for (i = 0; i < Np; i++) {
            r_u[i] = u[Np*idx + i];
            rhs[i] = 0;
        }

        // Calculate Dr * u.
        for (i = 0; i < Np; i++) {
            for (j = 0; j < Np; j++) {
                rhs[i] += Dr[i*Np + j] * r_u[j];
            }
        }

        // Scale RHS up.
        for (i = 0; i < Np; i++) {
            rhs[i] *= -a*rx[Np*idx + i];
        }

        // Read the flux contributions.
        lflux  = f[idx];
        rflux  = f[idx+1];

        // LIFT
        rhs[0]    += rx[Np*idx] * lflux;
        rhs[Np-1] += rx[Np*(idx + 1) - 1] * rflux;

        // Store result
        for (i = 0; i < Np; i++) {
            k[Np*idx + i] = dt * rhs[i];
        }
    }
}

/* tempstorage for RK4
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
__global__ void rk4_tempstorage(float *u, float *kstar, float*k, float alpha, float dt, int Np, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < Np * K) {
        kstar[idx] = u[idx] + alpha * k[idx];
    }
}

/* rk4
 *
 * computes the runge-kutta solution 
 * u_n+1 = u_n + k1/6 + k2/3 + k3/3 + k4/6
 */
__global__ void rk4(float *u, float *k1, float *k2, float *k3, float *k4, int Np, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < Np * K) {
        u[idx] += k1[idx]/6 + k2[idx]/3 + k3[idx]/3 + k4[idx]/6;
    }
}
