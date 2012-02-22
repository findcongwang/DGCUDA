#include <cuda.h>

#define NP_MAX 7

// Important variables for GPU shit
float *d_u;
float *d_f;
float *d_rx;
float *d_x;
float *d_mesh;
float *d_r;
float *d_w;

// Runge-Kutta time integration storage
float *d_kstar;
float *d_k1;
float *d_k2;
float *d_k3;
float *d_k4;

/* flux function f(u)
 *
 * evaluate the flux function f(u)
 */
__device__ float flux(float u) {
    float aspeed = 2*3.14159; // the wave speed
    return aspeed*u;
}

/* initial condition function
 *
 * returns the value of the intial condition at point x
 */
__device__ float u0(float x) {
    return -sin(x);
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
 * That is, fi is the flux between nodes i and i+1, making f a m+1 length vector.
 * Store results into f
 */
__global__ void calcFlux(float *u, float *f, float aspeed, float time, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i;
    float ul, ur;
    float cl[2], cr[2];

    if (idx < K+1) {
        // periodic
        if (idx == 0) {
            f[idx] = f[Np*(K + 1) - 1];//sinf(time);
        }
        if (idx > 0) {
            for (i = 0; i < Np; i++) {
                cl[i] = u[Np*(idx - 1) + i];
                cr[i] = u[Np*idx + i];
            }

            // Left value
            ul = 0;
            for (i = 0; i < Np; i++) {
                ul += cl[i];
            }
            // Evaluate flux 
            ul = flux(ul);

            // Right value
            ur = 0;
            for (i = 0; i < Np; i++) {
                ur += powf(-1, i) * cr[i];
            }
            // Evaluate flux 
            ur = flux(ur);

            // Upwind flux
            f[idx] = ul;
        }
        // Outflow conditions
        //if (idx == K) {
            //f[idx] = 0;
        //}
    }
}

/* legendre polynomials
 *
 * Calculates the value of P_i(x) 
 */
__device__ float legendre(float x, int i) {
    switch (i) {
        case 0: return 1;
        case 1: return x;
        case 2: return (3*powf(x,2) -1) / 2;
    }
    return -1;
}

/* legendre polynomials derivatives
 *
 * Calculates the value of d/dx P_i(x) 
 */
__device__ float legendreDeriv(float x, int i) {
    switch (i) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 3*x;
    }
    return -1;
}

/* calculate the initial data for U
 *
 * needs to interpolate u0 with legendre polynomials to grab the right coefficients.
 */
__global__ void initU(float *u, float *x, float *w, float *r, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;

    if (idx < K) {
        for (i = 0; i < Np; i++) {
            u[Np*idx + i] = 0;
            for (j = 0; j < Np; j++) {
                u[Np*idx + i] += w[j] * u0(x[Np*idx + j]) * legendre(r[j], i);
            }
            u[Np*idx +i] *= (2*i + 1)/2;
        }
    }
}

/* right hand side calculations
 *
 * Calculates the flux integral 
 *  int_k (u * vprime) dx
 * and adds it to the flux boundary integral.
 * Store results into k, the RK variable
 */
__global__ void rhs(float *c, float *kstar, float *f, float *w, float *r, float a, float dt, float dx, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i,j, k;
    float rhs[2], register_c[2];
    float lflux, rflux, u;

    if (idx < K) {
        // Read the global u into a register variable and set rhs = 0.
        for (i = 0; i < Np; i++) {
            register_c[i] = c[Np*idx + i];
            rhs[i] = 0;
        }

        // Perform quadrature W*P'*f(U) at integration points
        for (i = 0; i < Np; i++) {
            for (j = 0; j < Np; j++) {
                // Evaluate f(u) at integration points x_i and P_j'(x_i)
                u = 0;
                for (k = 0; k < Np; k++) {
                    u += legendre(r[j], k) * register_c[k];
                }
                rhs[i] += w[j] * legendreDeriv(r[j], i) * flux(u);
            }
        }

        // Read the flux contributions.
        lflux  = f[idx];
        rflux  = f[idx+1];

        // Store result
        for (i = 0; i < Np; i++) {
            kstar[Np*idx + i] = ((2*i + 1) / dx) * dt * (-rflux + powf(-1, i) * lflux + rhs[i]);
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
