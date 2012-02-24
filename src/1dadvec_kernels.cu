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
    float aspeed = 2.*3.14159; // the wave speed
    return aspeed*u;
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
__global__ void initX(float *mesh, float *x, float *r, float dx, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i;

    if (idx < K + 1) {
        // mesh[idx] holds the begining point for this element.
        for (i = 0; i < Np+1; i++) {
            x[(Np + 1)*idx + i] = mesh[idx] + (1. + r[i])/2.*dx;
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
    float cl[NP_MAX], cr[NP_MAX];

    if (idx < K+1) {
        // periodic
        // inflow
        if (idx == 0) {
            f[idx] = 0;//-sinf(aspeed*time);
        }
        if (idx > 0) {
            for (i = 0; i < Np+1; i++) {
                cl[i] = u[(Np + 1)*(idx - 1) + i];
                cr[i] = u[(Np + 1)*idx + i];
            }

            // Left value
            ul = 0;
            for (i = 0; i < Np+1; i++) {
                ul += cl[i];
            }
            // Evaluate flux 
            ul = flux(ul);

            // Right value
            ur = 0;
            for (i = 0; i < Np+1; i++) {
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
        case 0: return  1.;
        case 1: return  x;
        case 2: return  (3.*powf(x,2) -1.) / 2.;
        case 3: return  (5.*powf(x,3) - 3.*x) / 2.;
        case 4: return  (35.*powf(x,4) - 30.*powf(x,2) + 3.)/8.;
        case 5: return  (63.*powf(x,5) - 70.*powf(x,3) + 15.*x)/8.;
        case 6: return  (231.*powf(x,6) - 315.*powf(x,4) + 105.*powf(x,2) -5.)/16.;
        case 7: return  (429.*powf(x,7) - 693.*powf(x,5) + 315.*powf(x,3) - 35.*x)/16.;
        case 8: return  (6435.*powf(x,8) - 12012.*powf(x,6) + 6930.*powf(x,4) - 1260.*powf(x,2) + 35.)/128.;
        case 9: return  (12155.*powf(x,9) - 25740.*powf(x,7) + 18018*powf(x,5) - 4620.*powf(x,3) + 315.*x)/128.;
        case 10: return (46189.*powf(x,10) - 109395.*powf(x,8) + 90090.*powf(x,6) - 30030.*powf(x,4) + 3465.*powf(x,2) - 63.)/256.;
    }
    return -1;
}

/* legendre polynomials derivatives
 *
 * Calculates the value of d/dx P_i(x) 
 */
__device__ float legendreDeriv(float x, int i) {
    switch (i) {
        case 0: return 0.;
        case 1: return 1.;
        case 2: return 3.*x;
        case 3: return (15.*powf(x,2) - 3.) / 2.;
        case 4: return (140.*powf(x,3) - 60*x)/8.;
        case 5: return (315.*powf(x,4) - 210.*powf(x,2) + 15.)/8.;
        case 6: return (1386.*powf(x,5) - 1260.*powf(x,3) + 210.*x)/16.;
        case 7: return (3003.*powf(x,6) - 3465.*powf(x,4) + 945.*powf(x,2) - 35.)/16.;
        case 8: return (51480.*powf(x,7) - 72072.*powf(x,5) + 27720.*powf(x,3) - 2520.*x)/128.;
        case 9: return (109395.*powf(x,8) - 180180.*powf(x,6) + 90090.*powf(x,4) - 13860.*powf(x,2) + 315.)/128.;
        case 10: return (461890.*powf(x,9) - 875160.*powf(x,7) + 540540.*powf(x,5) - 120120.*powf(x,3) + 6930.*x)/256.;
    }
    return -1;
}

/* initial condition function
 *
 * returns the value of the intial condition at point x
 */
__device__ float u0(float x) {
    return sinf(2*3.14159*x);
}

/* calculate the initial data for U
 *
 * needs to interpolate u0 with legendre polynomials to grab the right coefficients.
 */
__global__ void initU(float *u, float *x, float *w, float *r, float dx, int K, int Np) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;
    float xi, uval;

    if (idx < K) {
        for (i = 0; i < Np+1; i++) {
            uval = 0.;
            for (j = 0; j < Np+1; j++) {
                // The mapping to the integration points for u0
                xi = x[(Np+1)*idx + j] + dx*(r[j] - 1.)/2.;
                uval += w[j] * u0(xi) * legendre(r[j], i);
            }
            // Leftover from integration
            u[(Np+1)*idx + i] = (2.*i + 1.)/2. * uval;
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
    float rhs[NP_MAX], register_c[NP_MAX];
    float lflux, rflux, u;

    if (idx < K) {
        // Read the global u into a register variable and set rhs = 0.
        for (i = 0; i < Np+1; i++) {
            register_c[i] = c[(Np+1)*idx + i];
        }

        // Perform quadrature W*P'*f(U) at integration points
        for (i = 0; i < Np+1; i++) {
            rhs[i] = 0.;
            for (j = 0; j < Np+1; j++) {
                // Evaluate u(r_j)
                u = 0.;
                for (k = 0; k < Np+1; k++) {
                    u += legendre(r[j], k) * register_c[k];
                }
                // rhs = sum w_j P'(r_j) flux(u_j)
                rhs[i] += w[j] * legendreDeriv(r[j], i) * flux(u);
            }
        }

        // Read the flux contributions.
        lflux  = f[idx];
        rflux  = f[idx+1];

        // Store result
        for (i = 0; i < Np+1; i++) {
            kstar[(Np+1)*idx + i] = dt*(((2.*i+1.) / dx) * (-rflux + powf(-1.,i) * lflux + rhs[i]));
        }
    }
}

/* tempstorage for RK4
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
__global__ void rk4_tempstorage(float *u, float *kstar, float*k, float alpha, float dt, int Np, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < (Np + 1) * K) {
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

    if (idx < (Np + 1) * K) {
        u[idx] += k1[idx]/6. + k2[idx]/3. + k3[idx]/3. + k4[idx]/6.;
    }
}
