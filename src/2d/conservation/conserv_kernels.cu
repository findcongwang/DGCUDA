/* 2dadvec_kernels_euler.cu
 *
 * This file contains the kernels for the 2D euler DG method.
 *
 * d_t [   rho   ] + d_x [     rho * u    ] + d_y [    rho * v     ] = 0
 * d_t [ rho * u ] + d_x [ rho * u^2 + p  ] + d_y [   rho * u * v  ] = 0
 * d_t [ rho * v ] + d_x [  rho * u * v   ] + d_y [  rho * v^2 + p ] = 0
 * d_t [    E    ] + d_x [ u * ( E +  p ) ] + d_y [ v * ( E +  p ) ] = 0
 *

 */

#include "conserv_headers.cuh"

#define PI 3.14159
#define GAMMA 1.4
#define MACH 2.25
#define N_MAX 10

/***********************
 *
 * DEVICE VARIABLES
 *
 ***********************/
/* These are always prefixed with d_ for "device" */
double *d_c;                 // coefficients for [rho, rho * u, rho * v, E]
double *d_c_prev;            // coefficients for [rho, rho * u, rho * v, E]
double *d_quad_rhs;          // the right hand side containing the quadrature contributions
double *d_left_riemann_rhs;  // the right hand side containing the left riemann contributions
double *d_right_riemann_rhs; // the right hand side containing the right riemann contributions

// TODO: switch to low storage runge-kutta
// runge kutta variables
double *d_kstar;
double *d_k1;
double *d_k2;
double *d_k3;
double *d_k4;

// precomputed basis functions 
// TODO: maybe making these 2^n makes sure the offsets are cached more efficiently? who knows...
// precomputed basis functions ordered like so
//
// [phi_1(r1, s1), phi_1(r2, s2), ... , phi_1(r_nq, s_nq)   ]
// [phi_2(r1, s1), phi_2(r2, s2), ... , phi_2(r_nq, s_nq)   ]
// [   .               .           .            .           ]
// [   .               .           .            .           ]
// [   .               .           .            .           ]
// [phi_np(r1, s1), phi_np(r2, s2), ... , phi_np(r_nq, s_nq)]
//
__device__ __constant__ int N;
__device__ __constant__ double basis[2048];
// note: these are multiplied by the weights
__device__ __constant__ double basis_grad_x[2048]; 
__device__ __constant__ double basis_grad_y[2048]; 

// precomputed basis functions evaluated along the sides. ordered
// similarly to basis and basis_grad_{x,y} but with one "matrix" for each side
// starting with side 0. to get to each side, offset with:
//      side_number * n_p * num_quad1d.
//__device__ __constant__ int n_p;
//__device__ __constant__ int num_elem;
//__device__ __constant__ int num_sides;
//__device__ __constant__ int n_quad;
//__device__ __constant__ int n_quad1d;

__device__ __constant__ double basis_side[1024];
__device__ __constant__ double basis_vertex[256];

// weights for 2d and 1d quadrature rules
__device__ __constant__ double w[64];
__device__ __constant__ double w_oned[16];

__device__ __constant__ double r1[32];
__device__ __constant__ double r2[32];
__device__ __constant__ double r_oned[32];

void set_N(int value) {
    cudaMemcpyToSymbol("N", (void *) &value, sizeof(int));
}

void set_basis(void *value, int size) {
    cudaMemcpyToSymbol("basis", value, size * sizeof(double));
}
void set_basis_grad_x(void *value, int size) {
    cudaMemcpyToSymbol("basis_grad_x", value, size * sizeof(double));
}
void set_basis_grad_y(void *value, int size) {
    cudaMemcpyToSymbol("basis_grad_y", value, size * sizeof(double));
}
void set_basis_side(void *value, int size) {
    cudaMemcpyToSymbol("basis_side", value, size * sizeof(double));
}
void set_basis_vertex(void *value, int size) {
    cudaMemcpyToSymbol("basis_vertex", value, size * sizeof(double));
}
void set_w(void *value, int size) {
    cudaMemcpyToSymbol("w", value, size * sizeof(double));
}
void set_w_oned(void *value, int size) {
    cudaMemcpyToSymbol("w_oned", value, size * sizeof(double));
}
void set_r1(void *value, int size) {
    cudaMemcpyToSymbol("r1", value, size * sizeof(double));
}
void set_r2(void *value, int size) {
    cudaMemcpyToSymbol("r2", value, size * sizeof(double));
}
void set_r_oned(void *value, int size) {
    cudaMemcpyToSymbol("r_oned", value, size * sizeof(double));
}

// tells which side (1, 2, or 3) to evaluate this boundary integral over
int *d_left_side_number;
int *d_right_side_number;

double *d_J;         // jacobian determinant 
double *d_reduction; // for the min / maxes in the reductions 
double *d_lambda;    // stores computed lambda values for each element
double *d_s_length;  // length of sides

// the num_elem values of the x and y coordinates for the two vertices defining a side
// TODO: can i delete these after the lengths are precomputed?
//       maybe these should be in texture memory?
double *d_s_V1x;
double *d_s_V1y;
double *d_s_V2x;
double *d_s_V2y;

// the num_elem values of the x and y partials
double *d_xr;
double *d_yr;
double *d_xs;
double *d_ys;

// the K indices of the sides for each element ranged 0->H-1
int *d_elem_s1;
int *d_elem_s2;
int *d_elem_s3;

// vertex x and y coordinates on the mesh which define an element
// TODO: can i delete these after the jacobians are precomputed?
//       maybe these should be in texture memory?
double *d_V1x;
double *d_V1y;
double *d_V2x;
double *d_V2y;
double *d_V3x;
double *d_V3y;

// stores computed values at three vertices
double *d_Uv1;
double *d_Uv2;
double *d_Uv3;

// for computing the error
double *d_error;

// normal vectors for the sides
double *d_Nx;
double *d_Ny;

// index lists for sides
int *d_left_elem;  // index of left  element for side idx
int *d_right_elem; // index of right element for side idx

/***********************
 *
 * DEVICE FUNCTIONS
 *
 ***********************/
//__device__ double pressure(double rho, double u, double v, double E) {
    // TODO: this is a dirty fix, but it's necessary or else c collapses into NAN
    // This happens because 
    //     E < 0.5 rho (u^2 + v^2) 
    // which shouldn't ever be possible...
    //if ((GAMMA - 1) * (E - (u*u + v*v) / 2. * rho) < 0) {
        //return 0.0001;
    //}
    //return (GAMMA - 1.) * (E - (u*u + v*v) / 2. * rho);
//}

/* evaluate c
 *
 * evaulates the speed of sound c
 */
//__device__ double eval_c(double rho, double u, double v, double E) {
    //double p = pressure(rho, u, v, E);

    //return sqrtf(GAMMA * p / rho);
//}    

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x
 */
//__device__ double rho0(double x, double y) {
    //double r = sqrt(x*x + y*y);
    //return powf(1+1.0125*(1.- 1./(r * r)),2.5);
    //return powf(1 + (GAMMA - 1)/ 2. * MACH * MACH * (1 - powf(1. / r, 2)), 1./(GAMMA - 1));
//}
//__device__ double u0(double x, double y) {
    //double r = sqrt(x*x + y*y);
    //return sin(atan(y / x)) * MACH / r;
//}
//__device__ double v0(double x, double y) {
    //double r = sqrt(x*x + y*y);
    //return -cos(atan(y / x)) * MACH / r;
//}
//__device__ double E0(double x, double y) {
    //double r = sqrt(x*x + y*y);
    //double p = (1.0 / GAMMA) * powf(rho0(x, y), GAMMA);
    //return  0.5 * rho0(x,y) * (MACH*MACH/(r * r)) + p * (1./(GAMMA - 1.));
    //return powf(rho0(x,y),GAMMA) / (GAMMA * (GAMMA - 1)) + (powf(u0(x, y), 2) + powf(v0(x, y), 2)) / 2. * rho0(x, y);
//}

/*
__device__ void reflecting_boundary(double rho_left, double *rho_right,
                         double u_left,   double *u_right,
                         double v_left,   double *v_right,
                         double E_left,   double *E_right,
                         double v1x,      double v1y, 
                         double v2x,      double v2y,
                         double v3x,      double v3y,
                         double nx,       double ny,
                         int j, int left_side, int n_quad1d) {

    double r1_eval, r2_eval;
    // we need the mapping back to the grid space
    switch (left_side) {
        case 0: 
            r1_eval = 0.5 + 0.5 * r_oned[j];
            r2_eval = 0.;
            break;
        case 1: 
            r1_eval = (1. + r_oned[j]) / 2.;
            r2_eval = (1. - r_oned[j]) / 2.;
            break;
        case 2: 
            r1_eval = 0.;
            r2_eval = 0.5 + 0.5 * r_oned[n_quad1d - 1 - j];
            break;
    }

    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    double x = v2x * r1_eval + v3x * r2_eval + v1x * (1 - r1_eval - r2_eval);
    double y = v2y * r1_eval + v3y * r2_eval + v1y * (1 - r1_eval - r2_eval);

    // set the sides to reflect
    *rho_right = rho_left;
    *E_right   = E_left;

    // make the velocities reflect wrt the normal
    // -2 (V dot N) * N + V
    //double dot = u_left * nx + v_left * ny;
    // *u_right   = u_left - 2 * dot * nx;
    // *v_right   = v_left - 2 * dot * ny;

    // taken from lilia's code:
    //double vn = -(u_left * nx + v_left * ny);
    //double vt = u_left * ny - v_left * nx;

    // *u_right = vn * nx + vt * ny;
    // *v_right = vn * ny - vt * nx;

    // taken from algorithm 1
    // *u_right = (u_left * ny - v_left * nx)*ny;
    // *v_right = -(u_left * ny - v_left * nx)*nx;

    // *u_right = u0(x,y);
    // *v_right = v0(x,y);

    double dot = sqrtf(x*x + y*y);
    double Nx = x / dot;
    double Ny = y / dot;

    if (Nx * nx + Ny * ny < 0) {
        Nx *= -1;
        Ny *= -1;
    }

    *u_right = (u_left * Ny - v_left * Nx)*Ny;
    *v_right = -(u_left * Ny - v_left * Nx)*Nx;

}

__device__ void outflow_boundary(double rho_left, double *rho_right,
                                 double u_left,   double *u_right,
                                 double v_left,   double *v_right,
                                 double E_left,   double *E_right,
                                 double nx,       double ny) {
    // make the flow move along the normal outside the cell so we don't introduce any new flow
    double dot = sqrtf(u_left * u_left + v_left * v_left);
    *rho_right = rho_left;
    *u_right   = dot * nx; //TODO: is this right? it just sort of worked...
    *v_right   = dot * ny; //TODO: is this right? it just sort of worked...
    *E_right   = E_left;
}

__device__ void inflow_boundary(double *rho_right, double *u_right, double *v_right, double *E_right,
                                double v1x, double v1y, 
                                double v2x, double v2y,
                                double v3x, double v3y,
                                int j,
                                int left_side, int n_quad1d) {

    double r1_eval, r2_eval;
    double x, y;

    // we need the mapping back to the grid space
    switch (left_side) {
        case 0: 
            r1_eval = 0.5 + 0.5 * r_oned[j];
            r2_eval = 0.;
            break;
        case 1: 
            r1_eval = (1. - r_oned[j]) / 2.;
            r2_eval = (1. + r_oned[j]) / 2.;
            break;
        case 2: 
            r1_eval = 0.;
            r2_eval = 0.5 + 0.5 * r_oned[n_quad1d - 1 - j];
            break;
    }

    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    x = v2x * r1_eval + v3x * r2_eval + v1x * (1 - r1_eval - r2_eval);
    y = v2y * r1_eval + v3y * r2_eval + v1y * (1 - r1_eval - r2_eval);
        
    *rho_right = rho0(x, y);
    *u_right   = u0(x, y);
    *v_right   = v0(x, y);
    *E_right   = E0(x, y);
}
*/

/* initial conditions
 *
 * computes the coefficients for the initial conditions
 * THREADS: num_elem
 */
__global__ void init_conditions(double *c, double *J,
                                double *V1x, double *V1y,
                                double *V2x, double *V2y,
                                double *V3x, double *V3y,
                                int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, n;
    double U[N_MAX];

    if (idx < num_elem) {
        for (i = 0; i < n_p; i++) {
            // evaluate U times the i'th basis function
            evalU0(U, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], i, n_p, n_quad);

            // store the coefficients
            for (n = 0; n < 4; n++) {
                c[num_elem * n_p * n + i * num_elem + idx] = U[n];
            }
        } 
    }
}

/* min reduction function
 *
 * returns the min value from the global data J and stores in min_J
 * each block computes the min jacobian inside of that block and stores it in the
 * blockIdx.x spot of the shared min_J variable.
 * NOTE: this is fixed for 256 threads.
 */
__global__ void min_reduction(double *D, double *min_D, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int i   = (blockIdx.x * 256 * 2) + threadIdx.x;

    __shared__ double s_min[256];

    if (idx < num_elem) {
        // set all of min to D[idx] initially
        s_min[tid] = D[idx];
        __syncthreads();

        // test a few
        while (i < num_elem) {
            s_min[tid] = (s_min[tid] < D[i]) ? s_min[tid] : D[i];
            s_min[tid] = (s_min[tid] < D[i + 256]) ? s_min[tid] : D[i];
            i += gridDim.x * 256 * 2;
            __syncthreads();
        }

        // first half of the warps
        __syncthreads();
        if (tid < 128) {
            s_min[tid] = (s_min[tid] < s_min[tid + 128]) ? s_min[tid] : s_min[tid + 128];
        }

        // first and second warps
        __syncthreads();
        if (tid < 64) {
            s_min[tid] = (s_min[tid] < s_min[tid + 64]) ? s_min[tid] : s_min[tid + 64];
        }

        // unroll last warp
        __syncthreads();
        if (tid < 32) {
            if (blockDim.x >= 64) {
                s_min[tid] = (s_min[tid] < s_min[tid + 32]) ? s_min[tid] : s_min[tid + 32];
            }
            if (blockDim.x >= 32) {
                s_min[tid] = (s_min[tid] < s_min[tid + 16]) ? s_min[tid] : s_min[tid + 16];
            }
            if (blockDim.x >= 16) {
                s_min[tid] = (s_min[tid] < s_min[tid + 8]) ? s_min[tid] : s_min[tid + 8];
            }
            if (blockDim.x >= 8) {
                s_min[tid] = (s_min[tid] < s_min[tid + 4]) ? s_min[tid] : s_min[tid + 4];
            }
            if (blockDim.x >= 4) {
                s_min[tid] = (s_min[tid] < s_min[tid + 2]) ? s_min[tid] : s_min[tid + 2];
            }
            if (blockDim.x >= 2) {
                s_min[tid] = (s_min[tid] < s_min[tid + 1]) ? s_min[tid] : s_min[tid + 1];
            }
        }

        __syncthreads();
        if (tid == 0) {
            min_D[blockIdx.x] = s_min[0];
        }
    }
}

/* max reduction function
 *
 * returns the max value from the global data D and stores in max
 * each block computes the max jacobian inside of that block and stores it in the
 * blockIdx.x spot of the shared max variable.
 * NOTE: this is fixed for 256 threads.
 */
__global__ void max_reduction(double *D, double *max_D, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int i   = (blockIdx.x * 256 * 2) + threadIdx.x;

    __shared__ double s_max[256];

    if (idx < num_elem) {
        // set all of max to D[idx] initially
        s_max[tid] = D[idx];
        __syncthreads();

        // test a few
        while (i + 256 < num_elem) {
            s_max[tid] = (s_max[tid] > D[i]) ? s_max[tid] : D[i];
            s_max[tid] = (s_max[tid] > D[i + 256]) ? s_max[tid] : D[i];
            i += gridDim.x * 256 * 2;
            __syncthreads();
        }

        // first half of the warps
        __syncthreads();
        if (tid < 128) {
            s_max[tid] = (s_max[tid] > s_max[tid + 128]) ? s_max[tid] : s_max[tid + 128];
        }

        // first and second warps
        __syncthreads();
        if (tid < 64) {
            s_max[tid] = (s_max[tid] > s_max[tid + 64]) ? s_max[tid] : s_max[tid + 64];
        }

        // unroll last warp
        __syncthreads();
        if (tid < 32) {
            if (blockDim.x >= 64) {
                s_max[tid] = (s_max[tid] > s_max[tid + 32]) ? s_max[tid] : s_max[tid + 32];
            }
            if (blockDim.x >= 32) {
                s_max[tid] = (s_max[tid] > s_max[tid + 16]) ? s_max[tid] : s_max[tid + 16];
            }
            if (blockDim.x >= 16) {
                s_max[tid] = (s_max[tid] > s_max[tid + 8]) ? s_max[tid] : s_max[tid + 8];
            }
            if (blockDim.x >= 8) {
                s_max[tid] = (s_max[tid] > s_max[tid + 4]) ? s_max[tid] : s_max[tid + 4];
            }
            if (blockDim.x >= 4) {
                s_max[tid] = (s_max[tid] > s_max[tid + 2]) ? s_max[tid] : s_max[tid + 2];
            }
            if (blockDim.x >= 2) {
                s_max[tid] = (s_max[tid] > s_max[tid + 1]) ? s_max[tid] : s_max[tid + 1];
            }
        }

        __syncthreads();
        if (tid == 0) {
            max_D[blockIdx.x] = s_max[0];
        }
    }
}

/***********************
 *
 * PRECOMPUTING
 *
 ***********************/

/* side length computer
 *
 * precomputes the length of each side.
 * THREADS: num_sides
 */ 
__global__ void preval_side_length(double *s_length, 
                              double *s_V1x, double *s_V1y, 
                              double *s_V2x, double *s_V2y,
                              int num_sides) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        // compute and store the length of the side
        s_length[idx] = sqrtf(powf(s_V1x[idx] - s_V2x[idx],2) + powf(s_V1y[idx] - s_V2y[idx],2));
    }
}

/* inscribed circle radius computing
 *
 * computes the radius of each inscribed circle. stores in d_J to find the minumum,
 * then we reuse d_J.
 */
__global__ void preval_inscribed_circles(double *J,
                                    double *V1x, double *V1y,
                                    double *V2x, double *V2y,
                                    double *V3x, double *V3y,
                                    int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double a, b, c, k;
        a = sqrtf(powf(V1x[idx] - V2x[idx], 2) + powf(V1y[idx] - V2y[idx], 2));
        b = sqrtf(powf(V2x[idx] - V3x[idx], 2) + powf(V2y[idx] - V3y[idx], 2));
        c = sqrtf(powf(V1x[idx] - V3x[idx], 2) + powf(V1y[idx] - V3y[idx], 2));

        k = 0.5 * (a + b + c);

        // for the diameter, we multiply by 2
        J[idx] = 2 * sqrtf(k * (k - a) * (k - b) * (k - c)) / k;
    }
}

/* jacobian computing
 *
 * precomputes the jacobian determinant for each element.
 * THREADS: num_elem
 */
__global__ void preval_jacobian(double *J, 
                           double *V1x, double *V1y, 
                           double *V2x, double *V2y, 
                           double *V3x, double *V3y,
                           int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double x1, y1, x2, y2, x3, y3;

        // read vertex points
        x1 = V1x[idx];
        y1 = V1y[idx];
        x2 = V2x[idx];
        y2 = V2y[idx];
        x3 = V3x[idx];
        y3 = V3y[idx];

        // calculate jacobian determinant
        // x = x2 * r + x3 * s + x1 * (1 - r - s)
        J[idx] = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    }
}

/* evaluate normal vectors
 *
 * computes the normal vectors for each element along each side.
 * THREADS: num_sides
 *
 */
__global__ void preval_normals(double *Nx, double *Ny, 
                          double *s_V1x, double *s_V1y, 
                          double *s_V2x, double *s_V2y,
                          double *V1x, double *V1y, 
                          double *V2x, double *V2y, 
                          double *V3x, double *V3y,
                          int *left_side_number, int num_sides) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        double x, y, length;
        double sv1x, sv1y, sv2x, sv2y;
    
        sv1x = s_V1x[idx];
        sv1y = s_V1y[idx];
        sv2x = s_V2x[idx];
        sv2y = s_V2y[idx];
    
        // lengths of the vector components
        x = sv2x - sv1x;
        y = sv2y - sv1y;
    
        // normalize
        length = sqrtf(powf(x, 2) + powf(y, 2));

        // store the result
        Nx[idx] = -y / length;
        Ny[idx] =  x / length;
    }
}

__global__ void preval_normals_direction(double *Nx, double *Ny, 
                          double *V1x, double *V1y, 
                          double *V2x, double *V2y, 
                          double *V3x, double *V3y,
                          int *left_elem, int *left_side_number, int num_sides) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        double new_x, new_y, dot;
        double initial_x, initial_y, target_x, target_y;
        double x, y;
        int left_idx, side;

        // get left side's vertices
        left_idx = left_elem[idx];
        side     = left_side_number[idx];

        // get the normal vector
        x = Nx[idx];
        y = Ny[idx];
    
        // make it point the correct direction by learning the third vertex point
        switch (side) {
            case 0: 
                target_x = V3x[left_idx];
                target_y = V3y[left_idx];
                initial_x = (V1x[left_idx] + V2x[left_idx]) / 2.;
                initial_y = (V1y[left_idx] + V2y[left_idx]) / 2.;
                break;
            case 1:
                target_x = V1x[left_idx];
                target_y = V1y[left_idx];
                initial_x = (V2x[left_idx] + V3x[left_idx]) / 2.;
                initial_y = (V2y[left_idx] + V3y[left_idx]) / 2.;
                break;
            case 2:
                target_x = V2x[left_idx];
                target_y = V2y[left_idx];
                initial_x = (V1x[left_idx] + V3x[left_idx]) / 2.;
                initial_y = (V1y[left_idx] + V3y[left_idx]) / 2.;
                break;
        }

        // create the vector pointing towards the third vertex point
        new_x = target_x - initial_x;
        new_y = target_y - initial_y;

        // find the dot product between the normal and new vectors
        dot = x * new_x + y * new_y;
        
        if (dot > 0) {
            Nx[idx] *= -1;
            Ny[idx] *= -1;
        }
    }
}

__global__ void preval_partials(double *V1x, double *V1y,
                                double *V2x, double *V2y,
                                double *V3x, double *V3y,
                                double *xr,  double *yr,
                                double *xs,  double *ys, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elem) {
        // evaulate the jacobians of the mappings for the chain rule
        // x = x2 * r + x3 * s + x1 * (1 - r - s)
        xr[idx] = V2x[idx] - V1x[idx];
        yr[idx] = V2y[idx] - V1y[idx];
        xs[idx] = V3x[idx] - V1x[idx];
        ys[idx] = V3y[idx] - V1y[idx];
    }
}

/***********************
 *
 * MAIN FUNCTIONS
 *
 ***********************/

/* limiter
 *
 * the standard limiter for coefficient values
 */
__global__ void limit_c(double *c_inner, 
                   double *c_s1, double *c_s2, double *c_s3,
                   int n_p, int num_elem) {

    //int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // get cell averages
    //avg_inner = c_inner[0];
    //avg_s1 = c_s1[0];
    //avg_s2 = c_s2[0];
    //avg_s3 = c_s3[0];

    // determine if this is a "troubled" cell

    //for (i = n_p; i > 1; i++) {
        //c_prev = c[i - 1];
    //}
}

// calculate U_left and U_right
 __device__ void eval_left_right(double *C_left, double *C_right, 
                             double *U_left, double *U_right,
                             double nx, double ny,
                             double v1x, double v1y,
                             double v2x, double v2y,
                             double v3x, double v3y,
                             int j, // j, as usual, is the index of the integration point
                             int left_side, int right_side,
                             int left_idx, int right_idx,
                             int n_p, int num_elem, int n_quad1d,
                             int num_sides, double t) { 

    int i, n;

    // set U to 0
    for (n = 0; n < N; n++) {
        U_left[n]  = 0.;
        U_right[n] = 0.;
    }

    //evaluate U at the integration points
    for (i = 0; i < n_p; i++) {
        for (n = 0; n < N; n++) {
            U_left[n] += C_left[n*n_p + i] * 
                         basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
        }
    }

    // TODO: sort boundaries to avoid warp divergence
    switch (right_idx) {
        // reflecting 
        case -1: 
            reflecting_boundary(U_left, U_right,
                v1x, v1y, v2x, v2y, v3x, v3y, 
                nx, ny,
                j, left_side, n_quad1d);
            break;
        // outflow 
        case -2: 
            outflow_boundary(U_left, U_right,
                v1x, v1y, v2x, v2y, v3x, v3y, 
                nx, ny,
                j, left_side, n_quad1d);
            break;
        // inflow 
        case -3: 
            inflow_boundary(U_left, U_right,
                v1x, v1y, v2x, v2y, v3x, v3y, 
                nx, ny, 
                j, left_side, n_quad1d);
            break;
        // not a boundary
        default:
            // evaluate the right side at the integration point
            for (i = 0; i < n_p; i++) {
                for (n = 0; n < N; n++) {
                    U_right[n] += C_right[n*n_p + i] * 
                                  basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - j - 1];
                }
            }
            break;
    }
}

/* surface integral evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: num_sides
 */
__device__ void eval_surface(double *C_left, double *C_right,
                             double *left_riemann_rhs, double *right_riemann_rhs, 
                             double len, 
                             double v1x, double v1y,
                             double v2x, double v2y,
                             double v3x, double v3y,
                             int left_idx,  int right_idx,
                             int left_side, int right_side,
                             double nx, double ny, 
                             int n_quad1d, int n_quad, int n_p, int num_sides, 
                             int num_elem, double t, int idx) {
    int i, j, n;
    double s;
    double lambda;
    register double sum_left[4], sum_right[4];
    register double flux_x_l[4], flux_y_l[4];
    register double flux_x_r[4], flux_y_r[4];
    register double U_left[4], U_right[4];

    // multiply across by the i'th basis function
    for (i = 0; i < n_p; i++) {

        // initilize to zero
        for (n = 0; n < N; n++) {
            sum_left [n] = 0.;
            sum_right[n] = 0.;
        }

        for (j = 0; j < n_quad1d; j++) {
            // calculate the left and right values along the surface
            eval_left_right(C_left, C_right,
                            U_left, U_right,
                            nx, ny,
                            v1x, v1y, v2x, v2y, v3x, v3y,
                            j, left_side, right_side,
                            left_idx, right_idx,
                            n_p, num_elem, n_quad1d, num_sides, t);

            // calculate the left and right fluxes
            eval_flux(U_left, flux_x_l, flux_y_l);
            eval_flux(U_right, flux_x_r, flux_y_r);

            // calculate the max wave speed at this integration point
            lambda = eval_lambda(U_left, U_right, nx, ny);

            // calculate the riemann problem
            for (n = 0; n < N; n++) {
                s = 0.5 * ((flux_x_l[n] + flux_x_r[n]) * nx + (flux_y_l[n] + flux_y_r[n]) * ny 
                            + lambda * (U_left[n] - U_right[n]));
                sum_left[n]  += w_oned[j] * s * basis_side[left_side  * n_p * n_quad1d + i * n_quad1d + j];
                sum_right[n] += w_oned[j] * s * basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];
            }
        }

        // store this side's contribution in the riemann rhs vectors
        __syncthreads();
        for (n = 0; n < N; n++) {
            left_riemann_rhs[num_sides * n_p * n + i * num_sides + idx]  = -len / 2. * sum_left[n];
            right_riemann_rhs[num_sides * n_p * n + i * num_sides + idx] =  len / 2. * sum_right[n];
        }
    }
}


/* volume integrals
 *
 * evaluates and adds the volume integral to the rhs vector
 * THREADS: num_elem
 */
__device__ void eval_volume(double *C, double *quad_rhs, 
                            double x_r, double y_r, double x_s, double y_s,
                            int n_quad, int n_p, int num_elem, int idx) {
    int i, j, k, n;
    register double U[4];
    register double flux_x[4], flux_y[4];
    register double sum[4];

    // evaluate the volume integral for each coefficient
    for (i = 0; i < n_p; i++) {

        // initialize sum to 0
        for (n = 0; n < N; n++) {
            sum[n] = 0.;
        }

        // for each integration point
        for (j = 0; j < n_quad; j++) {
            // initialize to zero
            for (n = 0; n < N; n++) {
                U[n] = 0.;
            }
            // calculate at the integration point
            for (k = 0; k < n_p; k++) {
                for (n = 0; n < N; n++) {
                    U[n] += C[n*n_p + k] * basis[n_quad * k + j];
                }
            }
            // evaluate the flux
            eval_flux(U, flux_x, flux_y);
            // compute the sum
            //     [fx fy] * [y_s, -y_r; -x_s, x_r] * [phi_x phi_y]
            for (n = 0; n < N; n++) {
                sum[n] += flux_x[n] * ( basis_grad_x[n_quad * i + j] * y_s
                                       -basis_grad_y[n_quad * i + j] * y_r)
                        + flux_y[n] * (-basis_grad_x[n_quad * i + j] * x_s 
                                      + basis_grad_y[n_quad * i + j] * x_r);
            }
        }

        // store the result
        for (n = 0; n < N; n++) {
            quad_rhs[num_elem * n_p * n + i * num_elem + idx] = sum[n];
        }
    }
}

/* evaluate u
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
__global__ void eval_u(double *C, 
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) {
        int i;
        double uv1, uv2, uv3;

        // calculate values at the integration points
        uv1 = 0.;
        uv2 = 0.;
        uv3 = 0.;
        for (i = 0; i < n_p; i++) {
            uv1 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 3 + 0];
            uv2 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 3 + 1];
            uv3 += C[num_elem * n_p * n + i * num_elem + idx] * basis_vertex[i * 3 + 2];
        }

        // store result
        Uv1[idx] = uv1;
        Uv2[idx] = uv2;
        Uv3[idx] = uv3;
    }
}

/* evaluate error
 * 
 * evaluates rho and E at the three vertex points for output
 * THREADS: num_elem
 */
__global__ void eval_error(double *C, double *error,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       int num_elem, int n_p, int n_quad, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) {
        int i, j;
        double e;
        double x, y;
        double U;

        double v1x, v1y, v2x, v2y, v3x, v3y;
        v1x = V1x[idx];
        v1y = V1y[idx];
        v2x = V2x[idx];
        v2y = V2y[idx];
        v3x = V3x[idx];
        v3y = V3y[idx];

        e = 0;
        for (j = 0; j < n_quad; j++) {
            // get the actual point on the mesh
            x = r1[j] * v2x + r2[j] * v3x + (1 - r1[j] - r2[j]) * v1x;
            y = r1[j] * v2y + r2[j] * v3y + (1 - r1[j] - r2[j]) * v1y;
            
            // evaluate U at the integration point
            U = 0.;
            for (i = 0; i < n_p; i++) {
                U += C[num_elem * n_p * n + i * num_elem + idx] * basis[i * n_quad + j];
            }

            e += w[j] * powf((U0(x, y) - U),2); 
            // evaluate exact conditions at the integration point
            //if (n == 0) {
                //e += w[j] * powf((U0(x, y) - U),2); 
            //} else if (n == 1) {
                //e += w[j] * powf((U1(x, y) - U),2);
            //} else if (n == 2) {
                //e += w[j] * powf((U2(x, y) - U),2);
            //} else if (n == 3) {
                //e += w[j] * powf((U3(x, y) - U),2);
            //}
        }

        // store the result
        error[idx] = e;
    }
}

/* evaluate u velocity
 * 
 * evaluates u and v at the three vertex points for output
 * THREADS: num_elem
 */
__device__ void eval_u_velocity(double *c, double *c_rho,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, int idx) {
    int i;
    double uv1, uv2, uv3;
    double rhov1, rhov2, rhov3;

    // calculate values at the integration points
    rhov1 = 0.;
    rhov2 = 0.;
    rhov3 = 0.;
    for (i = 0; i < n_p; i++) {
        rhov1 += c_rho[i] * basis_vertex[i * 3 + 0];
        rhov2 += c_rho[i] * basis_vertex[i * 3 + 1];
        rhov3 += c_rho[i] * basis_vertex[i * 3 + 2];
    }

    uv1 = 0.;
    uv2 = 0.;
    uv3 = 0.;
    for (i = 0; i < n_p; i++) {
        uv1 += c[i] * basis_vertex[i * 3 + 0];
        uv2 += c[i] * basis_vertex[i * 3 + 1];
        uv3 += c[i] * basis_vertex[i * 3 + 2];
    }

    uv1 = uv1 / rhov1;
    uv2 = uv2 / rhov2;
    uv3 = uv3 / rhov3;

    // store result
    Uv1[idx] = uv1;
    Uv2[idx] = uv2;
    Uv3[idx] = uv3;
}

/* check for convergence
 *
 * see if the difference in coefficients is less than the tolerance
 */
__global__ void check_convergence(double *c_prev, double *c, int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    c_prev[idx] = fabs(c[idx] - c_prev[idx]);
}


/*
__global__ void measure_error(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i;
    double rho, u, v, E;
    double v1x, v1y, v2x, v2y, v3x, v3y;
    double error1, error2, error3;
    double p1, p_exact1;
    double p2, p_exact2;
    double p3, p_exact3;

    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    v1x = V1x[idx];
    v1y = V1y[idx];
    v2x = V2x[idx];
    v2y = V2y[idx];
    v3x = V3x[idx];
    v3y = V3y[idx];

    // vertex 1 (r = 0, s = 0)
    rho = 0.;
    u = 0.;
    v = 0.;
    E = 0.;
    for (i = 0; i < n_p; i++) {
        rho += c[num_elem * n_p * 0 + i * num_elem + idx] * basis_vertex[i * 3 + 0];
        u   += c[num_elem * n_p * 1 + i * num_elem + idx] * basis_vertex[i * 3 + 0];
        v   += c[num_elem * n_p * 2 + i * num_elem + idx] * basis_vertex[i * 3 + 0];
        E   += c[num_elem * n_p * 3 + i * num_elem + idx] * basis_vertex[i * 3 + 0];
    }

    u = u / rho;
    v = v / rho;

    //p1 = pressure(rho, u, v, E);
    //p_exact1 = pressure(rho0(v1x, v1y), u0(v1x, v1y), v0(v1x, v2y), E0(v1x, v1y));
    error1 = powf(p1 - p_exact1, 2);

    // vertex 2 (r = 1, s = 0)
    rho = 0.;
    u = 0.;
    v = 0.;
    E = 0.;
    for (i = 0; i < n_p; i++) {
        rho += c[num_elem * n_p * 0 + i * num_elem + idx] * basis_vertex[i * 3 + 1];
        u   += c[num_elem * n_p * 1 + i * num_elem + idx] * basis_vertex[i * 3 + 1];
        v   += c[num_elem * n_p * 2 + i * num_elem + idx] * basis_vertex[i * 3 + 1];
        E   += c[num_elem * n_p * 3 + i * num_elem + idx] * basis_vertex[i * 3 + 1];
    }

    u = u / rho;
    v = v / rho;

    //p2 = pressure(rho, u, v, E);
    //p_exact2 = pressure(rho0(v2x, v2y), u0(v2x, v2y), v0(v2x, v2y), E0(v2x, v2y));
    error2 = powf(p2 - p_exact2, 2);

     // vertex 3 (r = 0, s = 1)
    rho = 0.;
    u = 0.;
    v = 0.;
    E = 0.;
    for (i = 0; i < n_p; i++) {
        rho += c[num_elem * n_p * 0 + i * num_elem + idx] * basis_vertex[i * 3 + 2];
        u   += c[num_elem * n_p * 1 + i * num_elem + idx] * basis_vertex[i * 3 + 2];
        v   += c[num_elem * n_p * 2 + i * num_elem + idx] * basis_vertex[i * 3 + 2];
        E   += c[num_elem * n_p * 3 + i * num_elem + idx] * basis_vertex[i * 3 + 2];
    }

    u = u / rho;
    v = v / rho;

    //p3 = pressure(rho, u, v, E);
    //p_exact3 = pressure(rho0(v3x, v3y), u0(v3x, v3y), v0(v3x, v3y), E0(v3x, v3y));
    error3 = powf(p3 - p_exact3, 2);

    // store result
    Uv1[idx] = p1;
    Uv2[idx] = p2;
    Uv3[idx] = p3;
}

__global__ void eval_error_L2(double *c,
                       double *error, 
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       int n_quad, int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elem) {
        int i, j;
        double rho;
        double x, y;
        double error_local = 0.;
        for (j = 0; j < n_quad; j++) {
            // evaluate rho at the j'th integration point
            rho = 0.;
            for (i = 0; i < n_p; i++) {
                rho += c[num_elem * n_p * 0 + i * num_elem + idx] * basis[n_quad * i + j];
            }

            // map from the canonical element to the actual point on the mesh
            // x = x2 * r + x3 * s + x1 * (1 - r - s)
            x = r1[j] * V2x[idx] + r2[j] * V3x[idx] + (1 - r1[j] - r2[j]) * V1x[idx];
            y = r1[j] * V2y[idx] + r2[j] * V3y[idx] + (1 - r1[j] - r2[j]) * V1y[idx];

            // evaluate (rho - rho0)^2 at x_j, y_j
            //error_local += w[j] * (rho0(x,y) - rho) * (rho0(x,y) - rho);
        }

        error[idx] = error_local;
    }
}
*/
