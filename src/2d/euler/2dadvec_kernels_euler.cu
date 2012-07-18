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

#define PI 3.14159
#define GAMMA 1.4

/***********************
 *
 * DEVICE VARIABLES
 *
 ***********************/
/* These are always prefixed with d_ for "device" */
float *d_c;                 // coefficients for [rho, u, v, E]
float *d_quad_rhs;          // the right hand side containing the quadrature contributions
float *d_left_riemann_rhs;  // the right hand side containing the left riemann contributions
float *d_right_riemann_rhs; // the right hand side containing the right riemann contributions

// TODO: switch to low storage runge-kutta
// runge kutta variables
float *d_kstar;
float *d_k1;
float *d_k2;
float *d_k3;
float *d_k4;

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
__device__ __constant__ float basis[2048];
// note: these are multiplied by the weights
__device__ __constant__ float basis_grad_x[2048]; 
__device__ __constant__ float basis_grad_y[2048]; 

// precomputed basis functions evaluated along the sides. ordered
// similarly to basis and basis_grad_{x,y} but with one "matrix" for each side
// starting with side 0. to get to each side, offset with:
//      side_number * n_p * num_quad1d.
__device__ __constant__ float basis_side[1024];
__device__ __constant__ float basis_vertex[256];

// weights for 2d and 1d quadrature rules
__device__ __constant__ float w[32];
__device__ __constant__ float w_oned[16];

__device__ __constant__ float r1[32];
__device__ __constant__ float r2[32];
__device__ __constant__ float r_oned[32];

void set_basis(void *value, int size) {
    cudaMemcpyToSymbol("basis", value, size * sizeof(float));
}
void set_basis_grad_x(void *value, int size) {
    cudaMemcpyToSymbol("basis_grad_x", value, size * sizeof(float));
}
void set_basis_grad_y(void *value, int size) {
    cudaMemcpyToSymbol("basis_grad_y", value, size * sizeof(float));
}
void set_basis_side(void *value, int size) {
    cudaMemcpyToSymbol("basis_side", value, size * sizeof(float));
}
void set_basis_vertex(void *value, int size) {
    cudaMemcpyToSymbol("basis_vertex", value, size * sizeof(float));
}
void set_w(void *value, int size) {
    cudaMemcpyToSymbol("w", value, size * sizeof(float));
}
void set_w_oned(void *value, int size) {
    cudaMemcpyToSymbol("w_oned", value, size * sizeof(float));
}
void set_r1(void *value, int size) {
    cudaMemcpyToSymbol("r1", value, size * sizeof(float));
}
void set_r2(void *value, int size) {
    cudaMemcpyToSymbol("r2", value, size * sizeof(float));
}
void set_r_oned(void *value, int size) {
    cudaMemcpyToSymbol("r_oned", value, size * sizeof(float));
}

// tells which side (1, 2, or 3) to evaluate this boundary integral over
int *d_left_side_number;
int *d_right_side_number;

float *d_J;        // jacobian determinant 
float *d_min_J;      // for the min sized jacobian
float *d_s_length; // length of sides

// the num_elem values of the x and y coordinates for the two vertices defining a side
// TODO: can i delete these after the lengths are precomputed?
//       maybe these should be in texture memory?
float *d_s_V1x;
float *d_s_V1y;
float *d_s_V2x;
float *d_s_V2y;

// the num_elem values of the x and y partials
float *d_xr;
float *d_yr;
float *d_xs;
float *d_ys;

// the K indices of the sides for each element ranged 0->H-1
int *d_elem_s1;
int *d_elem_s2;
int *d_elem_s3;

// vertex x and y coordinates on the mesh which define an element
// TODO: can i delete these after the jacobians are precomputed?
//       maybe these should be in texture memory?
float *d_V1x;
float *d_V1y;
float *d_V2x;
float *d_V2y;
float *d_V3x;
float *d_V3y;

// stores computed values at three vertices
float *d_Uv1;
float *d_Uv2;
float *d_Uv3;

// normal vectors for the sides
float *d_Nx;
float *d_Ny;

// index lists for sides
int *d_left_elem;  // index of left  element for side idx
int *d_right_elem; // index of right element for side idx

/***********************
 *
 * DEVICE FUNCTIONS
 *
 ***********************/
/* riemann solver
 *
 * evaluates the riemann problem over the boundary using Gaussian quadrature
 * with Legendre polynomials as basis functions.
 */
__device__ float riemann(float u_left, float u_right) {
    return 0.5 * (u_left + u_right);
}

__device__ float pressure(float rho, float u, float v, float E) {
    return (GAMMA - 1) * (E - (u*u + v*v) / 2 * rho);
}

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x
 */
__device__ float rho0(float x, float y) {
    return x;
}
__device__ float u0(float x, float y) {
    return 0.;
}
__device__ float v0(float x, float y) {
    return 1.;
}
__device__ float E0(float x, float y) {
    return 0.;
}

/* boundary exact
 *
 * returns the exact boundary conditions
 */
__device__ float boundary_exact_rho(float x, float y, float t) {
    return 0;
}
__device__ float boundary_exact_u(float x, float y, float t) {
    return 0;
}
__device__ float boundary_exact_v(float x, float y, float t) {
    return 1;
}
__device__ float boundary_exact_E(float x, float y, float t) {
    return 0;
}

/* u exact
 *
 * returns the exact value of u for error measurement.
 */
__device__ float uexact(float x, float y, float t) {
    return u0(x, y);
}

/* initial conditions
 *
 * computes the coefficients for the initial conditions
 * THREADS: num_elem
 */
__global__ void init_conditions(float *c, float *J,
                                float *V1x, float *V1y,
                                float *V2x, float *V2y,
                                float *V3x, float *V3y,
                                int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;
    float x, y, rho, u, v, E;

    if (idx < num_elem) {
        for (i = 0; i < n_p; i++) {
            u = 0.;
            // perform quadrature
            for (j = 0; j < n_quad; j++) {
                // map from the canonical element to the actual point on the mesh
                // x = x2 * r + x3 * s + x1 * (1 - r - s)
                x = r1[j] * V2x[idx] + r2[j] * V3x[idx] + (1 - r1[j] - r2[j]) * V1x[idx];
                y = r1[j] * V2y[idx] + r2[j] * V3y[idx] + (1 - r1[j] - r2[j]) * V1y[idx];

                // evaluate rho, u, v, E there
                rho += w[j] * rho0(x, y) * basis[i * n_quad + j];
                u   += w[j] * u0(x, y)   * basis[i * n_quad + j];
                v   += w[j] * v0(x, y)   * basis[i * n_quad + j];
                E   += w[j] * E0(x, y)   * basis[i * n_quad + j];
            }

            c[num_elem * n_p * 0 + i * num_elem + idx] = rho;
            c[num_elem * n_p * 1 + i * num_elem + idx] = u;
            c[num_elem * n_p * 2 + i * num_elem + idx] = v;
            c[num_elem * n_p * 3 + i * num_elem + idx] = E;
        } 
    }
}

/* find min jacobian
 *
 * returns the min jacobian inside of min_J. 
 * each block computes the min jacobian inside of that block and stores it in the
 * blockIdx.x spot of the shared min_J variable.
 * NOTE: this is fixed for 256 threads.
 */
__global__ void min_jacobian(float *J, float *min_J, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    int i   = blockDim.x * (blockIdx.x * 256 * 2) + threadIdx.x;

    __shared__ float s_min[256];

    if (idx < num_elem) {
        // set all of min to J[idx] initially
        s_min[tid] = J[idx];
        __syncthreads();

        // test a few
        while (i < num_elem) {
            s_min[tid] = (s_min[tid] < s_min[i]) ? s_min[tid] : s_min[i];
            s_min[tid] = (s_min[tid] < s_min[i + 256]) ? s_min[tid] : s_min[i];
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
            min_J[blockIdx.x] = s_min[0];
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
__global__ void preval_side_length(float *s_length, 
                              float *s_V1x, float *s_V1y, 
                              float *s_V2x, float *s_V2y,
                              int num_sides) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        // compute and store the length of the side
        s_length[idx] = sqrtf(pow(s_V1x[idx] - s_V2x[idx],2) + pow(s_V1y[idx] - s_V2y[idx],2));
    }
}

/* jacobian computing
 *
 * precomputes the jacobian determinant for each element.
 * THREADS: num_elem
 */
__global__ void preval_jacobian(float *J, 
                           float *V1x, float *V1y, 
                           float *V2x, float *V2y, 
                           float *V3x, float *V3y,
                           int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float x1, y1, x2, y2, x3, y3;

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
__global__ void preval_normals(float *Nx, float *Ny, 
                          float *s_V1x, float *s_V1y, 
                          float *s_V2x, float *s_V2y,
                          float *V1x, float *V1y, 
                          float *V2x, float *V2y, 
                          float *V3x, float *V3y,
                          int *left_side_number, int num_sides) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        float x, y, length;
        float sv1x, sv1y, sv2x, sv2y;
    
        sv1x = s_V1x[idx];
        sv1y = s_V1y[idx];
        sv2x = s_V2x[idx];
        sv2y = s_V2y[idx];
    
        // lengths of the vector components
        x = sv2x - sv1x;
        y = sv2y - sv1y;
    
        // normalize
        length = sqrtf(pow(x, 2) + pow(y, 2));

        // store the result
        Nx[idx] = -y / length;
        Ny[idx] =  x / length;
    }
}

__global__ void preval_normals_direction(float *Nx, float *Ny, 
                          float *V1x, float *V1y, 
                          float *V2x, float *V2y, 
                          float *V3x, float *V3y,
                          int *left_elem, int *left_side_number, int num_sides) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        float new_x, new_y, dot;
        float initial_x, initial_y, target_x, target_y;
        float x, y;
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

__global__ void preval_partials(float *V1x, float *V1y,
                                float *V2x, float *V2y,
                                float *V3x, float *V3y,
                                float *xr,  float *yr,
                                float *xs,  float *ys, int num_elem) {
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

/* riemann evaluation
 *
 * device function to solve the riemann problem.
 */
__device__ void eval_riemann(float *c_rho_left, float *c_rho_right,
                              float *c_u_left,   float *c_u_right,
                              float *c_v_left,   float *c_v_right,
                              float *c_E_left,   float *c_E_right,
                              float v1x, float v1y,
                              float v2x, float v2y,
                              float v3x, float v3y,
                              int j, // j, as usual, is the index of the integration point
                              int left_side, int right_side,
                              int left_idx, int right_idx,
                              int n_p, int n_quad1d,
                              int num_sides, float t, 
                              float *rho, float *u, float *v, float *E) {

    int i;

    float rho_left, u_left, v_left, E_left;
    float rho_right, u_right, v_right, E_right;

    rho_left = 0.;
    u_left   = 0.;
    v_left   = 0.;
    E_left   = 0.;
    rho_right = 0.;
    u_right   = 0.;
    v_right   = 0.;
    E_right   = 0.;
    
    for (i = 0; i < n_p; i++) {
        rho_left += c_rho_left[i] * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
        u_left   += c_u_left[i]   * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
        v_left   += c_v_left[i]   * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
        E_left   += c_E_left[i]   * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
    }

    // make all threads in the first warps be boundary sides
    if (right_idx == -1) {
        float r1_eval, r2_eval;
        float x, y;

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
            
        // deal with the boundary element here
        rho_right = boundary_exact_rho(x, y, t);
        u_right   = boundary_exact_u(x, y, t);
        v_right   = boundary_exact_v(x, y, t);
        E_right   = boundary_exact_E(x, y, t);

    } else {
        // evaluate the right side at the integration point
        for (i = 0; i < n_p; i++) {
            rho_right += c_rho_right[i] * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            u_right   += c_u_right[i]   * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            v_right   += c_v_right[i]   * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            E_right   += c_E_right[i]   * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
        }
    }

    *rho =  riemann(rho_left, rho_right);
    *u   =  riemann(rho_left, rho_right);
    *v   =  riemann(rho_left, rho_right);
    *E   =  riemann(rho_left, rho_right);
}

/* surface integral evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: num_sides
 */
/*
 * d_t [   rho   ] + d_x [     rho * u    ] + d_y [    rho * v     ] = 0
 * d_t [ rho * u ] + d_x [ rho * u^2 + p  ] + d_y [   rho * u * v  ] = 0
 * d_t [ rho * v ] + d_x [  rho * u * v   ] + d_y [  rho * v^2 + p ] = 0
 * d_t [    E    ] + d_x [ u * ( E +  p ) ] + d_y [ v * ( E +  p ) ] = 0
 */

__device__ void eval_flux(float rho, float u, float v, float E, 
                     float *flux_x1, float *flux_y1,
                     float *flux_x2, float *flux_y2,
                     float *flux_x3, float *flux_y3,
                     float *flux_x4, float *flux_y4) {
    // evaluate pressure

    float p = pressure(rho, u, v, E);

    // flux_1 
    *flux_x1 = rho * u;
    *flux_y1 = rho * v;

    // flux_2
    *flux_x2 = rho * u * u + p;
    *flux_y2 = rho * u * v;

    // flux_3
    *flux_x3 = rho * u * v;
    *flux_y3 = rho * v * v + p;

    // flux_4
    *flux_x4 = u * (E + p);
    *flux_y4 = v * (E + p);
}

__device__ void eval_surface(float *rho_left, float *u_left, float *v_left, float *E_left,
                             float *rho_right, float *u_right, float *v_right, float *E_right,
                             float *left_riemann_rhs, float *right_riemann_rhs, 
                             float len,
                             float v1x, float v1y,
                             float v2x, float v2y,
                             float v3x, float v3y,
                             int left_idx,  int right_idx,
                             int left_side, int right_side, 
                             float nx, float ny, 
                             int n_quad1d, int n_p, int num_sides, 
                             int num_elem, float t, int idx) {
    int i, j;
    float rho, u, v, E;
    float left_sum1, right_sum1, flux_x1, flux_y1;
    float left_sum2, right_sum2, flux_x2, flux_y2;
    float left_sum3, right_sum3, flux_x3, flux_y3;
    float left_sum4, right_sum4, flux_x4, flux_y4;

    // multiply across by the i'th basis function
    for (i = 0; i < n_p; i++) {

        left_sum1  = 0.;
        left_sum2  = 0.;
        left_sum3  = 0.;
        left_sum4  = 0.;
        right_sum1 = 0.;
        right_sum2 = 0.;
        right_sum3 = 0.;
        right_sum4 = 0.;

        for (j = 0; j < n_quad1d; j++) {

            // calculate the riemann problems
            eval_riemann(rho_left, rho_right,
                         u_left,   u_right,
                         v_left,   v_right,
                         E_left,   E_right,
                         v1x, v1y, v2x, v2y, v3x, v3y,
                         j, left_side, right_side,
                         left_idx, right_idx,
                         n_p, n_quad1d, num_sides, t,
                         &rho, &u, &v, &E);

            // calculate the fluxes
            eval_flux(rho, u, v, E,
                 &flux_x1, &flux_y1, &flux_x2, &flux_y2,
                 &flux_x3, &flux_y3, &flux_x4, &flux_y4);

            // 1st row
            left_sum1  += (nx * flux_x1 + ny * flux_y1) * w_oned[j] * 
                           basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            right_sum1 += (nx * flux_x1 + ny * flux_y1) * w_oned[j] * 
                           basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];
            // 2nd row
            left_sum2  += (nx * flux_x2 + ny * flux_y2) * w_oned[j] * 
                           basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            right_sum2 += (nx * flux_x2 + ny * flux_y2) * w_oned[j] * 
                           basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];

            // 3rd row
            left_sum3  += (nx * flux_x3 + ny * flux_y3) * w_oned[j] * 
                           basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            right_sum3 += (nx * flux_x3 + ny * flux_y3) * w_oned[j] * 
                           basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];

            // 4th row
            left_sum4  += (nx * flux_x4 + ny * flux_y4) * w_oned[j] * 
                           basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            right_sum4 += (nx * flux_x4 + ny * flux_y4) * w_oned[j] * 
                           basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];
        }

        // store this side's contribution in the riemann rhs vectors
        left_riemann_rhs[num_elem * n_p * 0 + i * num_sides + idx]  = -len / 2 * left_sum1;
        left_riemann_rhs[num_elem * n_p * 1 + i * num_sides + idx]  = -len / 2 * left_sum2;
        left_riemann_rhs[num_elem * n_p * 2 + i * num_sides + idx]  = -len / 2 * left_sum3;
        left_riemann_rhs[num_elem * n_p * 3 + i * num_sides + idx]  = -len / 2 * left_sum4;
        right_riemann_rhs[num_elem * n_p * 0 + i * num_sides + idx] =  len / 2 * right_sum1;
        right_riemann_rhs[num_elem * n_p * 1 + i * num_sides + idx] =  len / 2 * right_sum2;
        right_riemann_rhs[num_elem * n_p * 2 + i * num_sides + idx] =  len / 2 * right_sum3;
        right_riemann_rhs[num_elem * n_p * 3 + i * num_sides + idx] =  len / 2 * right_sum4;
    }
}
/* flux boundary evaluation 
 *
 * evaulates the flux at the boundaries by handling them somehow.
 * THREADS: num_boundary
 */

/* volume integrals
 *
 * evaluates and adds the volume integral to the rhs vector
 * THREADS: num_elem
 */
__device__ void eval_volume(float *c_rho, float *c_u,
                            float *c_v,   float *c_E,
                            float *quad_rhs, 
                            float x_r, float y_r,
                            float x_s, float y_s,
                            int n_quad, int n_p, int num_elem, int idx) {
    int i, j, k;
    float rho, u, v, E;
    float flux_x1, flux_y1, flux_x2, flux_y2;
    float flux_x3, flux_y3, flux_x4, flux_y4;
    float sum1, sum2, sum3, sum4;

    // evaluate the volume integral for each coefficient
    for (i = 0; i < n_p; i++) {
        sum1 = 0.;
        sum2 = 0.;
        sum3 = 0.;
        sum4 = 0.;
        for (j = 0; j < n_quad; j++) {

            // evaluate rho, u, v, E at the integration point.
            rho = 0.;
            u   = 0.;
            v   = 0.;
            E   = 0.;
            for (k = 0; k < n_p; k++) {
                rho += c_rho[k] * basis[n_quad * k + j];
                u   += c_u[k]   * basis[n_quad * k + j];
                v   += c_v[k]   * basis[n_quad * k + j];
                E   += c_E[k]   * basis[n_quad * k + j];
            }

            // evaluate flux
            eval_flux(rho, u, v, E,
                 &flux_x1, &flux_y1, &flux_x2, &flux_y2,
                 &flux_x3, &flux_y3, &flux_x4, &flux_y4);
                 
            // Add to the sum
            // [fx fy] * [y_s, -y_r; -x_s, x_r] * [phi_x phi_y]
            sum1 += (  flux_x1 * ( basis_grad_x[n_quad * i + j] * y_s
                                  -basis_grad_y[n_quad * i + j] * y_r)
                     + flux_y1 * (-basis_grad_x[n_quad * i + j] * x_s 
                                 + basis_grad_y[n_quad * i + j] * x_r));
            sum2 += (  flux_x2 * ( basis_grad_x[n_quad * i + j] * y_s
                                  -basis_grad_y[n_quad * i + j] * y_r)
                     + flux_y2 * (-basis_grad_x[n_quad * i + j] * x_s 
                                 + basis_grad_y[n_quad * i + j] * x_r));
            sum3 += (  flux_x3 * ( basis_grad_x[n_quad * i + j] * y_s
                                  -basis_grad_y[n_quad * i + j] * y_r)
                     + flux_y3 * (-basis_grad_x[n_quad * i + j] * x_s 
                                 + basis_grad_y[n_quad * i + j] * x_r));
            sum4 += (  flux_x4 * ( basis_grad_x[n_quad * i + j] * y_s
                                  -basis_grad_y[n_quad * i + j] * y_r)
                     + flux_y4 * (-basis_grad_x[n_quad * i + j] * x_s 
                                 + basis_grad_y[n_quad * i + j] * x_r));
        }

        // store the result
        quad_rhs[num_elem * n_p * 0 + i * num_elem + idx] = sum1;
        quad_rhs[num_elem * n_p * 1 + i * num_elem + idx] = sum2;
        quad_rhs[num_elem * n_p * 2 + i * num_elem + idx] = sum3;
        quad_rhs[num_elem * n_p * 3 + i * num_elem + idx] = sum4;
    }
}

/* evaluate error
 * 
 * evaluates u at the three vertex points for output
 * THREADS: num_elem
 */
__device__ void eval_error(float *c, 
                       float v1x, float v1y,
                       float v2x, float v2y,
                       float v3x, float v3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int idx) {

    int i;
    float uv1, uv2, uv3;

    // calculate values at three vertex points
    uv1 = 0.;
    uv2 = 0.;
    uv3 = 0.;
    for (i = 0; i < n_p; i++) {
        uv1 += c[i] * basis_vertex[i * 3 + 0];
        uv2 += c[i] * basis_vertex[i * 3 + 1];
        uv3 += c[i] * basis_vertex[i * 3 + 2];
    }

    // store result
    Uv1[idx] = uv1 - uexact(v1x, v1y, t);
    Uv2[idx] = uv2 - uexact(v2x, v2y, t);
    Uv3[idx] = uv3 - uexact(v3x, v3y, t);
}

/* evaluate u
 * 
 * evaluates u at the three vertex points for output
 * THREADS: num_elem
 */
__device__ void eval_u(float *c, 
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, int idx) {
    int i;
    float uv1, uv2, uv3;

    // calculate values at the integration points
    uv1 = 0.;
    uv2 = 0.;
    uv3 = 0.;
    for (i = 0; i < n_p; i++) {
        uv1 += c[i] * basis_vertex[i * 3 + 0];
        uv2 += c[i] * basis_vertex[i * 3 + 1];
        uv3 += c[i] * basis_vertex[i * 3 + 2];
    }

    // store result
    Uv1[idx] = uv1;
    Uv2[idx] = uv2;
    Uv3[idx] = uv3;
}
