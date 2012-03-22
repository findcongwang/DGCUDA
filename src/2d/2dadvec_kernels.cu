/* 2dadvec_kernels.cu
 *
 * This file contains the kernels for the 2D advection DG method.
 * We use K = number of elements
 * and    H = number of sides
 */

/***********************
 *
 * DEVICE VARIABLES
 *
 ***********************/
/* These are always prefixed with d_ for "device" */
float *d_c;      // holds coefficients for each element
float *d_rhs;    // the right hand side: 
                 //     we reset to 0 between each and add to this to build d_c
                 //     coefficients for each time step.

float *d_r1;     // integration points (x) for 2d integration
float *d_r2;     // integration points (y) for 2d integration
float *d_w;      // weights for 2d integration
float *d_oned_r; // integration points (x) for 1d integration
float *d_oned_w; // weights for 2d integration

float *d_J;     // jacobian determinant 
float *d_s_len; // length of sides

// the H values of the x and y coordinates for the two vertices defining a side
// TODO: can i delete these after the lengths are precomputed?
//       maybe these should be in texture memory?
float *d_s_V1x;
float *d_s_V1y;
float *d_s_V2x;
float *d_s_V2y;

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

// normal vectors for the sides
float *d_Nx;
float *d_Ny;

// index lists for sides
int *d_right_idx_list; // index of right element for side idx
int *d_left_idx_list;  // index of left  element for side idx


/***********************
 *
 * DEVICE FUNCTIONS
 *
 ***********************/

/* flux function
 *
 * evaluates the flux f(u) at the point u.
 */
__device__ float flux_x(float u) {
    return u;
}
__device__ float flux_y(float u) {
    return u;
}

/* 2-d basis functions
 *
 */
__device__ float basis(float x, float y, int i) {
    return x+y;
}

/* 1-d basis functions
 *
 */
__device__ float oned_basis(float x, int i) {
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
    return -1.;
}

/* basis function gradients
 *
 */
__device__ float grad_basis_x(float x, float y, int i) {
    return x;
}
__device__ float grad_basis_y(float x, float y, int i) {
    return y;
}

/* quadrature 
 *
 * uses gaussian quadrature to evaluate the integral over the 
 * element k. takes the coefficients for u_k in c, the integration 
 * points and weights r1, r2 and w, and the jacobian J.
 */
__device__ float quad(float *c, float *r1, float *r2, float *w, float J, int idx, int k, int N) {
    int i, j;
    float sum, u;
    register float register_c[10];

    for (i = 0; i < N; i++) {
        register_c[i] = c[i*N + idx];
    }

    sum = 0.0;
    for (i = 0; i < N; i++) {
        // Evaluate u at the integration point.
        u = 0;
        for (j = 0; j < N; j++) {
            u += register_c[j] * basis(r1[i], r2[i], j);
        }
        // Add to the sum
        sum += w[i] * (  flux_x(u) * grad_basis_x(r1[i], r2[i], k) 
                       + flux_y(u) * grad_basis_y(r1[i], r2[i], k));
    }

    // Multiply in the Jacobian
    return J * sum;
}

/* boundary elements
 *
 * does something to handle the boundary elements.
 */
__device__ float boundary(float *c, int k, int N) {
    return 0;
}

/* riemann solver
 *
 * evaluates the riemann problem over the boundary using Gaussian quadrature
 * with Legendre polynomials as basis functions.
 */
__device__ float riemann(float u_left, float u_right) {
    return 0.5 * (u_left + u_right);
}

/***********************
 *
 * PRECOMPUTING
 *
 ***********************/
/* side length computer
 *
 * precomputes the length of each side.
 * THREADS: K
 */ 
__global__ void preval_side_length(float *s_length, 
                              float *s_V1x, float *s_V1y, 
                              float *s_V2x, float *s_V2y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // compute and store the length of the side
    s_length[idx] = sqrtf(powf(s_V1x[idx] - s_V2x[idx],2) + powf(s_V1y[idx] - s_V2y[idx],2));
}

/* jacobian computing
 *
 * precomputes the jacobian determinant for each element.
 * THREADS: K
 */
__global__ void preval_jacobian(float *J, 
                           float *V1x, float *V1y, 
                           float *V2x, float *V2y, 
                           float *V3x, float *V3y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float xr, xs, yr, ys;
    float x1, y1, x2, y2, x3, y3;

    // read vertex points
    x1 = V1x[idx];
    y1 = V1y[idx];
    x2 = V2x[idx];
    y2 = V2y[idx];
    x3 = V3x[idx];
    y3 = V3y[idx];

    // calculate mappings for r
    xr = 0.5 * (x2 - x1);
    yr = 0.5 * (y2 - y1);

    // calculate mappings for s
    xs = 0.5 * (x3 - x1);
    ys = 0.5 * (y3 - y1);

    // calculate jacobian determinant
    J[idx] = xr * ys - xs * yr;
}

/* evaluate normal vectors
 *
 * computes the normal vectors for each element along each side.
 * THREADS: H
 * TODO: what the hell direction does this point? somehow i need to always
 *       make them point out of the cell, so... remember somehow?
 */
__global__ void preval_normals(float *Nx, float *Ny, 
                          float *s_V1x, float *s_V1y, 
                          float *s_V2x, float *s_V2y) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   float x, y, length;

   // lengths of the vector components
   x = s_V1x[idx] - s_V2x[idx];
   y = s_V1y[idx] - s_V2y[idx];

   // normalize and store
   length = sqrtf(powf(x, 2) + powf(y, 2));
   Nx[idx] = x / length;
   Ny[idx] = y / length;
}

/***********************
 *
 * MAIN FUNCTIONS
 *
 ***********************/


/* flux evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: H
 */
__global__ void eval_riemann(float *c, float *rhs,
                        float *s1_r1, float *s1_r2,
                        float *s2_r1, float *s2_r2,
                        float *s3_r1, float *s3_r2,
                        float *w, float *oned_r, float *oned_w,
                        int *left_idx_list, int *right_idx_list,
                        int *side_number, 
                        float *Nx, float *Ny, float *f, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int left_idx, right_idx, side, i, j;
    float c_left[10], c_right[10], u[10];
    float nx, ny, s;
    float u_left, u_right;
    float sum;

    // Find the left and right elements
    left_idx  = left_idx_list[idx];
    right_idx = right_idx_list[idx];

    // Get the normal vector for this side
    nx = Nx[idx];
    ny = Ny[idx];

    // Grab the coefficients for the left & right elements
    for (i = 0; i < N; i++) {
        c_left[i]  = u[i*N + left_idx];
        c_right[i] = u[i*N + right_idx];
    }
     
    // Need to find out what side we've got for evaluation (right, left, bottom)
    side = side_number[idx];
     
    // Evaluate the polynomial over that side for both elements
    // TODO: Order the basis function evaluations (and coefficients) 
    //       so that we can grab their values along the edges using some scheme
    switch (side) {
        case 1:
            for (i = 0; i < N; i++) {
                // evaluate u at the integration points
                u_left  = 0;
                u_right = 0;
                for (j = 0; j < N; j++) {
                    u_left  += c_left[i]  * basis(s1_r1[i], s1_r2[i], j) * w[i];
                    u_right += c_right[i] * basis(s1_r1[i], s1_r2[i], j) * w[i];
                }
                sum = 0;
                for (j = 0; j < N; j++) {
                    // solve the Riemann problem at this integration point
                    s = riemann(u_left, u_right);
                sum += (nx * flux_x(s) + ny * flux_y(s) * oned_w[j]) * (oned_basis(oned_r[j], i));
            }
                // add each side's contribution to the rhs vector
                rhs[i*N + left_idx]  += sum;
                rhs[i*N + right_idx] += sum;
            }
            break;
        case 2:
             for (i = 0; i < N; i++) {
                // evaluate u at the integration points
                u_left  = 0;
                u_right = 0;
                for (j = 0; j < N; j++) {
                    u_left  += c_left[i]  * basis(s2_r1[i], s2_r2[i], j);
                    u_right += c_right[i] * basis(s2_r1[i], s2_r2[i], j);
                }
                sum = 0;
                for (j = 0; j < N; j++) {
                    // solve the Riemann problem at this integration point
                    s = riemann(u_left, u_right);
                    sum += (nx * flux_x(s) + ny * flux_y(s) * oned_w[j]) * (oned_basis(oned_r[j], i));
                }
                // add each side's contribution to the rhs vector
                rhs[i*N + left_idx]  += sum;
                rhs[i*N + right_idx] += sum;
            }
            break;
        case 3:
            for (i = 0; i < N; i++) {
                // evaluate u at the integration points
                u_left  = 0;
                u_right = 0;
                for (j = 0; j < N; j++) {
                    u_left  += c_left[i]  * basis(s3_r1[i], s3_r2[i], j);
                    u_right += c_right[i] * basis(s3_r1[i], s3_r2[i], j);
                }
                sum = 0;
                for (j = 0; j < N; j++) {
                    // solve the Riemann problem at this integration point
                    s = riemann(u_left, u_right);
                    sum += (nx * flux_x(s) + ny * flux_y(s) * oned_w[j]) * (oned_basis(oned_r[j], i));
                }
                // add each side's contribution to the rhs vector
                rhs[i*N + left_idx]  += sum;
                rhs[i*N + right_idx] += sum;
            }
            break;
     }

    sum = 0;
    // evaluate for each coefficient
    for (i = 0; i < N; i++) {
    }
}

/* volume integrals
 *
 * evaluates and adds the volume integral to the rhs vector
 * THREADS: K
 */
 __global__ void eval_quad(float *c, float *rhs, 
                     int *elem_s1, int *elem_s2, int *elem_s3, 
                     float *r1, float *r2, float *w, float *J, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i;
    float quad_u, register_J;
    float register_c[10];

    // Get the coefficients for this element
    for (i = 0; i < N; i++) {
        register_c[i] = c[i*N + idx];
    }
     
    // Grab the Jacobian
    register_J = J[idx];

    for (i = 0; i < N; i++) {
        // Evaluate the volume integral
        quad_u = quad(register_c, r1, r2, w, register_J, idx, i, N);

        // add the volume contribution result to the rhs
        rhs[i*N + idx] += 1./register_J*(-quad_u);
    }
}

/* right hand side
 *
 */
__global__ void eval_rhs(float *c, float *rhs) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    c[idx]   = rhs[idx];
    rhs[idx] = 0;
}

