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
float *d_c;     // holds coefficients for each element
float *d_r1;    // integration points (x)
float *d_r2;    // integration points (y)
float *d_w;     // gaussian integration weights
float *d_J;     // jacobian determinant 
float *d_s_len; // length of sides

// the H values of the x and y coordinates for the two vertices defining a side
float *d_s_V1x;
float *d_s_V1y;
float *d_s_V2x;
float *d_s_V2y;

// the K indices of the sides for each element ranged 0->H-1
int *d_elem_s1;
int *d_elem_s2;
int *d_elem_s3;

// vertex x and y coordinates on the mesh
float *d_V1x;
float *d_V1y;
float *d_V2x;
float *d_V2y;
float *d_V3x;
float *d_V3y;

// normal vectors
float *d_Nx;
float *d_Ny;

// index lists for sides
int *d_right_idx_list; // index of right element for side idx
int *d_left_idx_list;  // index of left element for side idx


/***********************
 *
 * DEVICE FUNCTIONS
 *
 ***********************/

/* flux function
 *
 * evaluates the flux f(u) at the point u.
 */
__device__ float flux(float u) {
    return u
}

/* quadrature 
 *
 * uses gaussian quadrature to evaluate the integral over the 
 * element k. takes the coefficients for u_k in c, the integration 
 * points and weights r1, r2 and w, and the jacobian J.
 */
__device__ float quad(float *c, float *r1, float *r2, float *w, float J, int k, int N) {
    int i, j;
    float sum, u;
    
    sum = 0;
    for (i = 0; i < N; i++) {
        // Evaluate u at the integration point.
        u = 0;
        for (j = 0; j < N; j++) {
            u += basis(r1[i], r2[i], j);
        }
        // Add to the sum
        sum += flux(u) * d_basis(r1[i], r2[i], k) * w[i];
    }

    // Multiply in the Jacobian
    return J * sum;
}

/* boundary elements
 *
 * does something to handle the boundary elements.
 */
__device__ boundary(float *c, int k, int N) {

}

/* riemann solver
 *
 * evaluates the riemann problem over the boundary using Gaussian quadrature
 * with Legendre polynomials as basis functions.
 */
__device__ riemann(float *eval_left, float *eval_right, // the left and right evaluations at the integration points
                   float *n, float *r1, float *r2, float *w, int k, int N) {
        
}

/***********************
 *
 * PRECOMPUTING
 *
 ***********************/
/* side lenght computer
 *
 * precomputes the length of each side.
 * THREADS: K
 */ 
__global__ preval_side_length(float *s_length, float *s_V1x, float *s_V1y, float *s_V2x, float *s_V2y) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // compute and store the length of the side
    s_length[idx] = sqrtf(powf(s_V1x[idx] - s_V2x[idx],2) + powf(s_V1y[idx] - sV2y[idx],2));
}

/* jacobian computing
 *
 * precomputes the jacobian determinant for each element.
 * THREADS: K
 */
__global__ preval_jacobian(float *J, float *V1x, float *V1y, float *V2x, float *V2y, float *V3x, float *V3y) {
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
 */
__global__ preval_normals(float *Nx, float *Ny, float *s_V1x, float *s_V1y, float *s_V2x, float *s_V2y) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;

   Nx[idx] = 
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
__global__ eval_riemann(float *c, int *left_idx_list, int *right_idx_list,
                                  int *side_idx_1, int *side_idx_2, 
                                  float *Nx, float *Ny, 
                                  float *f, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int left_idx, right_idx, side;
    float c_left[10], c_right[10];
    float nx, ny, s;

    // Find the left and right elements
    left_idx  = left_idx_list[idx];
    right_idx = left_idx_list[idx];

    // Find where to store the side in the f list
    idx1 = side_idx_1[idx];
    idx2 = side_idx_2[idx];

    // Get the normal vector for this side
    nx = Nx[idx];
    ny = Ny[idx];

    // Grab the coefficients for the left & right elements
    for (i = 0; i < N; i++) {
        c_left[i]  = u[i*left_idx  + idx];
        c_right[i] = u[i*right_idx + idx];
    }
    
    // Need to find out what side we've got
    side = side_number[idx];
    
    // Evaluate the polynomial over that side for both elements
    // TODO: Order the basis function evaluations (and coefficients) 
    //       so that we can grab their values along the edges using some scheme
    switch (side) {
        case 1:
            eval_left  = c_left[0]  * eval_basis[0]; // + ...           
            eval_right = c_right[0] * eval_basis[0]; // + ...           
            break;
        case 2:
            eval_left  = c_left[1]  * eval_basis[1]; // + ...           
            eval_right = c_right[1] * eval_basis[1]; // + ...           
            break;
        case 3:
            eval_left  = c_left[2]  * eval_basis[2]; // + ...           
            eval_right = c_right[2] * eval_basis[2]; // + ...           
            break;
    }

    // Solve the Riemann problem

    // Write solution into sides
    f[side_idx_1] = s;
    f[side_idx_2] = s;

}

/* right hand side
 *
 * combines the riemann solution & the volume integral to form the right
 * hand side of the RK problem.
 * THREADS: K
 */
 __global__ eval_rhs(float *c, float *f, float *r1, float *r2, float *w, float *J, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float quad_u, f1, f2, f3, register_J;
    float register_c[10];

    // Get the coefficients for this element
    for (i = 0; i < N; i++) {
        register_c[i] = c[i*N + idx];
    }
     
    // Grab the Jacobian
    register_J = J[idx];

    // Read in the boundary integrals
    f1 = f[0 * n_sides + idx];
    f2 = f[1 * n_sides + idx];
    f3 = f[2 * n_sides + idx];

    for (i = 0; i < N; i++) {
        // Evaluate the volume integral
        quad_u = quad(register_c, r1, r2, w, register_J, i, N);

        // Write the result into the coefficient
        c[N*idx + i] = 1./register_J*(-quad_u + f1 + f2 + f3);
     }
 }


