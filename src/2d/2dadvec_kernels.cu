/* 2dadvec_kernels.cu
 *
 * This file contains the kernels for the 2D advection DG method.
 */

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
 * evaluates the riemann problem over the boundary.
 */
__device__ riemann(float *c, float *n, float *r1, float *r2, float *w, int k, int N) {
    
}

/***********************
 *
 * PRECOMPUTING
 *
 ***********************/
/* jacobian computing
 *
 * precomputes the jacobian for each element.
 */
__global__ preval_jacobian(float *J) {

}

/* basis element area
 *
 * precomputes all the areas of the basis elements over their 
 * actual mesh element.
 */
__global__ preval_basisarea(float *basisarea) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float area;

    // Use Gaussian quadrature to get the area
    area = 0

    basisarea[idx] = area;
}

/***********************
 *
 * MAIN FUNCTIONS
 *
 ***********************/

/* flux evaluation
 *
 * evaluate all the riemann problems for each element.
 */
__global__ eval_riemann(float *c, int *left_idx_list, int *right_idx_list,
                                  int *side_idx1, int *side_idx2, 
                                  float *Nx, float *Ny, 
                                  float *f, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int left_idx, right_idx;
    float c_left[10], c_right[10];
    float nx, ny, s;

    // Find the left and right elements
    left_idx  = left_idx_list[idx];
    right_idx = left_idx_list[idx];

    // Find where to store the side in the f list
    idx1 = side_idx1[idx];
    idx2 = side_idx2[idx];

    // Get the normal vector for this side
    nx = Nx[idx];
    ny = Ny[idx];

    // Grab the coefficients for the left & right elements
    for (i = 0; i < N; i++) {
        c_left[i]  = u[N*left_idx  + i];
        c_right[i] = u[N*right_idx + i];
    }
    
    // Solve the Riemann problem


    // Write solution into sides
    f[side_idx1] = s;
    f[side_idx2] = s;

}

/* right hand side
 *
 * combines the riemann solution & the volume integral to form the right
 * hand side of the RK problem.
 */
 __global__ eval_rhs(float *c, float *r1, float *r2, float *w, float *J, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float quad_u, f1, f2, f3, register_J, area;
    float register_c[10];

    // Get the coefficients for this element
    for (i = 0; i < N; i++) {
        register_c[i] = c[N*idx + i];
    }
     
    // Grab the Jacobian
    register_J = J[idx];

    // Read in the boundary integrals
    f1 = side1[idx];
    f2 = side2[idx];
    f3 = side3[idx];

    for (i = 0; i < N; i++) {
        // Evaluate the volume integral
        quad_u = quad(register_c, r1, r2, w, register_J, i, N);

        // Get the area of the i'th basis over the element
        area = basisarea[N*idx + i];
    
        // Write the result into the coefficient
        c[N*idx + i] = 1./area*(-quad_u + f1 + f2 + f3);
     }
 }


