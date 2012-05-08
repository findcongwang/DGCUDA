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
float *d_c;           // holds coefficients for each element
float *d_quad_rhs;    // the right hand side containing the quadrature contributions
float *d_left_riemann_rhs;  // the right hand side containing the left riemann contributions
float *d_right_riemann_rhs; // the right hand side containing the right riemann contributions

// runge kutta variables
float *d_kstar;
float *d_k1;
float *d_k2;
float *d_k3;
float *d_k4;

float *d_r1;     // integration points (x) for 2d integration
float *d_r2;     // integration points (y) for 2d integration
float *d_w;      // weights for 2d integration
float *d_oned_w; // weights for 2d integration

// evaluation points for the boundary integrals depending on the side
float *d_s1_r1;
float *d_s1_r2;
float *d_s2_r1;
float *d_s2_r2;
float *d_s3_r1;
float *d_s3_r2;

// tells which side (1, 2, or 3) to evaluate this boundary integral over
int *d_left_side_number;
int *d_right_side_number;

float *d_J;     // jacobian determinant 
float *d_s_length; // length of sides

// the num_elem values of the x and y coordinates for the two vertices defining a side
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
int *d_left_elem;  // index of left  element for side idx
int *d_right_elem; // index of right element for side idx

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

/* basis functions
 *
 */
__device__ float basis(float x, float y, int i) {
    switch (i) {
        case 0: return 1.41421356;
    }
    return -1;
}

/* basis function gradients
 *
 */
__device__ float grad_basis_x(float x, float y, int i) {
    switch (i) {
        case 0: return 0;
    }
    return -1;
}
__device__ float grad_basis_y(float x, float y, int i) {
    switch (i) {
        case 0: return 0;
    }
    return -1;
}

/* quadrature 
 *
 * uses gaussian quadrature to evaluate the integral over the 
 * element k. takes the coefficients for u_k in c, the integration 
 * points and weights r1, r2 and w, and the jacobian J.
 */
__device__ float quad(float *c, float *r1, float *r2, float *w, float J, 
                      int k, int n_quad, int n_p) {
    int i, j;
    float sum, u;

    sum = 0.0;
    for (i = 0; i < n_quad; i++) {
        // Evaluate u at the integration point.
        u = 0;
        for (j = 0; j < n_p + 1; j++) {
            u += c[j] * basis(r1[i], r2[i], j);
        }
        // Add to the sum
        sum += w[i] * (  flux_x(u) * grad_basis_x(r1[i], r2[i], k) 
                       + flux_y(u) * grad_basis_y(r1[i], r2[i], k));
    }

    // Multiply in the Jacobian
    return sum;
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
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x
 */
__device__ float u0(float x, float y) {
    return 1.;
}

/* initial conditions
 *
 * computes the coefficients for the initial conditions
 * THREADS: num_elem
 */
__global__ void init_conditions(float *c, 
                                float *V1x, float *V1y,
                                float *V2x, float *V2y,
                                float *V3x, float *V3y,
                                float *r1, float *r2,
                                float *w,
                                int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;
    float x, y, u;

    if (idx < num_elem) {
        for (i = 0; i < n_p + 1; i++) {
            u = 0;
            for (j = 0; j < n_p + 1; j++) {
                // map from the canonical element to the actual point on the mesh
                x = (1 - r1[j] - r2[j]) * V1x[idx] + r1[j] * V2x[idx] + r2[j]*V3x[idx];
                y = (1 - r1[j] - r2[j]) * V1y[idx] + r1[j] * V2y[idx] + r2[j]*V3y[idx];
                // evaluate u there
                u += w[j] * u0(x, y) * basis(r1[j], r2[j], i);
            }
            c[i*num_elem + idx] = (2.*i + 1.) / 2. * u;
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
        s_length[idx] = sqrtf(powf(s_V1x[idx] - s_V2x[idx],2) + powf(s_V1y[idx] - s_V2y[idx],2));
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
        J[idx] = (-x1 + x2) * (-y1 + y3) - (-x1 + x3) * (-y1 + y2);
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
                          int *left_elem, int *left_side_number, int num_sides) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        float x, y, length;
        float sv1x, sv1y, sv2x, sv2y;
        float dot, left_x, left_y;
        float third_x, third_y;
        int left_idx, side;
    
        // get left side's vertices
        left_idx = left_elem[idx];
        side     = left_side_number[idx];

        sv1x = s_V1x[idx];
        sv1y = s_V1y[idx];
        sv2x = s_V2x[idx];
        sv2y = s_V2y[idx];
    
        // lengths of the vector components
        x = sv2x - sv1x;
        y = sv2y - sv1y;
    
        // normalize
        length = sqrtf(powf(x, 2) + powf(y, 2));
    
        // make it point the correct direction by learning the third vertex point
        switch (side) {
            case 1: 
                left_x = V3x[left_idx];
                left_y = V3y[left_idx];

                break;
            case 2:
                left_x = V1x[left_idx];
                left_y = V1y[left_idx];

                break;
            case 3:
                left_x = V2x[left_idx];
                left_y = V2y[left_idx];

                break;
        }

        // create the vector pointing towards the third vertex point
        third_x = left_x - (sv1x + sv2x) / 2.;
        third_y = left_y - (sv1y + sv2y) / 2.;
    
        // find the dot product between the normal vector and the third vetrex point
        dot = -y*third_x + x*third_y;
    
        // if the dot product is negative, reverse direction to point left to right
        // TODO: why the FUCK doesn't this boolean work? this is done on the CPU now.
        length = (dot < 0) ? -length : length;
        // store the result
        Nx[idx] = -y / length;
        Ny[idx] =  x / length;
    }
}

/***********************
 *
 * MAIN FUNCTIONS
 *
 ***********************/

/* flux evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: num_sides
 */
__global__ void eval_riemann(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                        float *J, float *s_length,
                        float *s1_r1, float *s1_r2,
                        float *s2_r1, float *s2_r2,
                        float *s3_r1, float *s3_r2,
                        float *oned_w,
                        int *left_idx_list, int *right_idx_list,
                        int *left_side_number, int *right_side_number, 
                        float *Nx, float *Ny, 
                        int n_quad1d, int n_p, int num_sides, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        int left_idx, right_idx, right_side, left_side, i, j;
        float c_left[10], c_right[10];
        float left_r1[10], right_r1[10];
        float left_r2[10], right_r2[10];
        float nx, ny, s;
        float u_left, u_right;
        float len, left_sum, right_sum;

        // find the left and right elements
        left_idx  = left_idx_list[idx];
        right_idx = right_idx_list[idx];

        // get the length of the side
        len = s_length[idx];

        // get the normal vector for this side
        nx = Nx[idx];
        ny = Ny[idx];

        // grab the coefficients for the left & right elements
        if (right_idx != -1) {
            // not a boundary side
            for (i = 0; i < (n_p + 1); i++) {
                c_left[i]  = c[i*num_elem + left_idx];
                c_right[i] = c[i*num_elem + right_idx];
            }
        } else {
            // this is a boundary side
            for (i = 0; i < (n_p + 1); i++) {
                c_left[i]  = c[i*num_elem + left_idx];
                c_right[i] = c[i*num_elem + left_idx];
            }
        }

        //TODO: does this speed it up
        __syncthreads();

        // need to find out what side we've got for evaluation (right, left, bottom)
        left_side  = left_side_number [idx];
        right_side = right_side_number[idx];

        // get the integration points for the left element's side
        switch (left_side) {
            case 1: 
                for (i = 0; i < n_quad1d; i++) {
                    left_r1[i] = s1_r1[i];
                    left_r2[i] = s1_r2[i];
                }
                break;
            case 2: 
                for (i = 0; i < n_quad1d; i++) {
                    left_r1[i] = s2_r1[i];
                    left_r2[i] = s2_r2[i];
                }
                break;
            case 3: 
                for (i = 0; i < n_quad1d; i++) {
                    left_r1[i] = s3_r1[i];
                    left_r2[i] = s3_r2[i];
                }
                break;
        }

        // TODO: does this speed it up?
        __syncthreads();
         
        // get the integration points for the right element's side
        switch (right_side) {
            case 0:
                for (i = 0; i < n_quad1d; i++) {
                    right_r1[i] = left_r1[i];
                    right_r2[i] = left_r2[i];
                }
                break;
            case 1: 
                for (i = 0; i < n_quad1d; i++) {
                    right_r1[i] = s1_r1[i];
                    right_r2[i] = s1_r2[i];
                }
                break;
            case 2: 
                for (i = 0; i < n_quad1d; i++) {
                    right_r1[i] = s2_r1[i];
                    right_r2[i] = s2_r2[i];
                }
                break;
            case 3: 
                for (i = 0; i < n_quad1d; i++) {
                    right_r1[i] = s3_r1[i];
                    right_r2[i] = s3_r2[i];
                }
                break;
        }

        // TODO: does this speed it up?
        __syncthreads();
         
        // evaluate the polynomial over that side for both elements and add the result to rhs
        left_sum  = 0;
        right_sum = 0;
        for (i = 0; i < (n_p + 1); i++) {
            u_left  = 0;
            u_right = 0;
            // compute u evaluated over the integration point
            for (j = 0; j < (n_p + 1); j++) {
                u_left  += c_left[i]  * basis(left_r1[i], left_r2[i], j);
                u_right += c_right[i] * basis(right_r1[i], right_r2[i], j);
            }

            // solve the Riemann problem at this integration point
            s = riemann(u_left, u_right);

            // calculate the quadrature over [-1,1] for these sides
            for (j = 0; j < n_quad1d; j++) {
                left_sum  += (nx * flux_x(s) + ny * flux_y(s)) * 
                              oned_w[i] * basis(left_r1[i],  left_r2[i],  j);
                right_sum += (nx * flux_x(s) + ny * flux_y(s)) * 
                              oned_w[i] * basis(right_r1[i], right_r2[i], j);
            }

            // store this side's contribution in the riemann rhs vector
            __syncthreads();
           //c [i * num_elem + left_idx]  += len / 2. * left_sum;
            //c[i * num_elem + right_idx] -= len / 2. * right_sum;
            left_riemann_rhs [i * num_sides + idx] =  len / 2. * left_sum;
            right_riemann_rhs[i * num_sides + idx] = -len / 2. * right_sum;
        }
    }
}

/* volume integrals
 *
 * evaluates and adds the volume integral to the rhs vector
 * THREADS: num_elem
 */
 __global__ void eval_quad(float *c, float *quad_rhs, 
                     float *r1, float *r2, float *w, float *J, 
                     int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        float quad_u, register_J;
        float register_c[10];

        // get the coefficients for this element
        for (i = 0; i < (n_p + 1); i++) {
            register_c[i] = c[i*num_elem + idx];
        }
         
        // the jacobian
        register_J = J[idx];

        // evaluate the volume integral for each coefficient
        for (i = 0; i < n_p; i++) {
            quad_u = quad(register_c, r1, r2, w, register_J, i, n_quad, n_p);
            quad_rhs[i * num_elem + idx] += -quad_u / register_J;
        }
    }
}

/* right hand side
 *
 * computes the sum of the quadrature and the riemann flux for the 
 * coefficients for each element
 * THREADS: num_elem
 */
__global__ void eval_rhs(float *c, float *quad_rhs, float *left_riemann_rhs, float *right_riemann_rhs, 
                         int *elem_s1, int *elem_s2, int *elem_s3,
                         int *left_elem, float dt, int n_p, int num_sides, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float s1, s2, s3;
    int i, s1_idx, s2_idx, s3_idx;

    if (idx < num_elem) {

        // get the indicies for the riemann contributions for this element
        s1_idx = elem_s1[idx];
        s2_idx = elem_s2[idx];
        s3_idx = elem_s3[idx];

        for (i = 0; i < n_p; i++) {

            // determine left or right pointing
            if (idx == left_elem[s1_idx]) {
                s1 = left_riemann_rhs[i * num_sides + s1_idx];
            } else {
                s1 = right_riemann_rhs[i * num_sides + s1_idx];
            }

            if (idx == left_elem[s2_idx]) {
                s2 = left_riemann_rhs[i * num_sides + s2_idx];
            } else {
                s2 = right_riemann_rhs[i * num_sides + s2_idx];
            }

            if (idx == left_elem[s3_idx]) {
                s3 = left_riemann_rhs[i * num_sides + s3_idx];
            } else {
                s3 = right_riemann_rhs[i * num_sides + s3_idx];
            }

            // calculate the coefficient c
            c[i * num_elem + idx] = dt * (quad_rhs[i * num_elem + idx] + s1 + s2 + s3);
        }
    }
}

/***********************
 *
 * TIME INTEGRATION
 *
 ***********************/

/* tempstorage for RK4
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
__global__ void rk4_tempstorage(float *c, float *kstar, float*k, float alpha, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < (n_p + 1) * num_elem) {
        kstar[idx] = c[idx] + alpha * k[idx];
    }
}

/* rk4
 *
 * computes the runge-kutta solution 
 * u_n+1 = u_n + k1/6 + k2/3 + k3/3 + k4/6
 */
__global__ void rk4(float *c, float *k1, float *k2, float *k3, float *k4, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < (n_p + 1) * num_elem) {
        c[idx] += k1[idx]/6. + k2[idx]/3. + k3[idx]/3. + k4[idx]/6.;
    }
}

