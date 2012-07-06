/* 2dadvec_kernels.cu
 *
 * This file contains the kernels for the 2D advection DG method.
 * We use K = number of elements
 * and    H = number of sides
 */


#define PI 3.14159

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
    return x - 2 * y;
}

/* boundary exact
 *
 * returns the exact boundary conditions
 */
__device__ float boundary_exact(float x, float y, float t) {
    return x + t - 2 * y;
}

/* u exact
 *
 * returns the exact value of u for error measurement.
 */
__device__ float uexact(float x, float y) {
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
    float x, y, u;

    if (idx < num_elem) {
        for (i = 0; i < n_p; i++) {
            u = 0.;
            // perform quadrature
            for (j = 0; j < n_quad; j++) {
                // map from the canonical element to the actual point on the mesh
                // x = x2 * r + x3 * s + x1 * (1 - r - s)
                x = r1[j] * V2x[idx] + r2[j] * V3x[idx] + (1 - r1[j] - r2[j]) * V1x[idx];
                y = r1[j] * V2y[idx] + r2[j] * V3y[idx] + (1 - r1[j] - r2[j]) * V1y[idx];

                // evaluate u there
                u += w[j] * u0(x, y) * basis[i * n_quad + j];
            }
            c[i * num_elem + idx] = u;
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
        // x = x2 * r + x3 * s + x1 * (1 - r - s)
        J[idx] = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    }
}

/* force positive jacobians
 *
 * make sure all jacobians are positive by swapping two verticies.
 */
__global__ void preval_jacobian_sign(float *J, 
                           float *V1x, float *V1y, 
                           float *V2x, float *V2y, 
                           int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float tmpx, tmpy;
        if (J[idx] < 0) {
            tmpx = V1x[idx];
            tmpy = V1y[idx];
            V1x[idx] = V2x[idx];
            V1y[idx] = V2y[idx];
            V2x[idx] = tmpx;
            V2y[idx] = tmpy;

            J[idx] *= -1;
        }
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
__device__ float eval_riemann(float *c_left, float *c_right,
                              float v1x, float v1y,
                              float v2x, float v2y,
                              float v3x, float v3y,
                              int j, // j, as usual, is the index of the integration point
                              int left_side, int right_side,
                              int left_idx, int right_idx,
                              int n_p, int n_quad1d,
                              int num_sides, float t) {

    float u_left, u_right;
    int i;

    u_left  = 0.;
    u_right = 0.;

    for (i = 0; i < n_p; i++) {
        u_left  += c_left[i] * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
    }

    // make all threads in the first warps be boundary sides
    if (right_idx == -1) {
        float r1, r2;
        float x, y;

        // we (sometimes?) need the mapping back to the grid space
        switch (left_side) {
            case 0: 
                r1 = 0.5 + 0.5 * r_oned[j];
                r2 = 0.;
                break;
            case 1: 
                r1 = (1. - r_oned[j]) / 2.;
                r2 = (1. + r_oned[j]) / 2.;
                break;
            case 2: 
                r1 = 0.;
                r2 = 0.5 + 0.5 * r_oned[n_quad1d - 1 - j];
                break;
        }

        // x = x2 * r + x3 * s + x1 * (1 - r - s)
        x = v2x * r1
          + v3x * r2
          + v1x * (1 - r1 - r2);
        y = v2y * r1
          + v3y * r2
          + v1y * (1 - r1 - r2);
            
        // deal with the boundary element here
        u_right = boundary_exact(x, y, t);

    } else {
        // evaluate the right side at the integration point
        for (i = 0; i < n_p; i++) {
            u_right  += c_right[i] * basis_side[right_side * n_p * n_quad1d + i * n_quad1d + j];
        }
    }

    return riemann(u_left, u_right);

}

/* surface integral evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: num_sides
 */
__device__ void eval_surface(float *c_left, float *c_right, 
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
    float s, left_sum, right_sum;

    // multiply across by the i'th basis function
    for (i = 0; i < n_p; i++) {
        left_sum  = 0.;
        right_sum = 0.;
        // we're at the j'th integration point
        for (j = 0; j < n_quad1d; j++) {
            // solve the Riemann problem at this integration point
            s = eval_riemann(c_left, c_right,
                             v1x, v1y, v2x, v2y, v3x, v3y,
                             j, left_side, right_side, left_idx, right_idx,
                             n_p, n_quad1d, num_sides, t);

            // calculate the quadrature over [-1,1] for these sides
            left_sum  += (nx * flux_x(s) + ny * flux_y(s)) *
                         w_oned[j] * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            right_sum += (nx * flux_x(s) + ny * flux_y(s)) *
                         w_oned[j] * basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];
        }

        // store this side's contribution in the riemann rhs vectors
        left_riemann_rhs[i * num_sides + idx]  = -len / 2 * left_sum;
        right_riemann_rhs[i * num_sides + idx] =  len / 2 * right_sum;
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
__device__ void eval_volume(float *r_c, float *quad_rhs, 
                            float x_r, float y_r,
                            float x_s, float y_s,
                            int n_quad, int n_p, int num_elem, int idx) {
    int i, j, k;
    float sum, u;

    // evaluate the volume integral for each coefficient
    for (i = 0; i < n_p; i++) {
        sum = 0.;
        for (j = 0; j < n_quad; j++) {
            // Evaluate u at the integration point.
            u = 0.;
            for (k = 0; k < n_p; k++) {
                u += r_c[k] * basis[n_quad * k + j];
            }

            // Add to the sum
            // [fx fy] * [y_s, -y_r; -x_s, x_r] * [phi_x phi_y]
            sum += (  flux_x(u) * ( basis_grad_x[n_quad * i + j] * y_s
                                   -basis_grad_y[n_quad * i + j] * y_r)
                    + flux_y(u) * (-basis_grad_x[n_quad * i + j] * x_s 
                                  + basis_grad_y[n_quad * i + j] * x_r));
        }

        // store the result
        quad_rhs[i * num_elem + idx] = sum;
    }
}

/* evaluate error
 * 
 * evaluates u at the three vertex points for output
 * THREADS: num_elem
 */
__global__ void eval_error(float *c, 
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < num_elem) {
        int i;
        float register_c[36];
        float uv1, uv2, uv3;

        // read coefficient values
        for (i = 0; i < n_p; i++) {
            register_c[i] = c[i * num_elem + idx];
        }

        uv1 = 0.;
        uv2 = 0.;
        uv3 = 0.;

        // calculate values at three vertex points
        for (i = 0; i < n_p; i++) {
            uv1 += register_c[i] * basis_vertex[i * 3 + 0];
            uv2 += register_c[i] * basis_vertex[i * 3 + 1];
            uv3 += register_c[i] * basis_vertex[i * 3 + 2];
        }

        // store result
        Uv1[idx] = abs(uv1 - uexact(V1x[idx], V1y[idx]));
        Uv2[idx] = abs(uv2 - uexact(V2x[idx], V2y[idx]));
        Uv3[idx] = abs(uv3 - uexact(V3x[idx], V3y[idx]));
    }
}

/* evaluate u
 * 
 * evaluates u at the three vertex points for output
 * THREADS: num_elem
 */
__global__ void eval_u(float *c, 
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < num_elem) {
        int i;
        float register_c[36];
        float uv1, uv2, uv3;

        // read coefficient values
        for (i = 0; i < n_p; i++) {
            register_c[i] = c[i * num_elem + idx];
        }

        uv1 = 0.;
        uv2 = 0.;
        uv3 = 0.;

        // calculate values at the integration points
        for (i = 0; i < n_p; i++) {
            uv1 += register_c[i] * basis_vertex[i * n_p + 0];
            uv2 += register_c[i] * basis_vertex[i * n_p + 1];
            uv3 += register_c[i] * basis_vertex[i * n_p + 2];
        }

        // store result
        Uv1[idx] = uv1;
        Uv2[idx] = uv2;
        Uv3[idx] = uv3;
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
                         int *left_elem, float *J, 
                         float dt, int n_p, int num_sides, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float s1, s2, s3, register_J;
    int i, s1_idx, s2_idx, s3_idx;

    if (idx < num_elem) {

        register_J = J[idx];

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
            c[i * num_elem + idx] = 1. / register_J * dt * (quad_rhs[i * num_elem + idx] + s1 + s2 + s3);
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

    if (idx < n_p * num_elem) {
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

    if (idx < n_p * num_elem) {
        c[idx] += k1[idx]/6. + k2[idx]/3. + k3[idx]/3. + k4[idx]/6.;
    }
}
