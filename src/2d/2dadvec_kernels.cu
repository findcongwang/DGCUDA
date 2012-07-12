/* 2dadvec_kernels.cu
 *
 * This file contains the kernels for the 2D advection DG method.
 * We use K = number of elements
 * and    H = number of sides
 */

#include "basis_eval.cu"


#define PI 3.14159

/***********************
 *
 * DEVICE VARIABLES
 *
 ***********************/
/* These are always prefixed with d_ for "device" */
double *d_c;           // holds coefficients for each element
double *d_quad_rhs;    // the right hand side containing the quadrature contributions
double *d_left_riemann_rhs;  // the right hand side containing the left riemann contributions
double *d_right_riemann_rhs; // the right hand side containing the right riemann contributions

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
__device__ __constant__ double basis[2048];
// note: these are multiplied by the weights
__device__ __constant__ double basis_grad_x[2048]; 
__device__ __constant__ double basis_grad_y[2048]; 

// precomputed basis functions evaluated along the sides. ordered
// similarly to basis and basis_grad_{x,y} but with one "matrix" for each side
// starting with side 0. to get to each side, offset with:
//      side_number * n_p * num_quad1d.
__device__ __constant__ double basis_side[1024];
__device__ __constant__ double basis_vertex[256];

// weights for 2d and 1d quadrature rules
__device__ __constant__ double w[32];
__device__ __constant__ double w_oned[16];

__device__ __constant__ double r1[32];
__device__ __constant__ double r2[32];
__device__ __constant__ double r_oned[32];

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

double *d_J;        // jacobian determinant 
double *d_s_length; // length of sides

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

/* flux function
 *
 * evaluates the flux f(u) at the point u.
 */
__device__ double flux_x(double u) {
    return u;
}
__device__ double flux_y(double u) {
    return u;
}

/* riemann solver
 *
 * evaluates the riemann problem over the boundary using Gaussian quadrature
 * with Legendre polynomials as basis functions.
 */
__device__ double riemann(double u_left, double u_right) {
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
__device__ double u0(double x, double y, int alpha) {
    return pow(x - y, alpha);
}

/* boundary exact
 *
 * returns the exact boundary conditions
 */
__device__ double boundary_exact(double x, double y, double t, int alpha) {
    return u0(x, y, alpha);
}

/* u exact
 *
 * returns the exact value of u for error measurement.
 */
__device__ double uexact(double x, double y, double t, int alpha) {
    return u0(x, y, alpha);
}

/* initial conditions
 *
 * computes the coefficients for the initial conditions
 * THREADS: num_elem
 */
__global__ void init_conditions(double *c, double *J,
                                double *V1x, double *V1y,
                                double *V2x, double *V2y,
                                double *V3x, double *V3y,
                                int n_quad, int n_p, int num_elem, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j;
    double x, y, u;

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
                u += w[j] * u0(x, y, alpha) * basis[i * n_quad + j];
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
__global__ void preval_side_length(double *s_length, 
                              double *s_V1x, double *s_V1y, 
                              double *s_V2x, double *s_V2y,
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
        length = sqrtf(pow(x, 2) + pow(y, 2));

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

/* riemann evaluation
 *
 * device function to solve the riemann problem.
 */
__device__ double eval_riemann(double *c_left, double *c_right,
                              double v1x, double v1y,
                              double v2x, double v2y,
                              double v3x, double v3y,
                              int j, // j, as usual, is the index of the integration point
                              int left_side, int right_side,
                              int left_idx, int right_idx,
                              int n_p, int n_quad1d,
                              int num_sides, double t, int alpha) {

    double u_left, u_right;
    int i;

    u_left  = 0.;
    u_right = 0.;

    for (i = 0; i < n_p; i++) {
        u_left  += c_left[i] * basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
    }

    // make all threads in the first warps be boundary sides
    if (right_idx == -1) {
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
            
        // deal with the boundary element here
        u_right = boundary_exact(x, y, t, alpha);

    } else {
        // evaluate the right side at the integration point
        for (i = 0; i < n_p; i++) {
            u_right  += c_right[i] * basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];
        }
    }

    return riemann(u_left, u_right);
}

/* surface integral evaluation
 *
 * evaluate all the riemann problems for each element.
 * THREADS: num_sides
 */
__device__ void eval_surface(double *c_left, double *c_right, 
                             double *left_riemann_rhs, double *right_riemann_rhs, 
                             double len,
                             double v1x, double v1y,
                             double v2x, double v2y,
                             double v3x, double v3y,
                             int left_idx,  int right_idx,
                             int left_side, int right_side, 
                             double nx, double ny, 
                             int n_quad1d, int n_p, int num_sides, 
                             int num_elem, double t, int idx, int alpha) {
    int i, j;
    double s, left_sum, right_sum;

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
                             n_p, n_quad1d, num_sides, t, alpha);

            // calculate the quadrature over [-1,1] for these sides
            left_sum  += (nx * flux_x(s) + ny * flux_y(s)) * w_oned[j] * 
                         basis_side[left_side * n_p * n_quad1d + i * n_quad1d + j];
            right_sum += (nx * flux_x(s) + ny * flux_y(s)) * w_oned[j] * 
                         basis_side[right_side * n_p * n_quad1d + i * n_quad1d + n_quad1d - 1 - j];
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
__device__ void eval_volume(double *r_c, double *quad_rhs, 
                            double x_r, double y_r,
                            double x_s, double y_s,
                            int n_quad, int n_p, int num_elem, int idx) {
    int i, j, k;
    double sum, u;

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
__device__ void eval_error(double *c, 
                       double v1x, double v1y,
                       double v2x, double v2y,
                       double v3x, double v3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int idx, int alpha) {

    int i;
    double uv1, uv2, uv3;

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
    Uv1[idx] = uv1 - uexact(v1x, v1y, t, alpha);
    Uv2[idx] = uv2 - uexact(v2x, v2y, t, alpha);
    Uv3[idx] = uv3 - uexact(v3x, v3y, t, alpha);
}

/* evaluate u
 * 
 * evaluates u at the three vertex points for output
 * THREADS: num_elem
 */
__device__ void eval_u(double *c, 
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, int idx) {
    int i;
    double uv1, uv2, uv3;

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
