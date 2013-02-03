#ifndef WRAPPER_H_GUARD
#define WRAPPER_H_GUARD
void eval_surface(double*, double*,
                   double*, double*,
                   double,
                   double, double,
                   double, double,
                   double, double,
                   int, int,
                   double, double,
                   int, int, int, int, double, int, int);
#endif

/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper0(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[1], r_c_right[1];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}


/* eval surface wrapper (n = 1)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper1(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[3], r_c_right[3];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}


/* eval surface wrapper (n = 2)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper2(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[6], r_c_right[6];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
            r_c_left [3] = c[3 * num_elem + left_elem[idx]];
            r_c_right[3] = c[3 * num_elem + right_elem[idx]];
            r_c_left [4] = c[4 * num_elem + left_elem[idx]];
            r_c_right[4] = c[4 * num_elem + right_elem[idx]];
            r_c_left [5] = c[5 * num_elem + left_elem[idx]];
            r_c_right[5] = c[5 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
            r_c_left[3]  = c[3 * num_elem + left_elem[idx]];
            r_c_left[4]  = c[4 * num_elem + left_elem[idx]];
            r_c_left[5]  = c[5 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}


/* eval surface wrapper (n = 3)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper3(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[10], r_c_right[10];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
            r_c_left [3] = c[3 * num_elem + left_elem[idx]];
            r_c_right[3] = c[3 * num_elem + right_elem[idx]];
            r_c_left [4] = c[4 * num_elem + left_elem[idx]];
            r_c_right[4] = c[4 * num_elem + right_elem[idx]];
            r_c_left [5] = c[5 * num_elem + left_elem[idx]];
            r_c_right[5] = c[5 * num_elem + right_elem[idx]];
            r_c_left [6] = c[6 * num_elem + left_elem[idx]];
            r_c_right[6] = c[6 * num_elem + right_elem[idx]];
            r_c_left [7] = c[7 * num_elem + left_elem[idx]];
            r_c_right[7] = c[7 * num_elem + right_elem[idx]];
            r_c_left [8] = c[8 * num_elem + left_elem[idx]];
            r_c_right[8] = c[8 * num_elem + right_elem[idx]];
            r_c_left [9] = c[9 * num_elem + left_elem[idx]];
            r_c_right[9] = c[9 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
            r_c_left[3]  = c[3 * num_elem + left_elem[idx]];
            r_c_left[4]  = c[4 * num_elem + left_elem[idx]];
            r_c_left[5]  = c[5 * num_elem + left_elem[idx]];
            r_c_left[6]  = c[6 * num_elem + left_elem[idx]];
            r_c_left[7]  = c[7 * num_elem + left_elem[idx]];
            r_c_left[8]  = c[8 * num_elem + left_elem[idx]];
            r_c_left[9]  = c[9 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}

/* eval surface wrapper (n = 4)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper4(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[15], r_c_right[15];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
            r_c_left [3] = c[3 * num_elem + left_elem[idx]];
            r_c_right[3] = c[3 * num_elem + right_elem[idx]];
            r_c_left [4] = c[4 * num_elem + left_elem[idx]];
            r_c_right[4] = c[4 * num_elem + right_elem[idx]];
            r_c_left [5] = c[5 * num_elem + left_elem[idx]];
            r_c_right[5] = c[5 * num_elem + right_elem[idx]];
            r_c_left [6] = c[6 * num_elem + left_elem[idx]];
            r_c_right[6] = c[6 * num_elem + right_elem[idx]];
            r_c_left [7] = c[7 * num_elem + left_elem[idx]];
            r_c_right[7] = c[7 * num_elem + right_elem[idx]];
            r_c_left [8] = c[8 * num_elem + left_elem[idx]];
            r_c_right[8] = c[8 * num_elem + right_elem[idx]];
            r_c_left [9] = c[9 * num_elem + left_elem[idx]];
            r_c_right[9] = c[9 * num_elem + right_elem[idx]];

            r_c_left [10] = c[10 * num_elem + left_elem[idx]];
            r_c_right[10] = c[10 * num_elem + right_elem[idx]];
            r_c_left [11] = c[11 * num_elem + left_elem[idx]];
            r_c_right[11] = c[11 * num_elem + right_elem[idx]];
            r_c_left [12] = c[12 * num_elem + left_elem[idx]];
            r_c_right[12] = c[12 * num_elem + right_elem[idx]];
            r_c_left [13] = c[13 * num_elem + left_elem[idx]];
            r_c_right[13] = c[13 * num_elem + right_elem[idx]];
            r_c_left [14] = c[14 * num_elem + left_elem[idx]];
            r_c_right[14] = c[14 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
            r_c_left[3]  = c[3 * num_elem + left_elem[idx]];
            r_c_left[4]  = c[4 * num_elem + left_elem[idx]];
            r_c_left[5]  = c[5 * num_elem + left_elem[idx]];
            r_c_left[6]  = c[6 * num_elem + left_elem[idx]];
            r_c_left[7]  = c[7 * num_elem + left_elem[idx]];
            r_c_left[8]  = c[8 * num_elem + left_elem[idx]];
            r_c_left[9]  = c[9 * num_elem + left_elem[idx]];

            r_c_left[10]  = c[10 * num_elem + left_elem[idx]];
            r_c_left[11]  = c[11 * num_elem + left_elem[idx]];
            r_c_left[12]  = c[12 * num_elem + left_elem[idx]];
            r_c_left[13]  = c[13 * num_elem + left_elem[idx]];
            r_c_left[14]  = c[14 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}

/* eval surface wrapper (n = 5)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper5(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[21], r_c_right[21];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
            r_c_left [3] = c[3 * num_elem + left_elem[idx]];
            r_c_right[3] = c[3 * num_elem + right_elem[idx]];
            r_c_left [4] = c[4 * num_elem + left_elem[idx]];
            r_c_right[4] = c[4 * num_elem + right_elem[idx]];
            r_c_left [5] = c[5 * num_elem + left_elem[idx]];
            r_c_right[5] = c[5 * num_elem + right_elem[idx]];
            r_c_left [6] = c[6 * num_elem + left_elem[idx]];
            r_c_right[6] = c[6 * num_elem + right_elem[idx]];
            r_c_left [7] = c[7 * num_elem + left_elem[idx]];
            r_c_right[7] = c[7 * num_elem + right_elem[idx]];
            r_c_left [8] = c[8 * num_elem + left_elem[idx]];
            r_c_right[8] = c[8 * num_elem + right_elem[idx]];
            r_c_left [9] = c[9 * num_elem + left_elem[idx]];
            r_c_right[9] = c[9 * num_elem + right_elem[idx]];

            r_c_left [10] = c[10 * num_elem + left_elem[idx]];
            r_c_right[10] = c[10 * num_elem + right_elem[idx]];
            r_c_left [11] = c[11 * num_elem + left_elem[idx]];
            r_c_right[11] = c[11 * num_elem + right_elem[idx]];
            r_c_left [12] = c[12 * num_elem + left_elem[idx]];
            r_c_right[12] = c[12 * num_elem + right_elem[idx]];
            r_c_left [13] = c[13 * num_elem + left_elem[idx]];
            r_c_right[13] = c[13 * num_elem + right_elem[idx]];
            r_c_left [14] = c[14 * num_elem + left_elem[idx]];
            r_c_right[14] = c[14 * num_elem + right_elem[idx]];
            r_c_left [15] = c[15 * num_elem + left_elem[idx]];
            r_c_right[15] = c[15 * num_elem + right_elem[idx]];
            r_c_left [16] = c[16 * num_elem + left_elem[idx]];
            r_c_right[16] = c[16 * num_elem + right_elem[idx]];
            r_c_left [17] = c[17 * num_elem + left_elem[idx]];
            r_c_right[17] = c[17 * num_elem + right_elem[idx]];
            r_c_left [18] = c[18 * num_elem + left_elem[idx]];
            r_c_right[18] = c[18 * num_elem + right_elem[idx]];
            r_c_left [19] = c[19 * num_elem + left_elem[idx]];
            r_c_right[19] = c[19 * num_elem + right_elem[idx]];

            r_c_left [20] = c[20 * num_elem + left_elem[idx]];
            r_c_right[20] = c[20 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
            r_c_left[3]  = c[3 * num_elem + left_elem[idx]];
            r_c_left[4]  = c[4 * num_elem + left_elem[idx]];
            r_c_left[5]  = c[5 * num_elem + left_elem[idx]];
            r_c_left[6]  = c[6 * num_elem + left_elem[idx]];
            r_c_left[7]  = c[7 * num_elem + left_elem[idx]];
            r_c_left[8]  = c[8 * num_elem + left_elem[idx]];
            r_c_left[9]  = c[9 * num_elem + left_elem[idx]];

            r_c_left[10]  = c[10 * num_elem + left_elem[idx]];
            r_c_left[11]  = c[11 * num_elem + left_elem[idx]];
            r_c_left[12]  = c[12 * num_elem + left_elem[idx]];
            r_c_left[13]  = c[13 * num_elem + left_elem[idx]];
            r_c_left[14]  = c[14 * num_elem + left_elem[idx]];
            r_c_left[15]  = c[15 * num_elem + left_elem[idx]];
            r_c_left[16]  = c[16 * num_elem + left_elem[idx]];
            r_c_left[17]  = c[17 * num_elem + left_elem[idx]];
            r_c_left[18]  = c[18 * num_elem + left_elem[idx]];
            r_c_left[19]  = c[19 * num_elem + left_elem[idx]];

            r_c_left[20]  = c[20 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}


/* eval surface wrapper (n = 6)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper6(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[28], r_c_right[28];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
            r_c_left [3] = c[3 * num_elem + left_elem[idx]];
            r_c_right[3] = c[3 * num_elem + right_elem[idx]];
            r_c_left [4] = c[4 * num_elem + left_elem[idx]];
            r_c_right[4] = c[4 * num_elem + right_elem[idx]];
            r_c_left [5] = c[5 * num_elem + left_elem[idx]];
            r_c_right[5] = c[5 * num_elem + right_elem[idx]];
            r_c_left [6] = c[6 * num_elem + left_elem[idx]];
            r_c_right[6] = c[6 * num_elem + right_elem[idx]];
            r_c_left [7] = c[7 * num_elem + left_elem[idx]];
            r_c_right[7] = c[7 * num_elem + right_elem[idx]];
            r_c_left [8] = c[8 * num_elem + left_elem[idx]];
            r_c_right[8] = c[8 * num_elem + right_elem[idx]];
            r_c_left [9] = c[9 * num_elem + left_elem[idx]];
            r_c_right[9] = c[9 * num_elem + right_elem[idx]];

            r_c_left [10] = c[10 * num_elem + left_elem[idx]];
            r_c_right[10] = c[10 * num_elem + right_elem[idx]];
            r_c_left [11] = c[11 * num_elem + left_elem[idx]];
            r_c_right[11] = c[11 * num_elem + right_elem[idx]];
            r_c_left [12] = c[12 * num_elem + left_elem[idx]];
            r_c_right[12] = c[12 * num_elem + right_elem[idx]];
            r_c_left [13] = c[13 * num_elem + left_elem[idx]];
            r_c_right[13] = c[13 * num_elem + right_elem[idx]];
            r_c_left [14] = c[14 * num_elem + left_elem[idx]];
            r_c_right[14] = c[14 * num_elem + right_elem[idx]];
            r_c_left [15] = c[15 * num_elem + left_elem[idx]];
            r_c_right[15] = c[15 * num_elem + right_elem[idx]];
            r_c_left [16] = c[16 * num_elem + left_elem[idx]];
            r_c_right[16] = c[16 * num_elem + right_elem[idx]];
            r_c_left [17] = c[17 * num_elem + left_elem[idx]];
            r_c_right[17] = c[17 * num_elem + right_elem[idx]];
            r_c_left [18] = c[18 * num_elem + left_elem[idx]];
            r_c_right[18] = c[18 * num_elem + right_elem[idx]];
            r_c_left [19] = c[19 * num_elem + left_elem[idx]];
            r_c_right[19] = c[19 * num_elem + right_elem[idx]];

            r_c_left [20] = c[20 * num_elem + left_elem[idx]];
            r_c_right[20] = c[20 * num_elem + right_elem[idx]];
            r_c_left [21] = c[21 * num_elem + left_elem[idx]];
            r_c_right[21] = c[21 * num_elem + right_elem[idx]];
            r_c_left [22] = c[22 * num_elem + left_elem[idx]];
            r_c_right[22] = c[22 * num_elem + right_elem[idx]];
            r_c_left [23] = c[23 * num_elem + left_elem[idx]];
            r_c_right[23] = c[23 * num_elem + right_elem[idx]];
            r_c_left [24] = c[24 * num_elem + left_elem[idx]];
            r_c_right[24] = c[24 * num_elem + right_elem[idx]];
            r_c_left [25] = c[25 * num_elem + left_elem[idx]];
            r_c_right[25] = c[25 * num_elem + right_elem[idx]];
            r_c_left [26] = c[26 * num_elem + left_elem[idx]];
            r_c_right[26] = c[26 * num_elem + right_elem[idx]];
            r_c_left [27] = c[27 * num_elem + left_elem[idx]];
            r_c_right[27] = c[27 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
            r_c_left[3]  = c[3 * num_elem + left_elem[idx]];
            r_c_left[4]  = c[4 * num_elem + left_elem[idx]];
            r_c_left[5]  = c[5 * num_elem + left_elem[idx]];
            r_c_left[6]  = c[6 * num_elem + left_elem[idx]];
            r_c_left[7]  = c[7 * num_elem + left_elem[idx]];
            r_c_left[8]  = c[8 * num_elem + left_elem[idx]];
            r_c_left[9]  = c[9 * num_elem + left_elem[idx]];

            r_c_left[10]  = c[10 * num_elem + left_elem[idx]];
            r_c_left[11]  = c[11 * num_elem + left_elem[idx]];
            r_c_left[12]  = c[12 * num_elem + left_elem[idx]];
            r_c_left[13]  = c[13 * num_elem + left_elem[idx]];
            r_c_left[14]  = c[14 * num_elem + left_elem[idx]];
            r_c_left[15]  = c[15 * num_elem + left_elem[idx]];
            r_c_left[16]  = c[16 * num_elem + left_elem[idx]];
            r_c_left[17]  = c[17 * num_elem + left_elem[idx]];
            r_c_left[18]  = c[18 * num_elem + left_elem[idx]];
            r_c_left[19]  = c[19 * num_elem + left_elem[idx]];

            r_c_left[20]  = c[20 * num_elem + left_elem[idx]];
            r_c_left[21]  = c[21 * num_elem + left_elem[idx]];
            r_c_left[22]  = c[22 * num_elem + left_elem[idx]];
            r_c_left[23]  = c[23 * num_elem + left_elem[idx]];
            r_c_left[24]  = c[24 * num_elem + left_elem[idx]];
            r_c_left[25]  = c[25 * num_elem + left_elem[idx]];
            r_c_left[26]  = c[26 * num_elem + left_elem[idx]];
            r_c_left[27]  = c[27 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}

/* eval surface wrapper (n = 7)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper7(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double r_c_left[36], r_c_right[36];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            r_c_left [0] = c[left_elem[idx]];
            r_c_right[0] = c[right_elem[idx]];
            r_c_left [1] = c[num_elem + left_elem[idx]];
            r_c_right[1] = c[num_elem + right_elem[idx]];
            r_c_left [2] = c[2 * num_elem + left_elem[idx]];
            r_c_right[2] = c[2 * num_elem + right_elem[idx]];
            r_c_left [3] = c[3 * num_elem + left_elem[idx]];
            r_c_right[3] = c[3 * num_elem + right_elem[idx]];
            r_c_left [4] = c[4 * num_elem + left_elem[idx]];
            r_c_right[4] = c[4 * num_elem + right_elem[idx]];
            r_c_left [5] = c[5 * num_elem + left_elem[idx]];
            r_c_right[5] = c[5 * num_elem + right_elem[idx]];
            r_c_left [6] = c[6 * num_elem + left_elem[idx]];
            r_c_right[6] = c[6 * num_elem + right_elem[idx]];
            r_c_left [7] = c[7 * num_elem + left_elem[idx]];
            r_c_right[7] = c[7 * num_elem + right_elem[idx]];
            r_c_left [8] = c[8 * num_elem + left_elem[idx]];
            r_c_right[8] = c[8 * num_elem + right_elem[idx]];
            r_c_left [9] = c[9 * num_elem + left_elem[idx]];
            r_c_right[9] = c[9 * num_elem + right_elem[idx]];

            r_c_left [10] = c[10 * num_elem + left_elem[idx]];
            r_c_right[10] = c[10 * num_elem + right_elem[idx]];
            r_c_left [11] = c[11 * num_elem + left_elem[idx]];
            r_c_right[11] = c[11 * num_elem + right_elem[idx]];
            r_c_left [12] = c[12 * num_elem + left_elem[idx]];
            r_c_right[12] = c[12 * num_elem + right_elem[idx]];
            r_c_left [13] = c[13 * num_elem + left_elem[idx]];
            r_c_right[13] = c[13 * num_elem + right_elem[idx]];
            r_c_left [14] = c[14 * num_elem + left_elem[idx]];
            r_c_right[14] = c[14 * num_elem + right_elem[idx]];
            r_c_left [15] = c[15 * num_elem + left_elem[idx]];
            r_c_right[15] = c[15 * num_elem + right_elem[idx]];
            r_c_left [16] = c[16 * num_elem + left_elem[idx]];
            r_c_right[16] = c[16 * num_elem + right_elem[idx]];
            r_c_left [17] = c[17 * num_elem + left_elem[idx]];
            r_c_right[17] = c[17 * num_elem + right_elem[idx]];
            r_c_left [18] = c[18 * num_elem + left_elem[idx]];
            r_c_right[18] = c[18 * num_elem + right_elem[idx]];
            r_c_left [19] = c[19 * num_elem + left_elem[idx]];
            r_c_right[19] = c[19 * num_elem + right_elem[idx]];

            r_c_left [20] = c[20 * num_elem + left_elem[idx]];
            r_c_right[20] = c[20 * num_elem + right_elem[idx]];
            r_c_left [21] = c[21 * num_elem + left_elem[idx]];
            r_c_right[21] = c[21 * num_elem + right_elem[idx]];
            r_c_left [22] = c[22 * num_elem + left_elem[idx]];
            r_c_right[22] = c[22 * num_elem + right_elem[idx]];
            r_c_left [23] = c[23 * num_elem + left_elem[idx]];
            r_c_right[23] = c[23 * num_elem + right_elem[idx]];
            r_c_left [24] = c[24 * num_elem + left_elem[idx]];
            r_c_right[24] = c[24 * num_elem + right_elem[idx]];
            r_c_left [25] = c[25 * num_elem + left_elem[idx]];
            r_c_right[25] = c[25 * num_elem + right_elem[idx]];
            r_c_left [26] = c[26 * num_elem + left_elem[idx]];
            r_c_right[26] = c[26 * num_elem + right_elem[idx]];
            r_c_left [27] = c[27 * num_elem + left_elem[idx]];
            r_c_right[27] = c[27 * num_elem + right_elem[idx]];
            r_c_left [28] = c[28 * num_elem + left_elem[idx]];
            r_c_right[28] = c[28 * num_elem + right_elem[idx]];
            r_c_left [29] = c[29 * num_elem + left_elem[idx]];
            r_c_right[29] = c[29 * num_elem + right_elem[idx]];

            r_c_left [30] = c[30 * num_elem + left_elem[idx]];
            r_c_right[30] = c[30 * num_elem + right_elem[idx]];
            r_c_left [31] = c[31 * num_elem + left_elem[idx]];
            r_c_right[31] = c[31 * num_elem + right_elem[idx]];
            r_c_left [32] = c[32 * num_elem + left_elem[idx]];
            r_c_right[32] = c[32 * num_elem + right_elem[idx]];
            r_c_left [33] = c[33 * num_elem + left_elem[idx]];
            r_c_right[33] = c[33 * num_elem + right_elem[idx]];
            r_c_left [34] = c[34 * num_elem + left_elem[idx]];
            r_c_right[34] = c[34 * num_elem + right_elem[idx]];
            r_c_left [35] = c[35 * num_elem + left_elem[idx]];
            r_c_right[35] = c[35 * num_elem + right_elem[idx]];
        } else {
            r_c_left[0]  = c[left_elem[idx]];
            r_c_left[1]  = c[num_elem + left_elem[idx]];
            r_c_left[2]  = c[2 * num_elem + left_elem[idx]];
            r_c_left[3]  = c[3 * num_elem + left_elem[idx]];
            r_c_left[4]  = c[4 * num_elem + left_elem[idx]];
            r_c_left[5]  = c[5 * num_elem + left_elem[idx]];
            r_c_left[6]  = c[6 * num_elem + left_elem[idx]];
            r_c_left[7]  = c[7 * num_elem + left_elem[idx]];
            r_c_left[8]  = c[8 * num_elem + left_elem[idx]];
            r_c_left[9]  = c[9 * num_elem + left_elem[idx]];

            r_c_left[10]  = c[10 * num_elem + left_elem[idx]];
            r_c_left[11]  = c[11 * num_elem + left_elem[idx]];
            r_c_left[12]  = c[12 * num_elem + left_elem[idx]];
            r_c_left[13]  = c[13 * num_elem + left_elem[idx]];
            r_c_left[14]  = c[14 * num_elem + left_elem[idx]];
            r_c_left[15]  = c[15 * num_elem + left_elem[idx]];
            r_c_left[16]  = c[16 * num_elem + left_elem[idx]];
            r_c_left[17]  = c[17 * num_elem + left_elem[idx]];
            r_c_left[18]  = c[18 * num_elem + left_elem[idx]];
            r_c_left[19]  = c[19 * num_elem + left_elem[idx]];

            r_c_left[20]  = c[20 * num_elem + left_elem[idx]];
            r_c_left[21]  = c[21 * num_elem + left_elem[idx]];
            r_c_left[22]  = c[22 * num_elem + left_elem[idx]];
            r_c_left[23]  = c[23 * num_elem + left_elem[idx]];
            r_c_left[24]  = c[24 * num_elem + left_elem[idx]];
            r_c_left[25]  = c[25 * num_elem + left_elem[idx]];
            r_c_left[26]  = c[26 * num_elem + left_elem[idx]];
            r_c_left[27]  = c[27 * num_elem + left_elem[idx]];
            r_c_left[28]  = c[28 * num_elem + left_elem[idx]];
            r_c_left[29]  = c[29 * num_elem + left_elem[idx]];

            r_c_left[30]  = c[30 * num_elem + left_elem[idx]];
            r_c_left[31]  = c[31 * num_elem + left_elem[idx]];
            r_c_left[32]  = c[32 * num_elem + left_elem[idx]];
            r_c_left[33]  = c[33 * num_elem + left_elem[idx]];
            r_c_left[34]  = c[34 * num_elem + left_elem[idx]];
            r_c_left[35]  = c[35 * num_elem + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(r_c_left, r_c_right,
                     left_riemann_rhs, right_riemann_rhs,
                     s_length[idx],
                     V1x[left_elem[idx]], V1y[left_elem[idx]],
                     V2x[left_elem[idx]], V2y[left_elem[idx]],
                     V3x[left_elem[idx]], V3y[left_elem[idx]],
                     left_elem[idx], right_elem[idx],
                     left_side_number[idx], right_side_number[idx],
                     Nx[idx], Ny[idx],
                     n_quad1d, n_p, num_sides, num_elem, t, idx, alpha);
    }
}

//* eval volume wrapper (n = 0)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper0(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[1];

        // get the coefficients for this element
        r_c[0] = c[idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 1)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper1(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[3];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 2)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper2(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[6];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 3)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper3(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[10];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 4)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 
 __global__ void eval_volume_wrapper4(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[15];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 5)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper5(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[21];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 6)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper6(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[28];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];
        r_c[21] = c[21 * num_elem + idx];
        r_c[22] = c[22 * num_elem + idx];
        r_c[23] = c[23 * num_elem + idx];
        r_c[24] = c[24 * num_elem + idx];
        r_c[25] = c[25 * num_elem + idx];
        r_c[26] = c[26 * num_elem + idx];
        r_c[27] = c[27 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 7)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper7(double *c, double *quad_rhs, 
                                      double *xr, double *yr,
                                      double *xs, double *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[36];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];
        r_c[21] = c[21 * num_elem + idx];
        r_c[22] = c[22 * num_elem + idx];
        r_c[23] = c[23 * num_elem + idx];
        r_c[24] = c[24 * num_elem + idx];
        r_c[25] = c[25 * num_elem + idx];
        r_c[26] = c[26 * num_elem + idx];
        r_c[27] = c[27 * num_elem + idx];
        r_c[28] = c[28 * num_elem + idx];
        r_c[29] = c[29 * num_elem + idx];

        r_c[30] = c[30 * num_elem + idx];
        r_c[31] = c[31 * num_elem + idx];
        r_c[32] = c[32 * num_elem + idx];
        r_c[33] = c[33 * num_elem + idx];
        r_c[34] = c[34 * num_elem + idx];
        r_c[35] = c[35 * num_elem + idx];

        eval_volume(r_c, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval u wrapper (n = 0)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper0(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[1];

        // get the coefficients for this element
        r_c[0] = c[idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 1)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper1(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[3];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 2)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper2(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[6];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 3)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper3(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[10];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 4)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
 
__global__ void eval_u_wrapper4(double *c, 
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[15];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 5)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper5(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[21];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 6)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper6(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[28];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];
        r_c[21] = c[21 * num_elem + idx];
        r_c[22] = c[22 * num_elem + idx];
        r_c[23] = c[23 * num_elem + idx];
        r_c[24] = c[24 * num_elem + idx];
        r_c[25] = c[25 * num_elem + idx];
        r_c[26] = c[26 * num_elem + idx];
        r_c[27] = c[27 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 7)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper7(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[36];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];
        r_c[21] = c[21 * num_elem + idx];
        r_c[22] = c[22 * num_elem + idx];
        r_c[23] = c[23 * num_elem + idx];
        r_c[24] = c[24 * num_elem + idx];
        r_c[25] = c[25 * num_elem + idx];
        r_c[26] = c[26 * num_elem + idx];
        r_c[27] = c[27 * num_elem + idx];
        r_c[28] = c[28 * num_elem + idx];
        r_c[29] = c[29 * num_elem + idx];

        r_c[30] = c[30 * num_elem + idx];
        r_c[31] = c[31 * num_elem + idx];
        r_c[32] = c[32 * num_elem + idx];
        r_c[33] = c[33 * num_elem + idx];
        r_c[34] = c[34 * num_elem + idx];
        r_c[35] = c[35 * num_elem + idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval error wrapper (n = 0)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper0(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[1];

        // get the coefficients for this element
        r_c[0] = c[idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
    }
}

//* eval error wrapper (n = 1)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper1(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[3];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
    }
}

//* eval error wrapper (n = 2)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper2(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[6];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
         
    }
}

//* eval error wrapper (n = 3)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper3(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[10];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
         
    }
}

//* eval error wrapper (n = 4)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
 
__global__ void eval_error_wrapper4(double *c, 
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[15];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
    }
}

//* eval error wrapper (n = 5)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper5(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[21];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
         
    }
}

//* eval error wrapper (n = 6)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper6(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[28];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];
        r_c[21] = c[21 * num_elem + idx];
        r_c[22] = c[22 * num_elem + idx];
        r_c[23] = c[23 * num_elem + idx];
        r_c[24] = c[24 * num_elem + idx];
        r_c[25] = c[25 * num_elem + idx];
        r_c[26] = c[26 * num_elem + idx];
        r_c[27] = c[27 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
    }
}

//* eval error wrapper (n = 7)
//*
//* wrapper function for the eval_error device function.
//* THREADS: num_sides
__global__ void eval_error_wrapper7(double *c,
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p, double t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double r_c[36];

        // get the coefficients for this element
        r_c[0] = c[idx];
        r_c[1] = c[num_elem + idx];
        r_c[2] = c[2 * num_elem + idx];
        r_c[3] = c[3 * num_elem + idx];
        r_c[4] = c[4 * num_elem + idx];
        r_c[5] = c[5 * num_elem + idx];
        r_c[6] = c[6 * num_elem + idx];
        r_c[7] = c[7 * num_elem + idx];
        r_c[8] = c[8 * num_elem + idx];
        r_c[9] = c[9 * num_elem + idx];

        r_c[10] = c[10 * num_elem + idx];
        r_c[11] = c[11 * num_elem + idx];
        r_c[12] = c[12 * num_elem + idx];
        r_c[13] = c[13 * num_elem + idx];
        r_c[14] = c[14 * num_elem + idx];
        r_c[15] = c[15 * num_elem + idx];
        r_c[16] = c[16 * num_elem + idx];
        r_c[17] = c[17 * num_elem + idx];
        r_c[18] = c[18 * num_elem + idx];
        r_c[19] = c[19 * num_elem + idx];

        r_c[20] = c[20 * num_elem + idx];
        r_c[21] = c[21 * num_elem + idx];
        r_c[22] = c[22 * num_elem + idx];
        r_c[23] = c[23 * num_elem + idx];
        r_c[24] = c[24 * num_elem + idx];
        r_c[25] = c[25 * num_elem + idx];
        r_c[26] = c[26 * num_elem + idx];
        r_c[27] = c[27 * num_elem + idx];
        r_c[28] = c[28 * num_elem + idx];
        r_c[29] = c[29 * num_elem + idx];

        r_c[30] = c[30 * num_elem + idx];
        r_c[31] = c[31 * num_elem + idx];
        r_c[32] = c[32 * num_elem + idx];
        r_c[33] = c[33 * num_elem + idx];
        r_c[34] = c[34 * num_elem + idx];
        r_c[35] = c[35 * num_elem + idx];

        eval_error(r_c, V1x[idx], V1y[idx], V2x[idx], V2y[idx], V3x[idx], V3y[idx], 
                   Uv1, Uv2, Uv3,
                   num_elem, n_p, t, idx, alpha);
    }
}
