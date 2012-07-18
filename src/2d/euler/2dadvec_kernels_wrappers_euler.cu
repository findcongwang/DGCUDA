#ifndef WRAPPER_H_GUARD
#define WRAPPER_H_GUARD
void eval_surface(float*, float*,
                   float*, float*,
                   float,
                   float, float,
                   float, float,
                   float, float,
                   int, int,
                   float, float,
                   int, int, int, int, float, int, int);
#endif

/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper0(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                                      float *s_length, 
                                      float *V1x, float *V1y,
                                      float *V2x, float *V2y,
                                      float *V3x, float *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      float *Nx, float *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register float rho_left[1], u_left[1], v_left[1], E_left[1];
        register float rho_right[1], u_right[1], v_right[1], E_right[1];
        register float c_left[1], c_right[1];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            rho_left[0] = c[num_elem * n_p * 0 + left_elem[idx]];
            u_left[0]   = c[num_elem * n_p * 1 + left_elem[idx]];
            v_left[0]   = c[num_elem * n_p * 2 + left_elem[idx]];
            E_left[0]   = c[num_elem * n_p * 3 + left_elem[idx]];
            rho_right[0] = c[num_elem * n_p * 0 + right_elem[idx]];
            u_right[0]   = c[num_elem * n_p * 1 + right_elem[idx]];
            v_right[0]   = c[num_elem * n_p * 2 + right_elem[idx]];
            E_right[0]   = c[num_elem * n_p * 3 + right_elem[idx]];
        } else {
            rho_left[0] = c[num_elem * n_p * 0 + left_elem[idx]];
            u_left[0]   = c[num_elem * n_p * 1 + left_elem[idx]];
            v_left[0]   = c[num_elem * n_p * 2 + left_elem[idx]];
            E_left[0]   = c[num_elem * n_p * 3 + left_elem[idx]];
        }

        __syncthreads();

        eval_surface(rho_left, u_left, v_left, E_left,
                     rho_right, u_right, v_right, E_right,
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
__global__ void eval_surface_wrapper1(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                                      float *s_length, 
                                      float *V1x, float *V1y,
                                      float *V2x, float *V2y,
                                      float *V3x, float *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      float *Nx, float *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register float rho_left[3], u_left[3], v_left[3], E_left[3];
        register float rho_right[3], u_right[3], v_right[3], E_right[3];
        register float r_c_left[3], r_c_right[3];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 3; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
                rho_right[i] = c[num_elem * n_p * 0 + i * num_elem + right_elem[idx]];
                u_right[i]   = c[num_elem * n_p * 1 + i * num_elem + right_elem[idx]];
                v_right[i]   = c[num_elem * n_p * 2 + i * num_elem + right_elem[idx]];
                E_right[i]   = c[num_elem * n_p * 3 + i * num_elem + right_elem[idx]];
            }
        } else {
            for (i = 0; i < 3; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(rho_left, u_left, v_left, E_left,
                     rho_right, u_right, v_right, E_right,
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
__global__ void eval_surface_wrapper2(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                                      float *s_length, 
                                      float *V1x, float *V1y,
                                      float *V2x, float *V2y,
                                      float *V3x, float *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      float *Nx, float *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register float rho_left[6], u_left[6], v_left[6], E_left[6];
        register float rho_right[6], u_right[6], v_right[6], E_right[6];
        register float r_c_left[6], r_c_right[6];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 6; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
                rho_right[i] = c[num_elem * n_p * 0 + i * num_elem + right_elem[idx]];
                u_right[i]   = c[num_elem * n_p * 1 + i * num_elem + right_elem[idx]];
                v_right[i]   = c[num_elem * n_p * 2 + i * num_elem + right_elem[idx]];
                E_right[i]   = c[num_elem * n_p * 3 + i * num_elem + right_elem[idx]];
            }
        } else {
            for (i = 0; i < 6; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(rho_left, u_left, v_left, E_left,
                     rho_right, u_right, v_right, E_right,
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
__global__ void eval_surface_wrapper3(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                                      float *s_length, 
                                      float *V1x, float *V1y,
                                      float *V2x, float *V2y,
                                      float *V3x, float *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      float *Nx, float *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register float rho_left[10], u_left[10], v_left[10], E_left[10];
        register float rho_right[10], u_right[10], v_right[10], E_right[10];
        register float r_c_left[10], r_c_right[10];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 10; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
                rho_right[i] = c[num_elem * n_p * 0 + i * num_elem + right_elem[idx]];
                u_right[i]   = c[num_elem * n_p * 1 + i * num_elem + right_elem[idx]];
                v_right[i]   = c[num_elem * n_p * 2 + i * num_elem + right_elem[idx]];
                E_right[i]   = c[num_elem * n_p * 3 + i * num_elem + right_elem[idx]];
            }
        } else {
            for (i = 0; i < 10; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(rho_left, u_left, v_left, E_left,
                     rho_right, u_right, v_right, E_right,
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
__global__ void eval_surface_wrapper4(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                                      float *s_length, 
                                      float *V1x, float *V1y,
                                      float *V2x, float *V2y,
                                      float *V3x, float *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      float *Nx, float *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register float rho_left[15], u_left[15], v_left[15], E_left[15];
        register float rho_right[15], u_right[15], v_right[15], E_right[15];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 15; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
                rho_right[i] = c[num_elem * n_p * 0 + i * num_elem + right_elem[idx]];
                u_right[i]   = c[num_elem * n_p * 1 + i * num_elem + right_elem[idx]];
                v_right[i]   = c[num_elem * n_p * 2 + i * num_elem + right_elem[idx]];
                E_right[i]   = c[num_elem * n_p * 3 + i * num_elem + right_elem[idx]];
            }
        } else {
            for (i = 0; i < 15; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(rho_left, u_left, v_left, E_left,
                     rho_right, u_right, v_right, E_right,
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
__global__ void eval_surface_wrapper5(float *c, float *left_riemann_rhs, float *right_riemann_rhs, 
                                      float *s_length, 
                                      float *V1x, float *V1y,
                                      float *V2x, float *V2y,
                                      float *V3x, float *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      float *Nx, float *Ny, 
                                      int n_quad1d, int n_p, int num_sides, int num_elem, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register float rho_left[21], u_left[21], v_left[21], E_left[21];
        register float rho_right[21], u_right[21], v_right[21], E_right[21];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 21; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
                rho_right[i] = c[num_elem * n_p * 0 + i * num_elem + right_elem[idx]];
                u_right[i]   = c[num_elem * n_p * 1 + i * num_elem + right_elem[idx]];
                v_right[i]   = c[num_elem * n_p * 2 + i * num_elem + right_elem[idx]];
                E_right[i]   = c[num_elem * n_p * 3 + i * num_elem + right_elem[idx]];
            }
        } else {
            for (i = 0; i < 21; i++) {
                rho_left[i] = c[num_elem * n_p * 0 + i * num_elem + left_elem[idx]];
                u_left[i]   = c[num_elem * n_p * 1 + i * num_elem + left_elem[idx]];
                v_left[i]   = c[num_elem * n_p * 2 + i * num_elem + left_elem[idx]];
                E_left[i]   = c[num_elem * n_p * 3 + i * num_elem + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(rho_left, u_left, v_left, E_left,
                     rho_right, u_right, v_right, E_right,
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
 __global__ void eval_volume_wrapper0(float *c, float *quad_rhs, 
                                      float *xr, float *yr,
                                      float *xs, float *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float rho[1], u[1], v[1], E[1];

        // get the coefficients for this element
        rho[0] = c[num_elem * n_p * 0 + idx];
        u[0]   = c[num_elem * n_p * 1 + idx];
        v[0]   = c[num_elem * n_p * 2 + idx];
        E[0]   = c[num_elem * n_p * 3 + idx];

        eval_volume(rho, u, v, E, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
    }
}

//* eval volume wrapper (n = 1)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper1(float *c, float *quad_rhs, 
                                      float *xr, float *yr,
                                      float *xs, float *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float rho[3], u[3], v[3], E[3];

        // get the coefficients for this element
        for (i = 0; i < 3; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_volume(rho, u, v, E, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}

//* eval volume wrapper (n = 2)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper2(float *c, float *quad_rhs, 
                                      float *xr, float *yr,
                                      float *xs, float *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float rho[6], u[6], v[6], E[6];

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_volume(rho, u, v, E, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}
//* eval volume wrapper (n = 3)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper3(float *c, float *quad_rhs, 
                                      float *xr, float *yr,
                                      float *xs, float *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
    }
}
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float rho[10], u[10], v[10], E[10];

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_volume(rho, u, v, E, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}
//* eval volume wrapper (n = 4)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 
 __global__ void eval_volume_wrapper4(float *c, float *quad_rhs, 
                                      float *xr, float *yr,
                                      float *xs, float *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float rho[15], u[15], v[15], E[15];

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_volume(rho, u, v, E, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}
//* eval volume wrapper (n = 5)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
 __global__ void eval_volume_wrapper5(float *c, float *quad_rhs, 
                                      float *xr, float *yr,
                                      float *xs, float *ys,
                                      int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float rho[15], u[15], v[15], E[15];

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_volume(rho, u, v, E, quad_rhs,
                    xr[idx], yr[idx],
                    xs[idx], ys[idx],
                    n_quad, n_p, num_elem, idx);
         
    }
}
//* eval u wrapper (n = 0)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper0(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[1];

        // get the coefficients for this element
        r_c[0] = c[idx];

        eval_u(r_c, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 1)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_u_wrapper1(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[3];

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
__global__ void eval_u_wrapper2(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[6];

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
__global__ void eval_u_wrapper3(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[10];

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
 
__global__ void eval_u_wrapper4(float *c, 
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[15];

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
__global__ void eval_u_wrapper5(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[21];

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
__global__ void eval_u_wrapper6(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[28];

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
__global__ void eval_u_wrapper7(float *c,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[36];

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
__global__ void eval_error_wrapper0(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[1];

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
__global__ void eval_error_wrapper1(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[3];

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
__global__ void eval_error_wrapper2(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[6];

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
__global__ void eval_error_wrapper3(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[10];

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
 
__global__ void eval_error_wrapper4(float *c, 
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[15];

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
__global__ void eval_error_wrapper5(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[21];

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
__global__ void eval_error_wrapper6(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[28];

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
__global__ void eval_error_wrapper7(float *c,
                       float *V1x, float *V1y,
                       float *V2x, float *V2y,
                       float *V3x, float *V3y,
                       float *Uv1, float *Uv2, float *Uv3,
                       int num_elem, int n_p, float t, int alpha) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        float r_c[36];

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
