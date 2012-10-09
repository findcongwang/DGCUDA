/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper0(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, double *J,
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double rho_left[1], u_left[1], v_left[1], E_left[1];
        register double rho_right[1], u_right[1], v_right[1], E_right[1];

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
                s_length[idx], J[left_elem[idx]],
                V1x[left_elem[idx]], V1y[left_elem[idx]],
                V2x[left_elem[idx]], V2y[left_elem[idx]],
                V3x[left_elem[idx]], V3y[left_elem[idx]],
                left_elem[idx], right_elem[idx],
                left_side_number[idx], right_side_number[idx],
                Nx[idx], Ny[idx],
                n_quad1d, n_quad, n_p, num_sides, num_elem, t, idx);
    }
}


/* eval surface wrapper (n = 1)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper1(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, double *J,
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double rho_left[3], u_left[3], v_left[3], E_left[3];
        register double rho_right[3], u_right[3], v_right[3], E_right[3];
        int i;

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
                s_length[idx], J[left_elem[idx]],
                V1x[left_elem[idx]], V1y[left_elem[idx]],
                V2x[left_elem[idx]], V2y[left_elem[idx]],
                V3x[left_elem[idx]], V3y[left_elem[idx]],
                left_elem[idx], right_elem[idx],
                left_side_number[idx], right_side_number[idx],
                Nx[idx], Ny[idx],
                n_quad1d, n_quad, n_p, num_sides, num_elem, t, idx);
    }
}

/* eval surface wrapper (n = 2)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper2(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, double *J,
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double rho_left[6], u_left[6], v_left[6], E_left[6];
        register double rho_right[6], u_right[6], v_right[6], E_right[6];
        int i;

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
                s_length[idx], J[left_elem[idx]],
                V1x[left_elem[idx]], V1y[left_elem[idx]],
                V2x[left_elem[idx]], V2y[left_elem[idx]],
                V3x[left_elem[idx]], V3y[left_elem[idx]],
                left_elem[idx], right_elem[idx],
                left_side_number[idx], right_side_number[idx],
                Nx[idx], Ny[idx],
                n_quad1d, n_quad, n_p, num_sides, num_elem, t, idx);
    }
}


/* eval surface wrapper (n = 3)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper3(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, double *J,
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double rho_left[10], u_left[10], v_left[10], E_left[10];
        register double rho_right[10], u_right[10], v_right[10], E_right[10];
        int i;

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
                s_length[idx], J[left_elem[idx]],
                V1x[left_elem[idx]], V1y[left_elem[idx]],
                V2x[left_elem[idx]], V2y[left_elem[idx]],
                V3x[left_elem[idx]], V3y[left_elem[idx]],
                left_elem[idx], right_elem[idx],
                left_side_number[idx], right_side_number[idx],
                Nx[idx], Ny[idx],
                n_quad1d, n_quad, n_p, num_sides, num_elem, t, idx);
    }
}


/* eval surface wrapper (n = 4)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper4(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, double *J,
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double rho_left[15], u_left[15], v_left[15], E_left[15];
        register double rho_right[15], u_right[15], v_right[15], E_right[15];
        int i;

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
                s_length[idx], J[left_elem[idx]],
                V1x[left_elem[idx]], V1y[left_elem[idx]],
                V2x[left_elem[idx]], V2y[left_elem[idx]],
                V3x[left_elem[idx]], V3y[left_elem[idx]],
                left_elem[idx], right_elem[idx],
                left_side_number[idx], right_side_number[idx],
                Nx[idx], Ny[idx],
                n_quad1d, n_quad, n_p, num_sides, num_elem, t, idx);
    }
}


/* eval surface wrapper (n = 5)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */
__global__ void eval_surface_wrapper5(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, double *J,
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double rho_left[21], u_left[21], v_left[21], E_left[21];
        register double rho_right[21], u_right[21], v_right[21], E_right[21];
        int i;

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
                s_length[idx], J[left_elem[idx]],
                V1x[left_elem[idx]], V1y[left_elem[idx]],
                V2x[left_elem[idx]], V2y[left_elem[idx]],
                V3x[left_elem[idx]], V3y[left_elem[idx]],
                left_elem[idx], right_elem[idx],
                left_side_number[idx], right_side_number[idx],
                Nx[idx], Ny[idx],
                n_quad1d, n_quad, n_p, num_sides, num_elem, t, idx);
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
        double rho[1], u[1], v[1], E[1];

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
__global__ void eval_volume_wrapper1(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[3], u[3], v[3], E[3];
        int i;

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
__global__ void eval_volume_wrapper2(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[6], u[6], v[6], E[6];
        int i;

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
__global__ void eval_volume_wrapper3(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[10], u[10], v[10], E[10];
        int i;

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

__global__ void eval_volume_wrapper4(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[15], u[15], v[15], E[15];
        int i;

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
__global__ void eval_volume_wrapper5(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[21], u[21], v[21], E[21];
        int i;

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
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

//* eval lambda wrapper (n = 0)
//*
//* wrapper function for the eval_global_lambda device function.
//* THREADS: num_sides
 __global__ void eval_global_lambda_wrapper0(double *c, double *lambda, int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[1], u[1], v[1], E[1];

        // get the coefficients for this element
        rho[0] = c[num_elem * n_p * 0 + idx];
        u[0]   = c[num_elem * n_p * 1 + idx];
        v[0]   = c[num_elem * n_p * 2 + idx];
        E[0]   = c[num_elem * n_p * 3 + idx];

        eval_global_lambda(rho, u, v, E, lambda, n_quad, n_p, idx);
    }
}

//* eval lambda wrapper (n = 1)
//*
//* wrapper function for the eval_global_lambda device function.
//* THREADS: num_sides
 __global__ void eval_global_lambda_wrapper1(double *c, double *lambda, int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[3], u[3], v[3], E[3];
        int i;

        // get the coefficients for this element
        for (i = 0; i < 3; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_global_lambda(rho, u, v, E, lambda, n_quad, n_p, idx);
    }
}

//* eval lambda wrapper (n = 2)
//*
//* wrapper function for the eval_global_lambda device function.
//* THREADS: num_sides
 __global__ void eval_global_lambda_wrapper2(double *c, double *lambda, int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[6], u[6], v[6], E[6];
        int i;

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_global_lambda(rho, u, v, E, lambda, n_quad, n_p, idx);
         
    }
}
//* eval lambda wrapper (n = 3)
//*
//* wrapper function for the eval_global_lambda device function.
//* THREADS: num_sides
 __global__ void eval_global_lambda_wrapper3(double *c, double *lambda, int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
    if (idx < num_elem) {
        double rho[10], u[10], v[10], E[10];
        int i;

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_global_lambda(rho, u, v, E, lambda, n_quad, n_p, idx);
         
    }
}
//* eval lambda wrapper (n = 4)
//*
//* wrapper function for the eval_global_lambda device function.
//* THREADS: num_sides
 
 __global__ void eval_global_lambda_wrapper4(double *c, double *lambda, int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[15], u[15], v[15], E[15];
        int i;

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_global_lambda(rho, u, v, E, lambda, n_quad, n_p, idx);
         
    }
}
//* eval lambda wrapper (n = 5)
//*
//* wrapper function for the eval_global_lambda device function.
//* THREADS: num_sides
 __global__ void eval_global_lambda_wrapper5(double *c, double *lambda, int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double rho[21], u[21], v[21], E[21];
        int i;

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            rho[i] = c[num_elem * n_p * 0 + i * num_elem + idx];
            u[i]   = c[num_elem * n_p * 1 + i * num_elem + idx];
            v[i]   = c[num_elem * n_p * 2 + i * num_elem + idx];
            E[i]   = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_global_lambda(rho, u, v, E, lambda, n_quad, n_p, idx);
         
    }
}
//* eval u wrapper (n = 0)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_rho_wrapper0(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_rho[1];

        // get the coefficients for this element
        c_rho[0] = c[idx];

        eval_u(c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 1)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_rho_wrapper1(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_rho[3];

        // get the coefficients for this element
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];

        eval_u(c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 2)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_rho_wrapper2(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_rho[6];

        // get the coefficients for this element
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];
        c_rho[3] = c[3 * num_elem + idx];
        c_rho[4] = c[4 * num_elem + idx];
        c_rho[5] = c[5 * num_elem + idx];

        eval_u(c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 3)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_rho_wrapper3(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_rho[10];

        // get the coefficients for this element
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];
        c_rho[3] = c[3 * num_elem + idx];
        c_rho[4] = c[4 * num_elem + idx];
        c_rho[5] = c[5 * num_elem + idx];
        c_rho[6] = c[6 * num_elem + idx];
        c_rho[7] = c[7 * num_elem + idx];
        c_rho[8] = c[8 * num_elem + idx];
        c_rho[9] = c[9 * num_elem + idx];

        eval_u(c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 4)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
 
__global__ void eval_rho_wrapper4(double *c, 
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_rho[15];

        // get the coefficients for this element
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];
        c_rho[3] = c[3 * num_elem + idx];
        c_rho[4] = c[4 * num_elem + idx];
        c_rho[5] = c[5 * num_elem + idx];
        c_rho[6] = c[6 * num_elem + idx];
        c_rho[7] = c[7 * num_elem + idx];
        c_rho[8] = c[8 * num_elem + idx];
        c_rho[9] = c[9 * num_elem + idx];

        c_rho[10] = c[10 * num_elem + idx];
        c_rho[11] = c[11 * num_elem + idx];
        c_rho[12] = c[12 * num_elem + idx];
        c_rho[13] = c[13 * num_elem + idx];
        c_rho[14] = c[14 * num_elem + idx];

        eval_u(c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 5)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_rho_wrapper5(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_rho[21];

        // get the coefficients for this element
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];
        c_rho[3] = c[3 * num_elem + idx];
        c_rho[4] = c[4 * num_elem + idx];
        c_rho[5] = c[5 * num_elem + idx];
        c_rho[6] = c[6 * num_elem + idx];
        c_rho[7] = c[7 * num_elem + idx];
        c_rho[8] = c[8 * num_elem + idx];
        c_rho[9] = c[9 * num_elem + idx];

        c_rho[10] = c[10 * num_elem + idx];
        c_rho[11] = c[11 * num_elem + idx];
        c_rho[12] = c[12 * num_elem + idx];
        c_rho[13] = c[13 * num_elem + idx];
        c_rho[14] = c[14 * num_elem + idx];
        c_rho[15] = c[15 * num_elem + idx];
        c_rho[16] = c[16 * num_elem + idx];
        c_rho[17] = c[17 * num_elem + idx];
        c_rho[18] = c[18 * num_elem + idx];
        c_rho[19] = c[19 * num_elem + idx];

        c_rho[20] = c[20 * num_elem + idx];

        eval_u(c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

__global__ void eval_u_wrapper0(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_u[1];
        double c_rho[1];

        // get the coefficients for this element
        c_u[0]   = c[n_p * num_elem + idx];
        c_rho[0] = c[idx];

        eval_u_velocity(c_u, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
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
        double c_u[3];
        double c_rho[3];

        // get the coefficients for this element
        c_u[0] = c[n_p * num_elem + idx];
        c_u[1] = c[n_p * num_elem + num_elem + idx];
        c_u[2] = c[n_p * num_elem + 2 * num_elem + idx];
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];

        eval_u_velocity(c_u, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
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
        int i;
        double c_u[6];
        double c_rho[6];

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            c_u[i]   = c[num_elem * n_p + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_u, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
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
        int i;
        double c_u[10];
        double c_rho[10];

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            c_u[i] = c[num_elem * n_p + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_u, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
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
        int i;
        double c_u[15];
        double c_rho[15];

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            c_u[i] = c[num_elem * n_p + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_u, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
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
        int i;
        double c_u[21];
        double c_rho[21];

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            c_u[i] = c[num_elem * n_p + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_u, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval v wrapper (n = 0)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_v_wrapper0(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_v[1];
        double c_rho[1];

        // get the coefficients for this element
        c_v[0] = c[num_elem * n_p * 2 + idx];
        c_rho[0] = c[idx];

        eval_u_velocity(c_v, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 1)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_v_wrapper1(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_v[3];
        double c_rho[3];

        // get the coefficients for this element
        c_v[0] = c[num_elem * n_p * 2 + idx];
        c_v[1] = c[num_elem * n_p * 2 + num_elem + idx];
        c_v[2] = c[num_elem * n_p * 2 + 2 * num_elem + idx];
        c_rho[0] = c[idx];
        c_rho[1] = c[num_elem + idx];
        c_rho[2] = c[2 * num_elem + idx];

        eval_u_velocity(c_v, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 2)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_v_wrapper2(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_v[6];
        double c_rho[6];

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            c_v[i] = c[num_elem * n_p * 2 + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_v, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 3)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_v_wrapper3(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_v[10];
        double c_rho[10];

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            c_v[i] = c[num_elem * n_p * 2 + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_v, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 4)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
 
__global__ void eval_v_wrapper4(double *c, 
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_v[15];
        double c_rho[15];

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            c_v[i] = c[num_elem * n_p * 2 + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_v, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 5)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_v_wrapper5(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_v[21];
        double c_rho[21];

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            c_v[i] = c[num_elem * n_p * 2 + i * num_elem + idx];
            c_rho[i] = c[i * num_elem + idx];
        }

        eval_u_velocity(c_v, c_rho, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 0)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_E_wrapper0(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_E[1];

        // get the coefficients for this element
        c_E[0] = c[num_elem * n_p * 3 + idx];

        eval_u(c_E, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 1)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_E_wrapper1(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double c_E[3];

        // get the coefficients for this element
        c_E[0] = c[num_elem * n_p * 3 + idx];
        c_E[1] = c[num_elem * n_p * 3 + num_elem + idx];
        c_E[2] = c[num_elem * n_p * 3 + 2 * num_elem + idx];

        eval_u(c_E, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 2)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_E_wrapper2(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_E[6];

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            c_E[i] = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_u(c_E, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 3)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_E_wrapper3(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_E[10];

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            c_E[i] = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_u(c_E, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}

//* eval u wrapper (n = 4)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
 
__global__ void eval_E_wrapper4(double *c, 
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_E[15];

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            c_E[i] = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_u(c_E, Uv1, Uv2, Uv3, num_elem, n_p, idx);
    }
}

//* eval u wrapper (n = 5)
//*
//* wrapper function for the eval_u device function.
//* THREADS: num_sides
__global__ void eval_E_wrapper5(double *c,
                       double *Uv1, double *Uv2, double *Uv3,
                       int num_elem, int n_p) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        int i;
        double c_E[21];

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            c_E[i] = c[num_elem * n_p * 3 + i * num_elem + idx];
        }

        eval_u(c_E, Uv1, Uv2, Uv3, num_elem, n_p, idx);
         
    }
}
