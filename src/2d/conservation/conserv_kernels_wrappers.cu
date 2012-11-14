/////////////////////////////////////////////////////////////
//
//
//                  N = 1
//
//
/////////////////////////////////////////////////////////////

/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */

__global__ void eval_surface_wrapper1_0(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;

    if (idx < num_sides) {
        register double C_left[1];
        register double C_right[1];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
                C_right[n*n_p + 0] = c[num_elem * n_p * n + right_elem[idx]];
            }
        } else {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper1_1(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[3];
        register double C_right[3];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper1_2(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[6];
        register double C_right[6];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper1_3(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[10];
        register double C_right[10];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper1_4(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[15];
        register double C_right[15];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper1_5(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[21];
        register double C_right[21];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_volume_wrapper1_0(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;
    if (idx < num_elem) {
        double C[1];

        // get the coefficients for this element
        for (n = 0; n < N; n++) {
            C[n] = c[num_elem * n_p * n + idx];
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);
    }
}

//* eval volume wrapper (n = 1)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper1_1(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[3];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 3; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}

//* eval volume wrapper (n = 2)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper1_2(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[6];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 3)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper1_3(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[10];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 4)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides

__global__ void eval_volume_wrapper1_4(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[15];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}

//* eval volume wrapper (n = 5)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper1_5(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[21];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
/////////////////////////////////////////////////////////////
//
//
//                  N = 2
//
//
/////////////////////////////////////////////////////////////

/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */

__global__ void eval_surface_wrapper2_0(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;

    if (idx < num_sides) {
        register double C_left[2];
        register double C_right[2];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
                C_right[n*n_p + 0] = c[num_elem * n_p * n + right_elem[idx]];
            }
        } else {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper2_1(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[2 * 3];
        register double C_right[2 * 3];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper2_2(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[2 * 6];
        register double C_right[2 * 6];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper2_3(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[2 * 10];
        register double C_right[2 * 10];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper2_4(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[2 * 15];
        register double C_right[2 * 15];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper2_5(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[2 * 21];
        register double C_right[2 * 21];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_volume_wrapper2_0(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;
    if (idx < num_elem) {
        double C[2];

        // get the coefficients for this element
        for (n = 0; n < N; n++) {
            C[n] = c[num_elem * n_p * n + idx];
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);
    }
}

//* eval volume wrapper (n = 1)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper2_1(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[2 * 3];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 3; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}

//* eval volume wrapper (n = 2)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper2_2(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[2 * 6];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 3)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper2_3(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[2 * 10];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 4)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides

__global__ void eval_volume_wrapper2_4(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[2 * 15];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 5)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper2_5(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[2 * 21];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
/////////////////////////////////////////////////////////////
//
//
//                  N = 3
//
//
/////////////////////////////////////////////////////////////

/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */

__global__ void eval_surface_wrapper3_0(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;

    if (idx < num_sides) {
        register double C_left[3];
        register double C_right[3];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
                C_right[n*n_p + 0] = c[num_elem * n_p * n + right_elem[idx]];
            }
        } else {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper3_1(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[3 * 3];
        register double C_right[3 * 3];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper3_2(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[3 * 6];
        register double C_right[3 * 6];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper3_3(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[3 * 10];
        register double C_right[3 * 10];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper3_4(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[3 * 15];
        register double C_right[3 * 15];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper3_5(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[3 * 21];
        register double C_right[3 * 21];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_volume_wrapper3_0(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;
    if (idx < num_elem) {
        double C[3];

        // get the coefficients for this element
        for (n = 0; n < N; n++) {
            C[n] = c[num_elem * n_p * n + idx];
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);
    }
}

//* eval volume wrapper (n = 1)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper3_1(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[3 * 3];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 3; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}

//* eval volume wrapper (n = 2)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper3_2(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[3 * 6];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 3)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper3_3(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[3 * 10];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 4)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides

__global__ void eval_volume_wrapper3_4(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[3 * 15];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 5)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper3_5(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[3 * 21];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
/////////////////////////////////////////////////////////////
//
//
//                  N = 4
//
//
/////////////////////////////////////////////////////////////

/* eval surface wrapper (n = 0)
 *
 * wrapper function for the eval_surface device function.
 * THREADS: num_sides
 */

__global__ void eval_surface_wrapper4_0(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
                                      double *s_length, 
                                      double *V1x, double *V1y,
                                      double *V2x, double *V2y,
                                      double *V3x, double *V3y,
                                      int *left_elem, int *right_elem,
                                      int *left_side_number, int *right_side_number, 
                                      double *Nx, double *Ny, 
                                      int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;

    if (idx < num_sides) {
        register double C_left[4];
        register double C_right[4];

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
                C_right[n*n_p + 0] = c[num_elem * n_p * n + right_elem[idx]];
            }
        } else {
            for (n = 0; n < N; n++) {
                C_left[n*n_p + 0] = c[num_elem * n_p * n + left_elem[idx]];
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper4_1(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[4 * 3];
        register double C_right[4 * 3];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 3; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right,
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper4_2(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[4 * 6];
        register double C_right[4 * 6];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 6; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper4_3(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[4 * 10];
        register double C_right[4 * 10];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 10; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper4_4(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[4 * 15];
        register double C_right[4 * 15];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 15; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_surface_wrapper4_5(double *c, double *left_riemann_rhs, double *right_riemann_rhs, 
        double *s_length, 
        double *V1x, double *V1y,
        double *V2x, double *V2y,
        double *V3x, double *V3y,
        int *left_elem, int *right_elem,
        int *left_side_number, int *right_side_number, 
        double *Nx, double *Ny, 
        int n_quad1d, int n_quad, int n_p, int num_sides, int num_elem, double t) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_sides) {
        register double C_left[4 * 21];
        register double C_right[4 * 21];
        int i, n;

        // grab the coefficients for the left & right elements
        // TODO: group all the boundary sides together so they are in the same warp;
        //       means no warp divergence
        if (right_elem[idx] != -1) {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                    C_right[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + right_elem[idx]];
                }
            }
        } else {
            for (i = 0; i < 21; i++) {
                for (n = 0; n < N; n++) {
                    C_left[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + left_elem[idx]];
                }
            }
        }

        __syncthreads();

        eval_surface(C_left, C_right, 
                left_riemann_rhs, right_riemann_rhs,
                s_length[idx], 
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
__global__ void eval_volume_wrapper4_0(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n;
    if (idx < num_elem) {
        double C[4];

        // get the coefficients for this element
        for (n = 0; n < N; n++) {
            C[n] = c[num_elem * n_p * n + idx];
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);
    }
}

//* eval volume wrapper (n = 1)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper4_1(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[4 * 3];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 3; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}

//* eval volume wrapper (n = 2)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper4_2(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[4 * 6];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 6; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 3)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper4_3(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[4 * 10];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 10; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 4)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides

__global__ void eval_volume_wrapper4_4(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[4 * 15];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 15; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
//* eval volume wrapper (n = 5)
//*
//* wrapper function for the eval_volume device function.
//* THREADS: num_sides
__global__ void eval_volume_wrapper4_5(double *c, double *quad_rhs, 
        double *xr, double *yr,
        double *xs, double *ys,
        int n_quad, int n_p, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < num_elem) {
        double C[4 * 21];
        int i, n;

        // get the coefficients for this element
        for (i = 0; i < 21; i++) {
            for (n = 0; n < N; n++) {
                C[n*n_p + i] = c[num_elem * n_p * n + i * num_elem + idx];
            }
        }

        eval_volume(C, quad_rhs,
                xr[idx], yr[idx],
                xs[idx], ys[idx],
                n_quad, n_p, num_elem, idx);

    }
}
