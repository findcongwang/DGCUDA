/* time_integrator.cu
 *
 * time integration functions.
 */
#ifndef TIMEINTEGRATOR_H_GUARD
#define TIMEINTEGRATOR_H_GUARD
void checkCudaError(const char*);
#endif

/***********************
 * RK4 
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

/* right hand side
 *
 * computes the sum of the quadrature and the riemann flux for the 
 * coefficients for each element
 * THREADS: num_elem
 */
__global__ void eval_rhs_rk4(float *c, float *quad_rhs, float *left_riemann_rhs, float *right_riemann_rhs, 
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

void time_integrate_rk4(float dt, int n_quad, int n_quad1d, int n_p, int n, 
                    int num_elem, int num_sides, int timesteps) {
    int n_threads = 128;
    int i;
    float t;

    int n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);
    int n_blocks_rk4     = ((n_p * num_elem) / n_threads) + (((n_p * num_elem) % n_threads) ? 1 : 0);

    void (*eval_surface_ftn)(float*, float*, float*, 
                         float*,
                         float*, float*,
                         float*, float*,
                         float*, float*,
                         int*, int*,
                         int*, int*,
                         float*, float*,
                         int, int, int, int, float) = NULL;
    void (*eval_volume_ftn)(float*, float*, 
                        float*, float*, 
                        float*, float*,
                        int, int, int) = NULL;
    switch (n) {
        case 0: eval_surface_ftn = eval_surface_wrapper0;
                eval_volume_ftn  = eval_volume_wrapper0;
                break;
        case 1: eval_surface_ftn = eval_surface_wrapper1;
                eval_volume_ftn  = eval_volume_wrapper1;
                break;
        case 2: eval_surface_ftn = eval_surface_wrapper2;
                eval_volume_ftn  = eval_volume_wrapper2;
                break;
        case 3: eval_surface_ftn = eval_surface_wrapper3;
                eval_volume_ftn  = eval_volume_wrapper3;
                break;
        case 4: eval_surface_ftn = eval_surface_wrapper4;
                eval_volume_ftn  = eval_volume_wrapper4;
                break;
        case 5: eval_surface_ftn = eval_surface_wrapper5;
                eval_volume_ftn  = eval_volume_wrapper5;
                break;
    }

    if ((eval_surface_ftn == NULL) || (eval_volume_ftn == NULL)) {
        printf("ERROR: dispatched kernel functions in rk4 were NULL.\n");
        exit(0);
    }

    for (i = 0; i < timesteps; i++) {
        t = i * dt;
        // stage 1
        checkCudaError("error before stage 1: eval_surface_ftn");
        eval_surface_ftn<<<n_blocks_sides, n_threads>>>
                        (d_c, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_p, num_sides, num_elem, t);

        checkCudaError("error after stage 1: eval_surface_ftn");

        eval_volume_ftn<<<n_blocks_elem, n_threads>>>
                        (d_c, d_quad_rhs, 
                         d_xr, d_yr, d_xs, d_ys,
                         n_quad, n_p, num_elem);
        cudaThreadSynchronize();

        eval_rhs_rk4<<<n_blocks_elem, n_threads>>>(d_k1, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt, n_p, num_sides, num_elem);
        cudaThreadSynchronize();

        rk4_tempstorage<<<n_blocks_rk4, n_threads>>>(d_c, d_kstar, d_k1, 0.5, n_p, num_elem);
        cudaThreadSynchronize();

        checkCudaError("error after stage 1.");

        // stage 2
        eval_surface_ftn<<<n_blocks_sides, n_threads>>>
                        (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_p, num_sides, num_elem, t);

        eval_volume_ftn<<<n_blocks_elem, n_threads>>>
                        (d_kstar, d_quad_rhs, 
                         d_xr, d_yr, d_xs, d_ys,
                         n_quad, n_p, num_elem);
        cudaThreadSynchronize();

        eval_rhs_rk4<<<n_blocks_elem, n_threads>>>(d_k2, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs,
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt, n_p, num_sides, num_elem);
        cudaThreadSynchronize();

        rk4_tempstorage<<<n_blocks_rk4, n_threads>>>(d_c, d_kstar, d_k2, 0.5, n_p, num_elem);
        cudaThreadSynchronize();

        checkCudaError("error after stage 2.");

        // stage 3
        eval_surface_ftn<<<n_blocks_sides, n_threads>>>
                        (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_p, num_sides, num_elem, t);

        eval_volume_ftn<<<n_blocks_elem, n_threads>>>
                        (d_kstar, d_quad_rhs, 
                         d_xr, d_yr, d_xs, d_ys,
                         n_quad, n_p, num_elem);
        cudaThreadSynchronize();

        eval_rhs_rk4<<<n_blocks_elem, n_threads>>>(d_k3, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt, n_p, num_sides, num_elem);
        cudaThreadSynchronize();

        rk4_tempstorage<<<n_blocks_rk4, n_threads>>>(d_c, d_kstar, d_k3, 1.0, n_p, num_elem);
        cudaThreadSynchronize();

        checkCudaError("error after stage 3.");

        // stage 4
        eval_surface_ftn<<<n_blocks_sides, n_threads>>>
                        (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length,
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_p, num_sides, num_elem, t);

        eval_volume_ftn<<<n_blocks_elem, n_threads>>>
                        (d_kstar, d_quad_rhs, 
                         d_xr, d_yr, d_xs, d_ys,
                         n_quad, n_p, num_elem);
        cudaThreadSynchronize();

        eval_rhs_rk4<<<n_blocks_elem, n_threads>>>(d_k4, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt, n_p, num_sides, num_elem);
        cudaThreadSynchronize();

        checkCudaError("error after stage 4.");
        
        // final stage
        rk4<<<n_blocks_rk4, n_threads>>>(d_c, d_k1, d_k2, d_k3, d_k4, n_p, num_elem);
        cudaThreadSynchronize();

        checkCudaError("error after final stage.");
    }
}

/***********************
 * FORWARD EULER
 ***********************/

__global__ void eval_rhs_fe(float *c, float *quad_rhs, float *left_riemann_rhs, float *right_riemann_rhs, 
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
            c[i * num_elem + idx] += 1. / register_J * dt * (quad_rhs[i * num_elem + idx] + s1 + s2 + s3);
        }
    }
}

// forward eulers
void time_integrate_fe(float dt, int n_quad, int n_quad1d, int n_p, int n, 
              int num_elem, int num_sides, int timesteps) {
    int n_threads = 128;
    int i;
    float t;

    int n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);

    void (*eval_surface_ftn)(float*, float*, float*, 
                         float*,
                         float*, float*,
                         float*, float*,
                         float*, float*,
                         int*, int*,
                         int*, int*,
                         float*, float*,
                         int, int, int, int, float) = NULL;
    void (*eval_volume_ftn)(float*, float*, 
                        float*, float*, 
                        float*, float*,
                        int, int, int) = NULL;
    switch (n) {
        case 0: eval_surface_ftn = eval_surface_wrapper0;
                eval_volume_ftn  = eval_volume_wrapper0;
                break;
        case 1: eval_surface_ftn = eval_surface_wrapper1;
                eval_volume_ftn  = eval_volume_wrapper1;
                break;
        case 2: eval_surface_ftn = eval_surface_wrapper2;
                eval_volume_ftn  = eval_volume_wrapper2;
                break;
        case 3: eval_surface_ftn = eval_surface_wrapper3;
                eval_volume_ftn  = eval_volume_wrapper3;
                break;
        case 4: eval_surface_ftn = eval_surface_wrapper4;
                eval_volume_ftn  = eval_volume_wrapper4;
                break;
        case 5: eval_surface_ftn = eval_surface_wrapper5;
                eval_volume_ftn  = eval_volume_wrapper5;
                break;
    }
    if ((eval_surface_ftn == NULL) || (eval_volume_ftn == NULL)) {
        printf("ERROR: dispatched kernel functions in fe were NULL.\n");
        exit(0);
    }

    for (i = 0; i < timesteps; i++) {
        t = i * dt;
        eval_surface_ftn<<<n_blocks_sides, n_threads>>>
                        (d_c, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_p, num_sides, num_elem, t);
        cudaThreadSynchronize();

        checkCudaError("error after eval_surface_ftn");

        eval_volume_ftn<<<n_blocks_elem, n_threads>>>
                        (d_c, d_quad_rhs, 
                         d_xr, d_yr, d_xs, d_ys,
                         n_quad, n_p, num_elem);
        cudaThreadSynchronize();

        eval_rhs_fe<<<n_blocks_elem, n_threads>>>(d_c, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                              d_elem_s1, d_elem_s2, d_elem_s3, 
                                              d_left_elem, d_J, dt, n_p, num_sides, num_elem);
        cudaThreadSynchronize();
    }
}
