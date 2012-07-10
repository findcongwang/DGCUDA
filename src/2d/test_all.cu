#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "2dadvec.cu"

#define TOL 1E-5
#define MAX_ALPHA 2

__device__ double *d_error;

/* test_all.cu
 * 
 * this is the standard test file for dgcuda.
 */

/* initial conditions
 *
 * find the initial projection for (x - y)^alpha
 * THREADS: num_elem
 */
__global__ void init_conditions_alpha(double *c, double *J,
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
                u += w[j] * pow(x - y, alpha) * basis[i * n_quad + j];
            }
            c[i * num_elem + idx] = u;
        } 
    }
}

/* evaluate error
 * 
 * evaluates u at the three vertex points for output
 * THREADS: num_elem
 */
__global__ void eval_error_alpha(double *c, 
                       double *V1x, double *V1y,
                       double *V2x, double *V2y,
                       double *V3x, double *V3y,
                       double *error, 
                       int num_elem, int n_p, int n_quad, double alpha) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) {
        int i, j;
        double u, sum;
        double x, y;

        sum = 0.;
        for (j = 0; j < n_quad; j++) {
            u = 0.;
            for (i = 0; i < n_p; i++) {
                u += c[i * num_elem + idx] * basis[n_quad * i + j];
            }
            // x = x2 * r + x3 * s + x1 * (1 - r - s)
            x = r1[j] * V2x[idx] + r2[j] * V3x[idx] + (1 - r1[j] - r2[j]) * V1x[idx];
            y = r1[j] * V2y[idx] + r2[j] * V3y[idx] + (1 - r1[j] - r2[j]) * V1y[idx];

            sum += w[j] * (pow(x - y, alpha) - u);
        }
        // store error
        error[idx] = abs(sum);
    }
}

__global__ void eval_rhs_fe(double *c, double *quad_rhs, double *left_riemann_rhs, double *right_riemann_rhs, 
                         int *elem_s1, int *elem_s2, int *elem_s3,
                         int *left_elem, double *J, 
                         double dt, int n_p, int num_sides, int num_elem) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double s1, s2, s3, register_J;
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
void time_integrate_fe(double dt, int n_quad, int n_quad1d, int n_p, int n, 
              int num_elem, int num_sides, double t) {
    int n_threads = 128;

    int n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);

    void (*eval_surface_ftn)(double*, double*, double*, 
                         double*,
                         double*, double*,
                         double*, double*,
                         double*, double*,
                         int*, int*,
                         int*, int*,
                         double*, double*,
                         int, int, int, int, double) = NULL;
    void (*eval_volume_ftn)(double*, double*, 
                        double*, double*, 
                        double*, double*,
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
        case 6: eval_surface_ftn = eval_surface_wrapper6;
                eval_volume_ftn  = eval_volume_wrapper6;
                break;
        case 7: eval_surface_ftn = eval_surface_wrapper7;
                eval_volume_ftn  = eval_volume_wrapper7;
                break;
    }
 
    if ((eval_surface_ftn == NULL) || (eval_volume_ftn == NULL)) {
        printf("ERROR: dispatched kernel functions were NULL.\n");
        exit(0);
    }

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

int test_initial_projection(int n, int alpha, FILE *mesh_file, FILE *out_file) {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int n_threads, n_blocks_elem;
    int i, n_p, n_quad, n_quad1d;

    double total_error; 
    double *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;
    double *sides_x1, *sides_x2;
    double *sides_y1, *sides_y2;

    double *r1_local, *r2_local, *w_local;

    double *s_r, *oned_w_local;

    int *left_elem, *right_elem;
    int *elem_s1, *elem_s2, *elem_s3;
    int *left_side_number, *right_side_number;

    char line[100];

    double *error;
    double *Uv1, *Uv2, *Uv3;

    void (*eval_u_ftn)(double*, double*, double*, double*, int, int) = NULL;

    // set the order of the approximation & timestep
    n_p = (n + 1) * (n + 2) / 2;

    // get the number of elements in the mesh
    fgets(line, 100, mesh_file);
    sscanf(line, "%i", &num_elem);

    // allocate vertex points
    V1x = (double *) malloc(num_elem * sizeof(double));
    V1y = (double *) malloc(num_elem * sizeof(double));
    V2x = (double *) malloc(num_elem * sizeof(double));
    V2y = (double *) malloc(num_elem * sizeof(double));
    V3x = (double *) malloc(num_elem * sizeof(double));
    V3y = (double *) malloc(num_elem * sizeof(double));

    elem_s1 = (int *) malloc(num_elem * sizeof(int));
    elem_s2 = (int *) malloc(num_elem * sizeof(int));
    elem_s3 = (int *) malloc(num_elem * sizeof(int));

    // TODO: these are too big; should be a way to figure out how many we actually need
    left_side_number  = (int *)   malloc(3*num_elem * sizeof(int));
    right_side_number = (int *)   malloc(3*num_elem * sizeof(int));

    sides_x1    = (double *) malloc(3*num_elem * sizeof(double));
    sides_x2    = (double *) malloc(3*num_elem * sizeof(double));
    sides_y1    = (double *) malloc(3*num_elem * sizeof(double));
    sides_y2    = (double *) malloc(3*num_elem * sizeof(double)); 
    left_elem   = (int *) malloc(3*num_elem * sizeof(int));
    right_elem  = (int *) malloc(3*num_elem * sizeof(int));

    for (i = 0; i < 3*num_elem; i++) {
        right_elem[i] = -1;
    }

    // read in the mesh and make all the mappings
    read_mesh(mesh_file, &num_sides, num_elem,
                         V1x, V1y, V2x, V2y, V3x, V3y,
                         left_side_number, right_side_number,
                         sides_x1, sides_y1, 
                         sides_x2, sides_y2, 
                         elem_s1, elem_s2, elem_s3,
                         left_elem, right_elem);

    // initialize the gpu
    init_gpu(num_elem, num_sides, n_p,
             V1x, V1y, V2x, V2y, V3x, V3y,
             left_side_number, right_side_number,
             sides_x1, sides_y1,
             sides_x2, sides_y2, 
             elem_s1, elem_s2, elem_s3,
             left_elem, right_elem);
    checkCudaError("error after gpu init.");

    n_threads        = 128;
    n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);

    // pre computations
    checkCudaError("error after prevals.");

    // get the correct quadrature rules for this scheme
    set_quadrature(n, &r1_local, &r2_local, &w_local, 
                   &s_r, &oned_w_local, &n_quad, &n_quad1d);

    preval_basis(r1_local, r2_local, s_r, w_local, oned_w_local, n_quad, n_quad1d, n_p);

    // initial conditions
    init_conditions_alpha<<<n_blocks_elem, n_threads>>>(d_c, d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y,
                    n_quad, n_p, num_elem, alpha);
    checkCudaError("error after initial conditions.");

    cudaThreadSynchronize();
    cudaMalloc((void **) &d_error, num_elem * sizeof(double));
    eval_error_alpha<<<n_blocks_elem, n_threads>>>(d_c, d_V1x, d_V1y, 
                                                   d_V2x, d_V2y, d_V3x, d_V3y,
                                                   d_error, num_elem, n_p, n_quad, alpha);

    switch (n) {
        case 0: eval_u_ftn = eval_u_wrapper0;
                break;
        case 1: eval_u_ftn = eval_u_wrapper1;
                break;
        case 2: eval_u_ftn = eval_u_wrapper2;
                break;
        case 3: eval_u_ftn = eval_u_wrapper3;
                break;
        case 4: eval_u_ftn = eval_u_wrapper4;
                break;
        case 5: eval_u_ftn = eval_u_wrapper5;
                break;
        case 6: eval_u_ftn = eval_u_wrapper6;
                break;
        case 7: eval_u_ftn = eval_u_wrapper7;
                break;
    }

    // evaluate at the vertex points and copy over data
    Uv1 = (double *) malloc(num_elem * sizeof(double));
    Uv2 = (double *) malloc(num_elem * sizeof(double));
    Uv3 = (double *) malloc(num_elem * sizeof(double));
    eval_u_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    // get the L1 error
    error = (double *) malloc(num_elem * sizeof(double));
    cudaMemcpy(error, d_error, num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    // get the total L1 error
    total_error = 0.;
    for (i = 0; i < num_elem; i++) {
        total_error += error[i];
    }

    // write data to file
    // TODO: this will output multiple vertices values. does gmsh care? i dunno...
    fprintf(out_file, "View \"Exported field \" {\n");
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%G,%G,0,%G,%G,0,%G,%G,0) {%G,%G,%G};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }

    // close the output file
    fprintf(out_file,"};");

    // free variables
    free_gpu();
    
    free(error);

    free(V1x);
    free(V1y);
    free(V2x);
    free(V2y);
    free(V3x);
    free(V3y);

    free(elem_s1);
    free(elem_s2);
    free(elem_s3);

    free(sides_x1);
    free(sides_x2);
    free(sides_y1);
    free(sides_y2);

    free(left_elem);
    free(right_elem);
    free(left_side_number);
    free(right_side_number);

    free(r1_local);
    free(r2_local);
    free(w_local);
    free(s_r);
    free(oned_w_local);
    
    printf(" (%lf) ", total_error);
    if (total_error < TOL) {
        return 0;
    } else {
        return 1;
    }
}

int test_timestep(int n, int alpha, int timesteps, double dt, FILE *mesh_file, FILE *out_file) {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int n_threads, n_blocks_elem, n_blocks_sides;
    int i, n_p, t, n_quad, n_quad1d;

    double *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;
    double *sides_x1, *sides_x2;
    double *sides_y1, *sides_y2;

    double *r1_local, *r2_local, *w_local;

    double *s_r, *oned_w_local;

    int *left_elem, *right_elem;
    int *elem_s1, *elem_s2, *elem_s3;
    int *left_side_number, *right_side_number;

    char line[100];

    double total_error;
    double *error;
    double *Uv1, *Uv2, *Uv3;

    void (*eval_u_ftn)(double*, double*, double*, double*, int, int) = NULL;

    // set the order of the approximation & timestep
    n_p = (n + 1) * (n + 2) / 2;

    // open the mesh to get num_elem for allocations
    if (!mesh_file) {
        printf("\nERROR: mesh file not found.\n");
        return 1;
    }
    fgets(line, 100, mesh_file);
    sscanf(line, "%i", &num_elem);

    // allocate vertex points
    V1x = (double *) malloc(num_elem * sizeof(double));
    V1y = (double *) malloc(num_elem * sizeof(double));
    V2x = (double *) malloc(num_elem * sizeof(double));
    V2y = (double *) malloc(num_elem * sizeof(double));
    V3x = (double *) malloc(num_elem * sizeof(double));
    V3y = (double *) malloc(num_elem * sizeof(double));

    elem_s1 = (int *) malloc(num_elem * sizeof(int));
    elem_s2 = (int *) malloc(num_elem * sizeof(int));
    elem_s3 = (int *) malloc(num_elem * sizeof(int));

    // TODO: these are too big; should be a way to figure out how many we actually need
    left_side_number  = (int *)   malloc(3*num_elem * sizeof(int));
    right_side_number = (int *)   malloc(3*num_elem * sizeof(int));

    sides_x1    = (double *) malloc(3*num_elem * sizeof(double));
    sides_x2    = (double *) malloc(3*num_elem * sizeof(double));
    sides_y1    = (double *) malloc(3*num_elem * sizeof(double));
    sides_y2    = (double *) malloc(3*num_elem * sizeof(double)); 
    left_elem   = (int *) malloc(3*num_elem * sizeof(int));
    right_elem  = (int *) malloc(3*num_elem * sizeof(int));

    for (i = 0; i < 3*num_elem; i++) {
        right_elem[i] = -1;
    }

    // read in the mesh and make all the mappings
    read_mesh(mesh_file, &num_sides, num_elem,
                         V1x, V1y, V2x, V2y, V3x, V3y,
                         left_side_number, right_side_number,
                         sides_x1, sides_y1, 
                         sides_x2, sides_y2, 
                         elem_s1, elem_s2, elem_s3,
                         left_elem, right_elem);

    // initialize the gpu
    init_gpu(num_elem, num_sides, n_p,
             V1x, V1y, V2x, V2y, V3x, V3y,
             left_side_number, right_side_number,
             sides_x1, sides_y1,
             sides_x2, sides_y2, 
             elem_s1, elem_s2, elem_s3,
             left_elem, right_elem);

    checkCudaError("error after gpu init.");
    n_threads        = 128;
    n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);

    // pre computations
    preval_jacobian<<<n_blocks_elem, n_threads>>>(d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, num_elem); 
    cudaThreadSynchronize();
    preval_side_length<<<n_blocks_sides, n_threads>>>(d_s_length, d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y, 
                                                      num_sides); 
    cudaThreadSynchronize();
    preval_normals<<<n_blocks_sides, n_threads>>>(d_Nx, d_Ny, 
                                                  d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y,
                                                  d_V1x, d_V1y, 
                                                  d_V2x, d_V2y, 
                                                  d_V3x, d_V3y, 
                                                  d_left_side_number, num_sides); 
    cudaThreadSynchronize();
    preval_normals_direction<<<n_blocks_sides, n_threads>>>(d_Nx, d_Ny, 
                                                  d_V1x, d_V1y, 
                                                  d_V2x, d_V2y, 
                                                  d_V3x, d_V3y, 
                                                  d_left_elem, d_left_side_number, num_sides); 
    preval_partials<<<n_blocks_elem, n_threads>>>(d_V1x, d_V1y,
                                                  d_V2x, d_V2y,
                                                  d_V3x, d_V3y,
                                                  d_xr,  d_yr,
                                                  d_xs,  d_ys, num_elem);
    cudaThreadSynchronize();
    checkCudaError("error after prevals.");

    // get the correct quadrature rules for this scheme
    set_quadrature(n, &r1_local, &r2_local, &w_local, 
                   &s_r, &oned_w_local, &n_quad, &n_quad1d);

    // evaluate the basis functions at those points and store on GPU
    preval_basis(r1_local, r2_local, s_r, w_local, oned_w_local, n_quad, n_quad1d, n_p);
    cudaThreadSynchronize();

    // initial conditions
    init_conditions_alpha<<<n_blocks_elem, n_threads>>>(d_c, d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y,
                    n_quad, n_p, num_elem, alpha);
    checkCudaError("error after initial conditions.");

    fprintf(out_file, "View \"Exported field \" {\n");

    for (i = 0; i < timesteps; i++) {
        t = dt * i;
        time_integrate_rk4(dt, n_quad, n_quad1d, n_p, n, num_elem, num_sides, 0, t);
    }

    // evaluate at the vertex points and copy over data
    Uv1 = (double *) malloc(num_elem * sizeof(double));
    Uv2 = (double *) malloc(num_elem * sizeof(double));
    Uv3 = (double *) malloc(num_elem * sizeof(double));

    switch (n) {
        case 0: eval_u_ftn = eval_u_wrapper0;
                break;
        case 1: eval_u_ftn = eval_u_wrapper1;
                break;
        case 2: eval_u_ftn = eval_u_wrapper2;
                break;
        case 3: eval_u_ftn = eval_u_wrapper3;
                break;
        case 4: eval_u_ftn = eval_u_wrapper4;
                break;
        case 5: eval_u_ftn = eval_u_wrapper5;
                break;
        case 6: eval_u_ftn = eval_u_wrapper6;
                break;
        case 7: eval_u_ftn = eval_u_wrapper7;
                break;
    }

    // get the L1 error
    error = (double *) malloc(num_elem * sizeof(double));
    cudaMalloc((void **) &d_error, num_elem * sizeof(double));
    eval_error_alpha<<<n_blocks_elem, n_threads>>>(d_c, d_V1x, d_V1y, 
                                                   d_V2x, d_V2y, d_V3x, d_V3y,
                                                   d_error, num_elem, n_p, n_quad, alpha);

    cudaMemcpy(error, d_error, num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    // get the total L1 error
    total_error = 0.;
    for (i = 0; i < num_elem; i++) {
        total_error += error[i];
    }

 
    eval_u_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaThreadSynchronize();
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    // write data to file
    // TODO: this will output multiple vertices values. does gmsh care? i dunno...
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%G,%G,0,%G,%G,0,%G,%G,0) {%G,%G,%G};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }

    fprintf(out_file,"};");

    // free variables
    free_gpu();
    
    free(Uv1);
    free(Uv2);
    free(Uv3);

    free(V1x);
    free(V1y);
    free(V2x);
    free(V2y);
    free(V3x);
    free(V3y);

    free(elem_s1);
    free(elem_s2);
    free(elem_s3);

    free(sides_x1);
    free(sides_x2);
    free(sides_y1);
    free(sides_y2);

    free(left_elem);
    free(right_elem);
    free(left_side_number);
    free(right_side_number);

    free(r1_local);
    free(r2_local);
    free(w_local);
    free(s_r);
    free(oned_w_local);

    free(error);

    printf(" (%lf) ", total_error);
    if (total_error < TOL) {
        return 0;
    } else {
        return 1;
    }
}

void run_initial_projection_tests() {
    int alpha, n;
    FILE *mesh, *out;
    printf("*************************\n");
    printf("* INITIAL PROJECTION\n");
    printf("*************************\n");
    ///////////////
    // CANONICAL
    ///////////////
    for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
        printf("*************************\n");
        printf("* u(x, y) = (x - y)^%i\n", alpha);
        printf("*************************\n");
        printf("    Testing canonical...\n");

        char outfilename[100];
        for (n = alpha; n < 6; n++) {
            sprintf(outfilename, "tests/t0/alpha%i_canonical%i.out", alpha, n);
            printf(outfilename);
            out  = fopen(outfilename, "w");
            mesh = fopen("mesh/canonical.pmsh", "r");
            printf("     > n = %i : ", n);

            if (!test_initial_projection(n, alpha, mesh, out)) {
                printf(" pass\n");
            } else {
                printf(" FAIL\n");
            }
            fclose(mesh);
            fclose(out);
        }
    }

    ///////////////
    // BOX
    ///////////////
    for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
        printf("*************************\n");
        printf("* u(x, y) = (x - y)^%i\n", alpha);
        printf("*************************\n");
        printf("    Testing box...\n");

        char outfilename[100];
        for (n = alpha; n < 6; n++) {
            sprintf(outfilename, "tests/t0/alpha%i_box%i.out", alpha, n);
            printf(outfilename);
            out  = fopen(outfilename, "w");
            mesh = fopen("mesh/box.pmsh", "r");
            printf("     > n = %i : ", n);

            if (!test_initial_projection(n, alpha, mesh, out)) {
                printf(" pass\n");
            } else {
                printf(" FAIL\n");
            }
            fclose(mesh);
            fclose(out);
        }
    }

    ///////////////
    // SUPERSIMPLE
    ///////////////
    for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
        printf("*************************\n");
        printf("* u(x, y) = (x - y)^%i\n", alpha);
        printf("*************************\n");
        printf("    Testing supersimple...\n");

        char outfilename[100];
        for (n = alpha; n < 6; n++) {
            sprintf(outfilename, "tests/t0/alpha%i_supersimple%i.out", alpha, n);
            printf(outfilename);
            out  = fopen(outfilename, "w");
            mesh = fopen("mesh/supersimple.pmsh", "r");
            printf("     > n = %i : ", n);

            if (!test_initial_projection(n, alpha, mesh, out)) {
                printf(" pass\n");
            } else {
                printf(" FAIL\n");
            }
            fclose(mesh);
            fclose(out);
        }
    }

    ///////////////
    // UNIFORM
    ///////////////
    for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
        printf("*************************\n");
        printf("* u(x, y) = (x - y)^%i\n", alpha);
        printf("*************************\n");
        printf("    Testing uniform...\n");

        char outfilename[100];
        for (n = alpha; n < 6; n++) {
            sprintf(outfilename, "tests/t0/alpha%i_uniform%i.out", alpha, n);
            printf(outfilename);
            out  = fopen(outfilename, "w");
            mesh = fopen("mesh/uniform.pmsh", "r");
            printf("     > n = %i : ", n);

            if (!test_initial_projection(n, alpha, mesh, out)) {
                printf(" pass\n");
            } else {
                printf(" FAIL\n");
            }
            fclose(mesh);
            fclose(out);
        }
    }
}

void run_timestep_tests() {
    int alpha, n, timesteps;
    double dt;
    FILE *mesh, *out;

    for (timesteps = 1; timesteps < 10; timesteps *= 10) {
        printf("*************************\n");
        printf("* TIMESTEPS : %i\n", timesteps);
        printf("*************************\n");

        ///////////////
        // CANONICAL
        ///////////////
        for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
            printf("*************************\n");
            printf("* u(x, y) = (x - y)^%i\n", alpha);
            printf("*************************\n");
            printf("    Testing canonical...\n");
            dt = 0.0001;

            char outfilename[100];
            for (n = alpha; n < 6; n++) {
                sprintf(outfilename, "tests/t%i/alpha%i_canonical%i.out", timesteps, alpha, n);
                printf(outfilename);
                out  = fopen(outfilename, "w");
                mesh = fopen("mesh/canonical.pmsh", "r");
                printf(" > n = %i : ", n);

                if (!test_timestep(n, alpha, timesteps, dt, mesh, out)) {
                    printf(" pass\n");
                } else {
                    printf(" FAIL\n");
                }
                fclose(mesh);
                fclose(out);
            }
        }
        ///////////////
        // BOX
        ///////////////
        for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
            printf("*************************\n");
            printf("* u(x, y) = (x - y)^%i\n", alpha);
            printf("*************************\n");
            printf("    Testing box...\n");
            dt = 0.01;

            char outfilename[100];
            for (n = alpha; n < 6; n++) {
                sprintf(outfilename, "tests/t%i/alpha%i_box%i.out", timesteps, alpha, n);
                printf(outfilename);
                out  = fopen(outfilename, "w");
                mesh = fopen("mesh/box.pmsh", "r");
                printf(" > n = %i : ", n);

                if (!test_timestep(n, alpha, timesteps, dt, mesh, out)) {
                    printf(" pass\n");
                } else {
                    printf(" FAIL\n");
                }
                fclose(mesh);
                fclose(out);
            }
        }
        ///////////////
        // UNIFORM
        ///////////////
        for (alpha = 0; alpha < MAX_ALPHA; alpha++) {
            printf("*************************\n");
            printf("* u(x, y) = (x - y)^%i\n", alpha);
            printf("*************************\n");
            printf("    Testing uniform...\n");
            dt = 0.01;

            char outfilename[100];
            for (n = alpha; n < 6; n++) {
                sprintf(outfilename, "tests/t%i/alpha%i_uniform%i.out", timesteps, alpha, n);
                printf(outfilename);
                out  = fopen(outfilename, "w");
                mesh = fopen("mesh/uniform.pmsh", "r");
                printf(" > n = %i : ", n);

                if (!test_timestep(n, alpha, timesteps, dt, mesh, out)) {
                    printf(" pass\n");
                } else {
                    printf(" FAIL\n");
                }
                fclose(mesh);
                fclose(out);
            }
        }
    }
}

int main() {
    //run_initial_projection_tests();
    run_timestep_tests();
}
