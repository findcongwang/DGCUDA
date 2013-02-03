#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "conserv_kernels.cu"
#include "conserv_kernels_wrappers.cu"
//#include "conserv_kernels_wrappers.cu"
#include "time_integrator.cu"
#include "../quadrature.cu"
#include "../basis.cu"

extern int local_N;
extern int limiter;
extern int time_integrator;

// limiter optoins
#define NO_LIMITER 0
#define LIMITER 1

// time integration options
#define RK4 1
#define RK2 2

// riemann solver options
#define LLF 1

/* 2dadvec_euler.cu
 * 
 * This file calls the kernels in 2dadvec_kernels_euler.cu for the 2D advection
 * DG method.
 */

/* set quadrature 
 *
 * sets the 1d quadrature integration points and weights for the boundary integrals
 * and the 2d quadrature integration points and weights for the volume intergrals.
 */
void set_quadrature(int n,
                    double **r1_local, double **r2_local, double **w_local,
                    double **s_r, double **oned_w_local, 
                    int *n_quad, int *n_quad1d) {
    int i;
    /*
     * The sides are mapped to the canonical element, so we want the integration points
     * for the boundary integrals for sides s1, s2, and s3 as shown below:

     s (r2) |\
     ^      | \
     |      |  \
     |      |   \
     |   s3 |    \ s2
     |      |     \
     |      |      \
     |      |       \
     |      |________\
     |         s1
     |
     ------------------------> r (r1)

    *
    */
    switch (n) {
        case 0: *n_quad = 1;
                *n_quad1d = 1;
                break;
        case 1: *n_quad = 3;
                *n_quad1d = 2;
                break;
        case 2: *n_quad = 6;
                *n_quad1d = 3;
                break;
        case 3: *n_quad = 12 ;
                *n_quad1d = 4;
                break;
        case 4: *n_quad = 16;
                *n_quad1d = 5;
                break;
        case 5: *n_quad = 25;
                *n_quad1d = 6;
                break;
    }
    // allocate integration points
    *r1_local = (double *)  malloc(*n_quad * sizeof(double));
    *r2_local = (double *)  malloc(*n_quad * sizeof(double));
    *w_local  = (double *) malloc(*n_quad * sizeof(double));

    *s_r = (double *) malloc(*n_quad1d * sizeof(double));
    *oned_w_local = (double *) malloc(*n_quad1d * sizeof(double));

    // set 2D quadrature rules
    for (i = 0; i < *n_quad; i++) {
        if (n > 0) {
            (*r1_local)[i] = quad_2d[2 * n - 1][3*i];
            (*r2_local)[i] = quad_2d[2 * n - 1][3*i+1];
            (*w_local) [i] = quad_2d[2 * n - 1][3*i+2] / 2.; //weights are 2 times too big for some reason
        } else {
            (*r1_local)[i] = quad_2d[0][3*i];
            (*r2_local)[i] = quad_2d[0][3*i+1];
            (*w_local) [i] = quad_2d[0][3*i+2] / 2.; //weights are 2 times too big for some reason
        }
    }

    // set 1D quadrature rules
    for (i = 0; i < *n_quad1d; i++) {
        (*s_r)[i] = quad_1d[n][2*i];
        (*oned_w_local)[i] = quad_1d[n][2*i+1];
    }
}

void checkCudaError(const char *message)
{
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
        exit(-1);
    }
}

void read_mesh(FILE *mesh_file, 
              int *num_sides,
              int *num_elem,
              double **V1x, double **V1y,
              double **V2x, double **V2y,
              double **V3x, double **V3y,
              int **left_side_number, int **right_side_number,
              double **sides_x1, double **sides_y1,
              double **sides_x2, double **sides_y2,
              int **elem_s1,  int **elem_s2, int **elem_s3,
              int **left_elem, int **right_elem) {

    int i, items;
    char line[1000];
    // stores the number of sides this element has.

    fgets(line, 1000, mesh_file);
    sscanf(line, "%i", num_elem);
    *V1x = (double *) malloc(*num_elem * sizeof(double));
    *V1y = (double *) malloc(*num_elem * sizeof(double));
    *V2x = (double *) malloc(*num_elem * sizeof(double));
    *V2y = (double *) malloc(*num_elem * sizeof(double));
    *V3x = (double *) malloc(*num_elem * sizeof(double));
    *V3y = (double *) malloc(*num_elem * sizeof(double));

    *elem_s1 = (int *) malloc(*num_elem * sizeof(int));
    *elem_s2 = (int *) malloc(*num_elem * sizeof(int));
    *elem_s3 = (int *) malloc(*num_elem * sizeof(int));


    for (i = 0; i < *num_elem; i++) {
        fgets(line, sizeof(line), mesh_file);
        // these three vertices define the element
        // and boundary_side tells which side is a boundary
        // while boundary determines the type of boundary
        items = sscanf(line, "%lf %lf %lf %lf %lf %lf %i %i %i", &(*V1x)[i], &(*V1y)[i], 
                                         &(*V2x)[i], &(*V2y)[i], 
                                         &(*V3x)[i], &(*V3y)[i], 
                                         &(*elem_s1)[i], &(*elem_s2)[i], &(*elem_s3)[i]);

        if (items != 9) {
            printf("error: not enough items (%i) while reading elements from mesh.\n", items);
            exit(0);
        }

    }

    fgets(line, 1000, mesh_file);
    sscanf(line, "%i", num_sides);

    *left_side_number  = (int *)   malloc(*num_sides * sizeof(int));
    *right_side_number = (int *)   malloc(*num_sides * sizeof(int));

    *sides_x1    = (double *) malloc(*num_sides * sizeof(double));
    *sides_x2    = (double *) malloc(*num_sides * sizeof(double));
    *sides_y1    = (double *) malloc(*num_sides * sizeof(double));
    *sides_y2    = (double *) malloc(*num_sides * sizeof(double)); 

    *left_elem   = (int *) malloc(*num_sides * sizeof(int));
    *right_elem  = (int *) malloc(*num_sides * sizeof(int));

    for (i = 0; i < *num_sides; i++) {
        fgets(line, sizeof(line), mesh_file);
        items = sscanf(line, "%lf %lf %lf %lf %i %i %i %i", &(*sides_x1)[i], &(*sides_y1)[i],
                                            &(*sides_x2)[i], &(*sides_y2)[i],
                                            &(*left_elem)[i], &(*right_elem)[i],
                                            &(*left_side_number)[i],
                                            &(*right_side_number)[i]);

        if (items != 8) {
            printf("error: not enough items (%i) while reading sides from mesh.\n", items);
            exit(0);
        }
    }
}

void init_gpu(int num_elem, int num_sides, int n_p,
              double *V1x, double *V1y, 
              double *V2x, double *V2y, 
              double *V3x, double *V3y, 
              int *left_side_number, int *right_side_number,
              double *sides_x1, double *sides_y1,
              double *sides_x2, double *sides_y2,
              int *elem_s1, int *elem_s2, int *elem_s3,
              int *left_elem, int *right_elem) {
    //int reduction_size = (num_elem  / 256) + ((num_elem  % 256) ? 1 : 0);

    checkCudaError("error before init.");
    cudaDeviceReset();

    double total_memory = num_elem*22*sizeof(double)  +
                   num_elem*6*sizeof(int)      +
                   num_sides*11*sizeof(double) + 
                   num_sides*3*sizeof(int)     +
                   local_N*num_elem*n_p*3*sizeof(double) +
                   local_N*num_sides*n_p*2*sizeof(double);

    switch (time_integrator) {
        case RK4: total_memory += 5*local_N*num_elem*n_p*sizeof(double);
                  break;
        case RK2: total_memory += 3*local_N*num_elem*n_p*sizeof(double);
                  break;
    }

    if (total_memory < 1e3) {
        printf("Total memory required: %lf B\n", total_memory);
    } else if (total_memory >= 1e3 && total_memory < 1e6) {
        printf("Total memory required: %lf KB\n", total_memory * 1e-3);
    } else if (total_memory >= 1e6 && total_memory < 1e9) {
        printf("Total memory required: %lf MB\n", total_memory * 1e-6);
    } else {
        printf("Total memory required: %lf GB\n", total_memory * 1e-9);
    }

    cudaMalloc((void **) &d_c,        local_N * num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_c_prev,   local_N * num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_quad_rhs, local_N * num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_left_riemann_rhs,  local_N * num_sides * n_p * sizeof(double));
    cudaMalloc((void **) &d_right_riemann_rhs, local_N * num_sides * n_p * sizeof(double));

    switch (time_integrator) {
        case RK4: 
            cudaMalloc((void **) &d_kstar, local_N * num_elem * n_p * sizeof(double));
            cudaMalloc((void **) &d_k1,    local_N * num_elem * n_p * sizeof(double));
            cudaMalloc((void **) &d_k2,    local_N * num_elem * n_p * sizeof(double));
            cudaMalloc((void **) &d_k3,    local_N * num_elem * n_p * sizeof(double));
            cudaMalloc((void **) &d_k4,    local_N * num_elem * n_p * sizeof(double));
            break;
        case RK2: 
            cudaMalloc((void **) &d_kstar, local_N * num_elem * n_p * sizeof(double));
            cudaMalloc((void **) &d_k1,    local_N * num_elem * n_p * sizeof(double));
            break;
        default:
            printf("Error selecting time integrator.\n");
            exit(0);
    }



    cudaMalloc((void **) &d_J        , num_elem * sizeof(double));
    cudaMalloc((void **) &d_lambda   , num_elem * sizeof(double));
    cudaMalloc((void **) &d_s_length , num_sides * sizeof(double));

    cudaMalloc((void **) &d_s_V1x, num_sides * sizeof(double));
    cudaMalloc((void **) &d_s_V2x, num_sides * sizeof(double));
    cudaMalloc((void **) &d_s_V1y, num_sides * sizeof(double));
    cudaMalloc((void **) &d_s_V2y, num_sides * sizeof(double));

    cudaMalloc((void **) &d_elem_s1, num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s2, num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s3, num_elem * sizeof(int));

    cudaMalloc((void **) &d_Uv1, num_elem * sizeof(double));
    cudaMalloc((void **) &d_Uv2, num_elem * sizeof(double));
    cudaMalloc((void **) &d_Uv3, num_elem * sizeof(double));

    cudaMalloc((void **) &d_error, num_elem * sizeof(double));

    cudaMalloc((void **) &d_V1x, num_elem * sizeof(double));
    cudaMalloc((void **) &d_V1y, num_elem * sizeof(double));
    cudaMalloc((void **) &d_V2x, num_elem * sizeof(double));
    cudaMalloc((void **) &d_V2y, num_elem * sizeof(double));
    cudaMalloc((void **) &d_V3x, num_elem * sizeof(double));
    cudaMalloc((void **) &d_V3y, num_elem * sizeof(double));

    cudaMalloc((void **) &d_xr, num_elem * sizeof(double));
    cudaMalloc((void **) &d_yr, num_elem * sizeof(double));
    cudaMalloc((void **) &d_xs, num_elem * sizeof(double));
    cudaMalloc((void **) &d_ys, num_elem * sizeof(double));

    cudaMalloc((void **) &d_left_side_number , num_sides * sizeof(int));
    cudaMalloc((void **) &d_right_side_number, num_sides * sizeof(int));

    cudaMalloc((void **) &d_Nx, num_sides * sizeof(double));
    cudaMalloc((void **) &d_Ny, num_sides * sizeof(double));

    cudaMalloc((void **) &d_right_elem, num_sides * sizeof(int));
    cudaMalloc((void **) &d_left_elem , num_sides * sizeof(int));
    checkCudaError("error after gpu malloc");

    // copy over data
    cudaMemcpy(d_s_V1x, sides_x1, num_sides * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V1y, sides_y1, num_sides * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2x, sides_x2, num_sides * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2y, sides_y2, num_sides * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_side_number , left_side_number , num_sides * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_side_number, right_side_number, num_sides * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_elem_s1, elem_s1, num_elem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elem_s2, elem_s2, num_elem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elem_s3, elem_s3, num_elem * sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("error after gpu copy");

    cudaMemcpy(d_V1x, V1x, num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V1y, V1y, num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2x, V2x, num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2y, V2y, num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3x, V3x, num_elem * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3y, V3y, num_elem * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_elem , left_elem , num_sides * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_elem, right_elem, num_sides * sizeof(int), cudaMemcpyHostToDevice);
}

void free_gpu() {
    cudaFree(d_c);
    //cudaFree(d_c_prev);
    cudaFree(d_quad_rhs);
    cudaFree(d_left_riemann_rhs);
    cudaFree(d_right_riemann_rhs);

    switch (time_integrator) {
        case RK4: 
            cudaFree(d_kstar);
            cudaFree(d_k1);
            cudaFree(d_k2);
            cudaFree(d_k3);
            cudaFree(d_k4);
            break;
        case RK2:
            cudaFree(d_kstar);
            cudaFree(d_k1);
            break;
     }

    cudaFree(d_J);
    cudaFree(d_lambda);
    cudaFree(d_reduction);
    cudaFree(d_s_length);

    cudaFree(d_elem_s1);
    cudaFree(d_elem_s2);
    cudaFree(d_elem_s3);

    cudaFree(d_Uv1);
    cudaFree(d_Uv2);
    cudaFree(d_Uv3);

    cudaFree(d_V1x);
    cudaFree(d_V1y);
    cudaFree(d_V2x);
    cudaFree(d_V2y);
    cudaFree(d_V3x);
    cudaFree(d_V3y);

    cudaFree(d_xr);
    cudaFree(d_yr);
    cudaFree(d_xs);
    cudaFree(d_ys);

    cudaFree(d_left_side_number);
    cudaFree(d_right_side_number);

    cudaFree(d_Nx);
    cudaFree(d_Ny);

    cudaFree(d_right_elem);
    cudaFree(d_left_elem);
}

void usage_error() {
    printf("\nUsage: dgcuda [OPTIONS] [MESH] [OUTFILE]\n");
    printf(" Options: [-n] Order of polynomial approximation.\n");
    printf("          [-t] Number of timesteps.\n");
    printf("          [-d] Debug.\n");
}

int get_input(int argc, char *argv[],
               int *n, int *timesteps, 
               double *endtime,
               char **mesh_filename) {

    int i;

    *timesteps = 1;
    // read command line input
    if (argc < 5) {
        usage_error();
        return 1;
    }
    for (i = 0; i < argc; i++) {
        // order of polynomial
        if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) {
                *n = atoi(argv[i+1]);
                if (*n < 0 || *n > 5) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        // number of timesteps
        if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                *timesteps = atoi(argv[i+1]);
                if (*timesteps < 0) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
        if (strcmp(argv[i], "-T") == 0) {
            if (i + 1 < argc) {
                *endtime = atof(argv[i+1]);
                if (*endtime < 0) {
                    usage_error();
                    return 1;
                }
            } else {
                usage_error();
                return 1;
            }
        }
    } 

    // second last argument is filename
    *mesh_filename = argv[argc - 1];

    return 0;
}
