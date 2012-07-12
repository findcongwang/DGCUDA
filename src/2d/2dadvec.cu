#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "2dadvec_kernels.cu"
#include "2dadvec_kernels_wrappers.cu"
#include "quadrature.cu"
#include "basis.cu"
#include "time_integrator.cu"

/* 2dadvec.cu
 * 
 * This file calls the kernels in 2dadvec_kernels.cu for the 2D advection
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
        //case 6: *n_quad = 13;
                //*n_quad1d = 7;
                //break;
        //case 7: *n_quad = 16;
                //*n_quad1d = 8;
                //break;
        //case 8: *n_quad = 19;
                //*n_quad1d = 9;
                //break;
        //case 9: *n_quad = 25;
                //*n_quad1d = 10;
                //break;
    }
    // allocate integration points
    *r1_local = (double *)  malloc(*n_quad * sizeof(double));
    *r2_local = (double *)  malloc(*n_quad * sizeof(double));
    *w_local  =  (double *) malloc(*n_quad * sizeof(double));

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
              int num_elem,
              double *V1x, double *V1y,
              double *V2x, double *V2y,
              double *V3x, double *V3y,
              int *left_side_number, int *right_side_number,
              double *sides_x1, double *sides_y1,
              double *sides_x2, double *sides_y2,
              int *elem_s1,  int *elem_s2, int *elem_s3,
              int *left_elem, int *right_elem) {

    int i, j, s1, s2, s3, numsides;
    double J, tmpx, tmpy;
    char line[100];
    numsides = 0;
    // stores the number of sides this element has.
    int *total_sides = (int *) malloc(num_elem * sizeof(int));
    for (i = 0; i < num_elem; i++) {
        total_sides[i] = 0;
    }

    i = 0;
    while(fgets(line, sizeof(line), mesh_file) != NULL) {
        // these three vertices define the element
        sscanf(line, "%lf %lf %lf %lf %lf %lf", &V1x[i], &V1y[i], &V2x[i], &V2y[i], &V3x[i], &V3y[i]);
        //printf("(%lf, %lf, %lf, %lf, %lf, %lf)\n", V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i]);

        // determine whether we should add these three sides or not
        s1 = 1;
        s2 = 1;
        s3 = 1;

        // enforce strictly positive jacobian
        J = (V2x[i] - V1x[i]) * (V3y[i] - V1y[i]) - (V3x[i] - V1x[i]) * (V2y[i] - V1y[i]);
        if (J < 0) {
            tmpx = V1x[i];
            tmpy = V1y[i];
            V1x[i] = V2x[i];
            V1y[i] = V2y[i];
            V2x[i] = tmpx;
            V2y[i] = tmpy;
        }

        // scan through the existing sides to see if we already added it
        // TODO: yeah, there's a better way to do this.
        // TODO: Also, this is super sloppy. should be checking indices instead of double values.
        for (j = 0; j < numsides; j++) {
            // side 1
            if (s1 && ((sides_x1[j] == V1x[i] && sides_y1[j] == V1y[i]
             && sides_x2[j] == V2x[i] && sides_y2[j] == V2y[i]) 
            || (sides_x2[j] == V1x[i] && sides_y2[j] == V1y[i]
             && sides_x1[j] == V2x[i] && sides_y1[j] == V2y[i]))) {
                s1 = 0;
                // OK, we've added this side to element i
                right_elem[j] = i;
                // link the added side j to this element
                elem_s1[i] = j;
                right_side_number[j] = 0;
                break;
            }
        }
        for (j = 0; j < numsides; j++) {
            // side 2
            if (s2 && ((sides_x1[j] == V2x[i] && sides_y1[j] == V2y[i]
             && sides_x2[j] == V3x[i] && sides_y2[j] == V3y[i]) 
            || (sides_x2[j] == V2x[i] && sides_y2[j] == V2y[i]
             && sides_x1[j] == V3x[i] && sides_y1[j] == V3y[i]))) {
                s2 = 0;
                // OK, we've added this side to some element before; which one?
                right_elem[j] = i;
                elem_s2[i] = j;
                // link the added side to this element
                right_side_number[j] = 1;
                break;
            }
        }
        for (j = 0; j < numsides; j++) {
            // side 3
            if (s3 && ((sides_x1[j] == V1x[i] && sides_y1[j] == V1y[i]
             && sides_x2[j] == V3x[i] && sides_y2[j] == V3y[i]) 
            || (sides_x2[j] == V1x[i] && sides_y2[j] == V1y[i]
             && sides_x1[j] == V3x[i] && sides_y1[j] == V3y[i]))) {
                s3 = 0;
                // OK, we've added this side to some element before; which one?
                right_elem[j] = i;
                elem_s3[i] = j;
                // link the added side to this element
                right_side_number[j] = 2;
                break;
            }
        }
        // if we haven't added the side already, add it
        if (s1) {
            sides_x1[numsides] = V1x[i];
            sides_y1[numsides] = V1y[i];
            sides_x2[numsides] = V2x[i];
            sides_y2[numsides] = V2y[i];

            // link the added side to this element
            left_side_number[numsides] = 0;
            // and link the element to this side
            elem_s1[i] = numsides;

            // make this the left element
            left_elem[numsides] = i;
            numsides++;
        }
        if (s2) {
            sides_x1[numsides] = V2x[i];
            sides_y1[numsides] = V2y[i];
            sides_x2[numsides] = V3x[i];
            sides_y2[numsides] = V3y[i];

            // link the added side to this element
            left_side_number[numsides] = 1;
            // and link the element to this side
            elem_s2[i] = numsides;

            // make this the left element
            left_elem[numsides] = i;
            numsides++;
        }
        if (s3) {
            sides_x1[numsides] = V3x[i];
            sides_y1[numsides] = V3y[i];
            sides_x2[numsides] = V1x[i];
            sides_y2[numsides] = V1y[i];

            // link the added side to this element
            left_side_number[numsides] = 2;
            // and link the element to this side
            elem_s3[i] = numsides;

            // make this the left element
            left_elem[numsides] = i;
            numsides++;
        }
        i++;
    }
    //free(total_sides);
    *num_sides = numsides;
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
    checkCudaError("error before init.");
    cudaDeviceReset();

    cudaMalloc((void **) &d_c,        num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_quad_rhs, num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_left_riemann_rhs,  num_sides * n_p * sizeof(double));
    cudaMalloc((void **) &d_right_riemann_rhs, num_sides * n_p * sizeof(double));

    cudaMalloc((void **) &d_kstar, num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_k1, num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_k2, num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_k3, num_elem * n_p * sizeof(double));
    cudaMalloc((void **) &d_k4, num_elem * n_p * sizeof(double));

    cudaMalloc((void **) &d_J, num_elem * sizeof(double));
    cudaMalloc((void **) &d_s_length, num_sides * sizeof(double));

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

    // set d_c to 0 not necessary
    //cudaMemset(d_c, 0., num_elem * n_p * sizeof(double));
    cudaMemset(d_quad_rhs, 0., num_elem * n_p * sizeof(double));

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
    checkCudaError("error inside gpu init.");

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
    cudaFree(d_quad_rhs);
    cudaFree(d_left_riemann_rhs);
    cudaFree(d_right_riemann_rhs);

    cudaFree(d_kstar);
    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);

    cudaFree(d_J);
    cudaFree(d_s_length);

    cudaFree(d_s_V1x);
    cudaFree(d_s_V2x);
    cudaFree(d_s_V1y);
    cudaFree(d_s_V2y);

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
               int *n, int *debug, int *timesteps, int *alpha,
               char **mesh_filename, char **out_filename) {

    int i;

    *timesteps = 1;
    *debug     = 0;
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
        if (strcmp(argv[i], "-d") == 0) {
            *debug = 1;
        }
        if (strcmp(argv[i], "-a") == 0) {
            if (i + 1 < argc) {
                *alpha = atoi(argv[i+1]);
                if (*alpha < 0) {
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
    *mesh_filename = argv[argc - 2];
    // last argument is outfilename
    *out_filename  = argv[argc - 1];

    return 0;
}

void test_initial_condition(double *c,
                            double *V1x, double *V1y,
                            double *V2x, double *V2y,
                            double *V3x, double *V3y,
                            double *r1_local, double *r2_local,
                            double *w_local, double *basis_local, int n_quad, int n_p) {   

    int i, j;
    double u, x, y;

    for (i = 0; i < n_p; i++) {
        u = 0.;
        // perform quadrature
        for (j = 0; j < n_quad; j++) {
            // map from the canonical element to the actual point on the mesh
            // x = x2 * r + x3 * s + x1 * (1 - r - s)
            x = r1_local[j] * V2x[0] + r2_local[j] * V3x[0] + (1 - r1_local[j] - r2_local[j]) * V1x[0];
            y = r1_local[j] * V2y[0] + r2_local[j] * V3y[0] + (1 - r1_local[j] - r2_local[j]) * V1y[0];

                // evaluate u there
            u += w_local[j] * pow(x - y, 2) * basis_local[i * n_quad + j];
        }
        c[i] = u;
        printf("c[%i] = %lf\n", i, c[i]);
    }
}

