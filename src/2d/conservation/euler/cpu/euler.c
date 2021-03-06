#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "euler_kernels.c"
#include "time_integrator_euler.c"
#include "quadrature.c"
#include "basis.c"

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

    int i, j, items, s1, s2, s3, numsides, boundary_side, boundary;
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
        // and boundary_side tells which side is a boundary
        // while boundary determines the type of boundary
        items = sscanf(line, "%lf %lf %lf %lf %lf %lf %i %i", &V1x[i], &V1y[i], 
                                                              &V2x[i], &V2y[i], 
                                                              &V3x[i], &V3y[i], 
                                                              &boundary_side, &boundary);

        if (items != 8) {
            printf("error: not enough items (%i) while reading mesh.\n", items);
            exit(0);
        }

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

            // need to swap boundary sides 0 and 1 since we swapped sides 0 and 1
            if (boundary_side == 0) {
                boundary_side = 1;
            } else if (boundary_side == 1) {
                boundary_side = 0;
            }
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

            // see if this is a boundary side
            if (boundary_side == 0) {
                switch (boundary) {
                    case 10000: right_elem[numsides] = -1;
                                break;
                    case 20000: right_elem[numsides] = -2;
                                break;
                    case 30000: right_elem[numsides] = -3;
                                break;
                }
            }

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

            // see if this is a boundary side
            if (boundary_side == 1) {
                switch (boundary) {
                    case 10000: right_elem[numsides] = -1;
                                break;
                    case 20000: right_elem[numsides] = -2;
                                break;
                    case 30000: right_elem[numsides] = -3;
                                break;
                }
            }

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

            // see if this is a boundary side
            if (boundary_side == 2) {
                switch (boundary) {
                    case 10000: right_elem[numsides] = -1;
                                break;
                    case 20000: right_elem[numsides] = -2;
                                break;
                    case 30000: right_elem[numsides] = -3;
                                break;
                }
            }

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
    int reduction_size = (num_elem  / 256) + ((num_elem  % 256) ? 1 : 0);

    d_c = (double *) malloc(4 * num_elem * n_p * sizeof(double)); 
    d_c_prev = (double *) malloc(4 * num_elem * n_p * sizeof(double)); 
    d_quad_rhs = (double *) malloc(4 * num_elem * n_p * sizeof(double));
    d_left_riemann_rhs = (double *) malloc(4 * num_sides * n_p * sizeof(double));
    d_right_riemann_rhs = (double *) malloc(4 * num_sides * n_p * sizeof(double));

    d_kstar = (double *) malloc(4 * num_elem * n_p * sizeof(double));
    d_k1    = (double *) malloc(4 * num_elem * n_p * sizeof(double));
    d_k2    = (double *) malloc(4 * num_elem * n_p * sizeof(double));
    d_k3    = (double *) malloc(4 * num_elem * n_p * sizeof(double));
    d_k4    = (double *) malloc(4 * num_elem * n_p * sizeof(double));

    d_J         = (double *) malloc(num_elem * sizeof(double));
    d_lambda    = (double *) malloc(num_elem * sizeof(double));
    d_reduction = (double *) malloc(reduction_size * sizeof(double));
    d_s_length  = (double *) malloc(num_sides * sizeof(double));

    d_s_V1x = (double *) malloc(num_sides * sizeof(double));
    d_s_V2x = (double *) malloc(num_sides * sizeof(double));
    d_s_V1y = (double *) malloc(num_sides * sizeof(double));
    d_s_V2y = (double *) malloc(num_sides * sizeof(double));

    d_elem_s1 = (int *) malloc(num_elem * sizeof(int));
    d_elem_s2 = (int *) malloc(num_elem * sizeof(int));
    d_elem_s3 = (int *) malloc(num_elem * sizeof(int));

    d_Uv1 = (double *) malloc(num_elem * sizeof(double));
    d_Uv2 = (double *) malloc(num_elem * sizeof(double));
    d_Uv3 = (double *) malloc(num_elem * sizeof(double));

    d_V1x = (double *) malloc(num_elem * sizeof(double));
    d_V1y = (double *) malloc(num_elem * sizeof(double));
    d_V2x = (double *) malloc(num_elem * sizeof(double));
    d_V2y = (double *) malloc(num_elem * sizeof(double));
    d_V3x = (double *) malloc(num_elem * sizeof(double));
    d_V3y = (double *) malloc(num_elem * sizeof(double));

    d_xr = (double *) malloc(num_elem * sizeof(double));
    d_yr = (double *) malloc(num_elem * sizeof(double));
    d_xs = (double *) malloc(num_elem * sizeof(double));
    d_ys = (double *) malloc(num_elem * sizeof(double));

    d_left_side_number  = (int *) malloc(num_sides * sizeof(int));
    d_right_side_number = (int *) malloc( num_sides * sizeof(int));

    d_Nx = (double *) malloc(num_sides * sizeof(double));
    d_Ny = (double *) malloc(num_sides * sizeof(double));

    d_right_elem = (int *) malloc(num_sides * sizeof(int));
    d_left_elem  = (int *) malloc(num_sides * sizeof(int));

    // copy over data
    memcpy(d_s_V1x, sides_x1, num_sides * sizeof(double));
    memcpy(d_s_V1y, sides_y1, num_sides * sizeof(double));
    memcpy(d_s_V2x, sides_x2, num_sides * sizeof(double));
    memcpy(d_s_V2y, sides_y2, num_sides * sizeof(double));

    memcpy(d_left_side_number , left_side_number , num_sides * sizeof(int));
    memcpy(d_right_side_number, right_side_number, num_sides * sizeof(int));

    memcpy(d_elem_s1, elem_s1, num_elem * sizeof(int));
    memcpy(d_elem_s2, elem_s2, num_elem * sizeof(int));
    memcpy(d_elem_s3, elem_s3, num_elem * sizeof(int));

    memcpy(d_V1x, V1x, num_elem * sizeof(double));
    memcpy(d_V1y, V1y, num_elem * sizeof(double));
    memcpy(d_V2x, V2x, num_elem * sizeof(double));
    memcpy(d_V2y, V2y, num_elem * sizeof(double));
    memcpy(d_V3x, V3x, num_elem * sizeof(double));
    memcpy(d_V3y, V3y, num_elem * sizeof(double));

    memcpy(d_left_elem , left_elem , num_sides * sizeof(int));
    memcpy(d_right_elem, right_elem, num_sides * sizeof(int));
}

void free_gpu() {
    free(d_c);
    free(d_c_prev);
    free(d_quad_rhs);
    free(d_left_riemann_rhs);
    free(d_right_riemann_rhs);

    free(d_kstar);
    free(d_k1);
    free(d_k2);
    free(d_k3);
    free(d_k4);

    free(d_J);
    free(d_lambda);
    free(d_reduction);
    free(d_s_length);

    free(d_s_V1x);
    free(d_s_V2x);
    free(d_s_V1y);
    free(d_s_V2y);

    free(d_elem_s1);
    free(d_elem_s2);
    free(d_elem_s3);

    free(d_Uv1);
    free(d_Uv2);
    free(d_Uv3);

    free(d_V1x);
    free(d_V1y);
    free(d_V2x);
    free(d_V2y);
    free(d_V3x);
    free(d_V3y);

    free(d_xr);
    free(d_yr);
    free(d_xs);
    free(d_ys);

    free(d_left_side_number);
    free(d_right_side_number);

    free(d_Nx);
    free(d_Ny);

    free(d_right_elem);
    free(d_left_elem);
}

void usage_error() {
    printf("\nUsage: cpueuler [OPTIONS] [MESH] [OUTFILE]\n");
    printf(" Options: [-n] Order of polynomial approximation.\n");
    printf("          [-T] End time.\n");
    printf("          [-d] Debug.\n");
}

int get_input(int argc, char *argv[],
               int *n, int *timesteps, 
               double *endtime,
               char **mesh_filename, char **out_filename) {

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
    *mesh_filename = argv[argc - 2];
    // last argument is outfilename
    *out_filename  = argv[argc - 1];

    return 0;
}
