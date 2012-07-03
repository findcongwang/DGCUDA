#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "2dadvec_kernels.cu"
#include "quadrature.h"
#include "basis.h"

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
void set_quadrature(int p,
                    float **r1_local, float **r2_local, float **w_local,
                    float **s_r, float **oned_w_local, 
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
    switch (p) {
        case 0: *n_quad = 1;
                *n_quad1d = 1;
                break;
        case 1: *n_quad = 3;
                *n_quad1d = 2;
                break;
        case 2: *n_quad = 4;
                *n_quad1d = 3;
                break;
        case 3: *n_quad = 6 ;
                *n_quad1d = 4;
                break;
        case 4: *n_quad = 7;
                *n_quad1d = 5;
                break;
        case 5: *n_quad = 12;
                *n_quad1d = 6;
                break;
        case 6: *n_quad = 13;
                *n_quad1d = 7;
                break;
        case 7: *n_quad = 16;
                *n_quad1d = 8;
                break;
        case 8: *n_quad = 19;
                *n_quad1d = 9;
                break;
        case 9: *n_quad = 25;
                *n_quad1d = 10;
                break;
    }
    // allocate integration points
    *r1_local = (float *) malloc(*n_quad * sizeof(float));
    *r2_local = (float *) malloc(*n_quad * sizeof(float));
    *w_local  =  (float *) malloc(*n_quad * sizeof(float));

    *s_r = (float *) malloc(*n_quad1d * sizeof(float));
    *oned_w_local = (float *) malloc(*n_quad1d * sizeof(float));

    // set 2D quadrature rules
    for (i = 0; i < *n_quad; i++) {
        (*r1_local)[i] = quad_2d[p][3*i];
        (*r2_local)[i] = quad_2d[p][3*i+1];
        (*w_local) [i] = quad_2d[p][3*i+2] / 2.; //weights are 2 times too big for some reason
    }

    // set 1D quadrature rules
    for (i = 0; i < *n_quad1d; i++) {
        (*s_r)[i] = quad_1d[p][2*i];
        (*oned_w_local)[i] = quad_1d[p][2*i+1];
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
              float *V1x, float *V1y,
              float *V2x, float *V2y,
              float *V3x, float *V3y,
              int *left_side_number, int *right_side_number,
              float *sides_x1, float *sides_y1,
              float *sides_x2, float *sides_y2,
              int *elem_s1,  int *elem_s2, int *elem_s3,
              int *left_elem, int *right_elem) {

    int i, j, s1, s2, s3, numsides;
    float J, tmpx, tmpy;
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
        sscanf(line, "%f %f %f %f %f %f", &V1x[i], &V1y[i], &V2x[i], &V2y[i], &V3x[i], &V3y[i]);

        // determine whether we should add these three sides or not
        s1 = 1;
        s2 = 1;
        s3 = 1;

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
        // TODO: Also, this is super sloppy. should be checking indices instead of float values.
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
            //third_x[numsides] = V3x[i];
            //third_y[numsides] = V3y[i];

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

void time_integrate(float dt, int n_quad, int n_quad1d, int n_p, int num_elem, int num_sides, int debug) {
    int n_threads = 256;

    int n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);

    // stage 1
    checkCudaError("error before stage 1: eval_riemann");
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_c, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s_r,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    if (debug) {
        printf("\n\n dt = %f -\n", dt);
        printf("-------------------------\n");
        float *left_rhs = (float *) malloc(num_sides * n_p * sizeof(float));
        float *right_rhs = (float *) malloc(num_sides * n_p * sizeof(float));
        cudaMemcpy(left_rhs, d_left_riemann_rhs, num_sides * n_p * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(right_rhs, d_right_riemann_rhs, num_sides * n_p * sizeof(float), cudaMemcpyDeviceToHost);
        printf(" riemann\n");
        printf(" ~~~\n");
        for (int i = 0; i < num_sides * n_p; i++) {
            if (i != 0 && i % num_sides == 0) {
                printf("   --- \n");
            }
            printf(" > (%f, %f) \n", left_rhs[i], right_rhs[i]);
        }
        free(left_rhs);
        free(right_rhs);
    }

    checkCudaError("error after stage 1: eval_riemann");

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_c, d_quad_rhs, d_J,
                     d_xr, d_yr, d_xs, d_ys,
                     n_quad, n_p, num_elem);
    cudaThreadSynchronize();

    if (debug) {
        float *quad_rhs = (float *) malloc(num_elem * n_p * sizeof(float));
        cudaMemcpy(quad_rhs, d_quad_rhs, num_elem * n_p * sizeof(float), cudaMemcpyDeviceToHost);
        printf(" quad_rhs\n");
        printf(" ~~~\n");
        for (int i = 0; i < num_elem * n_p; i++) {
            if (i != 0 && i % num_elem == 0) {
                printf("   --- \n");
            }
            printf(" > %f \n", quad_rhs[i]);
            }
        free(quad_rhs);
    }

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k1, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, d_J, dt, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    if (debug) {
        float *rhs = (float *) malloc(num_elem * n_p * sizeof(float));
        cudaMemcpy(rhs, d_k1, num_elem * n_p * sizeof(float), cudaMemcpyDeviceToHost);
        printf(" eval_rhs\n");
        printf(" ~~~\n");
        for (int i = 0; i < num_elem * n_p; i++) {
            if (i != 0 && i % num_elem == 0) {
                printf("   --- \n");
            }
            printf(" > %f \n", rhs[i]);
            }
        free(rhs);
    }

    rk4_tempstorage<<<n_blocks_elem, n_threads>>>(d_c, d_kstar, d_k1, 0.5, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 1.");

    // stage 2
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s_r,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_c, d_quad_rhs, d_J,
                     d_xr, d_yr, d_xs, d_ys,
                     n_quad, n_p, num_elem);
    cudaThreadSynchronize();

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k2, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs,
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, d_J, dt, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    rk4_tempstorage<<<n_blocks_elem, n_threads>>>(d_c, d_kstar, d_k2, 0.5, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 2.");

    // stage 3
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s_r,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_c, d_quad_rhs, d_J,
                     d_xr, d_yr, d_xs, d_ys,
                     n_quad, n_p, num_elem);
    cudaThreadSynchronize();

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k3, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, d_J, dt, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    rk4_tempstorage<<<n_blocks_elem, n_threads>>>(d_c, d_kstar, d_k3, 1.0, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 3.");

    // stage 4
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s_r,
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_c, d_quad_rhs, d_J,
                     d_xr, d_yr, d_xs, d_ys,
                     n_quad, n_p, num_elem);
    cudaThreadSynchronize();

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k4, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, d_J, dt, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 4.");
    
    // final stage
    rk4<<<n_blocks_elem, n_threads>>>(d_c, d_k1, d_k2, d_k3, d_k4, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after final stage.");
}

void init_gpu(int num_elem, int num_sides, int n_p,
              float *V1x, float *V1y, 
              float *V2x, float *V2y, 
              float *V3x, float *V3y, 
              int *left_side_number, int *right_side_number,
              float *sides_x1, float *sides_y1,
              float *sides_x2, float *sides_y2,
              int *elem_s1, int *elem_s2, int *elem_s3,
              int *left_elem, int *right_elem) {
    checkCudaError("error before init.");
    cudaDeviceReset();

    // allocate allllllllllll the memory.
    // TODO: this takes a really really long time on valor.
    cudaMalloc((void **) &d_c,        num_elem * n_p * sizeof(float));
    cudaMalloc((void **) &d_quad_rhs, num_elem * n_p * sizeof(float));
    cudaMalloc((void **) &d_left_riemann_rhs,  num_sides * n_p * sizeof(float));
    cudaMalloc((void **) &d_right_riemann_rhs, num_sides * n_p * sizeof(float));

    cudaMalloc((void **) &d_kstar, num_elem * n_p * sizeof(float));
    cudaMalloc((void **) &d_k1, num_elem * n_p * sizeof(float));
    cudaMalloc((void **) &d_k2, num_elem * n_p * sizeof(float));
    cudaMalloc((void **) &d_k3, num_elem * n_p * sizeof(float));
    cudaMalloc((void **) &d_k4, num_elem * n_p * sizeof(float));

    cudaMalloc((void **) &d_J, num_elem * sizeof(float));
    cudaMalloc((void **) &d_s_length, num_sides * sizeof(float));

    cudaMalloc((void **) &d_s_V1x, num_sides * sizeof(float));
    cudaMalloc((void **) &d_s_V2x, num_sides * sizeof(float));
    cudaMalloc((void **) &d_s_V1y, num_sides * sizeof(float));
    cudaMalloc((void **) &d_s_V2y, num_sides * sizeof(float));

    cudaMalloc((void **) &d_elem_s1, num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s2, num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s3, num_elem * sizeof(int));

    cudaMalloc((void **) &d_Uv1, num_elem * sizeof(float));
    cudaMalloc((void **) &d_Uv2, num_elem * sizeof(float));
    cudaMalloc((void **) &d_Uv3, num_elem * sizeof(float));

    cudaMalloc((void **) &d_V1x, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V1y, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V2x, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V2y, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V3x, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V3y, num_elem * sizeof(float));

    cudaMalloc((void **) &d_xr, num_elem * sizeof(float));
    cudaMalloc((void **) &d_yr, num_elem * sizeof(float));
    cudaMalloc((void **) &d_xs, num_elem * sizeof(float));
    cudaMalloc((void **) &d_ys, num_elem * sizeof(float));

    cudaMalloc((void **) &d_s_r, n_p * sizeof(float));
    
    cudaMalloc((void **) &d_left_side_number , num_sides * sizeof(int));
    cudaMalloc((void **) &d_right_side_number, num_sides * sizeof(int));

    cudaMalloc((void **) &d_Nx, num_sides * sizeof(float));
    cudaMalloc((void **) &d_Ny, num_sides * sizeof(float));

    cudaMalloc((void **) &d_right_elem, num_sides * sizeof(int));
    cudaMalloc((void **) &d_left_elem , num_sides * sizeof(int));

    // set d_c to 0 not necessary
    //cudaMemset(d_c, 0., num_elem * n_p * sizeof(float));
    cudaMemset(d_quad_rhs, 0., num_elem * n_p * sizeof(float));

    // copy over data
    cudaMemcpy(d_s_V1x, sides_x1, num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V1y, sides_y1, num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2x, sides_x2, num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2y, sides_y2, num_sides * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_side_number , left_side_number , num_sides * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_side_number, right_side_number, num_sides * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_elem_s1, elem_s1, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elem_s2, elem_s2, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elem_s3, elem_s3, num_elem * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_V1x, V1x, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V1y, V1y, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2x, V2x, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V2y, V2y, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3x, V3x, num_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V3y, V3y, num_elem * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_elem , left_elem , num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_elem, right_elem, num_sides * sizeof(float), cudaMemcpyHostToDevice);
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

    cudaFree(d_s_r);
    
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
               int *n, int *debug, int *timesteps,
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
                if (*n < 0 || *n > 8) {
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
    } 

    // second last argument is filename
    *mesh_filename = argv[argc - 2];
    // last argument is outfilename
    *out_filename  = argv[argc - 1];

    return 0;
}

int main(int argc, char *argv[]) {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int n_threads, n_blocks_elem, n_blocks_sides;
    int i, n, n_p, t, timesteps, n_quad, n_quad1d;
    int debug;

    float dt; 
    float *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;
    float *sides_x1, *sides_x2;
    float *sides_y1, *sides_y2;

    float *r1_local, *r2_local, *w_local;

    float *s_r, *oned_w_local;

    int *left_elem, *right_elem;
    int *elem_s1, *elem_s2, *elem_s3;
    int *left_side_number, *right_side_number;

    FILE *mesh_file, *out_file;

    char line[100];
    char *mesh_filename;
    char *out_filename;

    float *Uv1, *Uv2, *Uv3;

    // get input 
    if (get_input(argc, argv, &n, &debug, &timesteps, &mesh_filename, &out_filename)) {
        return 1;
    }

    // set the order of the approximation & timestep
    n_p = (n + 1) * (n + 2) / 2;
    dt  = 0.01;

    // open the mesh to get num_elem for allocations
    mesh_file = fopen(mesh_filename, "r");
    out_file  = fopen(out_filename , "w");
    if (!mesh_file) {
        printf("\nERROR: mesh file not found.\n");
        return 1;
    }
    fgets(line, 100, mesh_file);
    sscanf(line, "%i", &num_elem);

    // allocate vertex points
    V1x = (float *) malloc(num_elem * sizeof(float));
    V1y = (float *) malloc(num_elem * sizeof(float));
    V2x = (float *) malloc(num_elem * sizeof(float));
    V2y = (float *) malloc(num_elem * sizeof(float));
    V3x = (float *) malloc(num_elem * sizeof(float));
    V3y = (float *) malloc(num_elem * sizeof(float));

    elem_s1 = (int *) malloc(num_elem * sizeof(int));
    elem_s2 = (int *) malloc(num_elem * sizeof(int));
    elem_s3 = (int *) malloc(num_elem * sizeof(int));

    // TODO: these are too big; should be a way to figure out how many we actually need
    left_side_number  = (int *)   malloc(3*num_elem * sizeof(int));
    right_side_number = (int *)   malloc(3*num_elem * sizeof(int));

    sides_x1    = (float *) malloc(3*num_elem * sizeof(float));
    sides_x2    = (float *) malloc(3*num_elem * sizeof(float));
    sides_y1    = (float *) malloc(3*num_elem * sizeof(float));
    sides_y2    = (float *) malloc(3*num_elem * sizeof(float)); 
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

    // close the file
    fclose(mesh_file);

    // initialize the gpu
    init_gpu(num_elem, num_sides, n_p,
             V1x, V1y, V2x, V2y, V3x, V3y,
             left_side_number, right_side_number,
             sides_x1, sides_y1,
             sides_x2, sides_y2, 
             elem_s1, elem_s2, elem_s3,
             left_elem, right_elem);

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
                                                  d_left_elem, d_left_side_number, num_sides); 
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

    // initial conditions
    init_conditions<<<n_blocks_elem, n_threads>>>(d_c, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y,
                    n_quad, n_p, num_elem);
    checkCudaError("error after initial conditions.");

    printf("Computing...\n");
    printf(" ? %i degree polynomial interpolation (n_p = %i)\n", n, n_p);
    printf(" ? %i precomputed basis points\n", n_quad * n_p);
    printf(" ? %i elements\n", num_elem);
    printf(" ? %i sides\n", num_sides);
    printf(" ? %i timesteps\n", timesteps);
    printf(" ? 1d quadrature rules:\n");
    for (i = 0; i < n_quad1d; i++) {
        printf("     > %f - %f \n", s_r[i], oned_w_local[i]);
    }
    printf(" ? 2d quadrature rules:\n");
    for (i = 0; i < n_quad; i++) {
        printf("     > (%f, %f) - %f \n", r1_local[i], r2_local[i], w_local[i]);
    }

    if (debug) {
        float *Nx = (float *) malloc(num_sides * sizeof(float));
        float *Ny = (float *) malloc(num_sides * sizeof(float));

        cudaMemcpy(Nx, d_Nx, num_sides *sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ny, d_Ny, num_sides *sizeof(float), cudaMemcpyDeviceToHost);

        printf(" ? normals\n");
        for (i = 0; i < num_sides; i++) {
            printf("    > (%f, %f) \n", Nx[i], Ny[i]);
        }
        free(Nx);
        free(Ny);
    }


    checkCudaError("error before time integration.");
    fprintf(out_file, "View \"Exported field \" {\n");
    for (t = 0; t < timesteps; t++) {
        // time integration
        time_integrate(dt, n_quad, n_quad1d, n_p, num_elem, num_sides, debug);
    }

    if (debug) {
        float *c = (float *) malloc(num_elem * n_p * sizeof(float));
        cudaMemcpy(c, d_c, num_elem * n_p * sizeof(float), cudaMemcpyDeviceToHost);
        printf(" c\n");
        printf(" ~~~\n");
        for (i = 0; i < num_elem * n_p; i++) {
            if (i != 0 && i % num_elem == 0) {
                printf("   --- \n");
            }
            printf(" > %f\n", c[i]);
        }
        free(c);
    }

    // evaluate at the vertex points and copy over data
    Uv1 = (float *) malloc(num_elem * sizeof(float));
    Uv2 = (float *) malloc(num_elem * sizeof(float));
    Uv3 = (float *) malloc(num_elem * sizeof(float));
    eval_u<<<n_blocks_elem, n_threads>>>(d_c, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, 
                                             d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaThreadSynchronize();
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(float), cudaMemcpyDeviceToHost);

    // write data to file
    // TODO: this will output multiple vertices values. does gmsh care? i dunno...
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%f,%f,0,%f,%f,0,%f,%f,0) {%f,%f,%f};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }

    fprintf(out_file,"};");

    // close the output file
    fclose(out_file);

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

    return 0;
}
