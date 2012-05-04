#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "2dadvec_kernels.cu"
/* 2dadvec.cu
 * 
 * This file calls the kernels in 2dadvec_kernels.cu for the 2D advection
 * DG method.
 */

void set_quadrature(int p,
                    float *r1, float *r2, float *w,
                    float *s1_r1, float *s1_r2,
                    float *s2_r1, float *s2_r2,
                    float *s3_r1, float *s3_r2,
                    float *oned_r, float *oned_w) {
    switch (p) {
        case 0:
            // set 2d integration points and weights
            r1[0] = 0.333333333333333;
            r2[0] = 0.333333333333333;
            w [0] = 1.0;

            // set 1d integration points and weights
            s1_r1[0] = 0.5;
            s1_r2[0] = 0.0;
            
            s2_r1[0] = 0.25;
            s2_r1[0] = 0.25;

            s3_r1[0] = 0.0;
            s3_r2[0] = 0.5;

            oned_w[0] = 1.0;

            break;
        case 2:
            r1[0] = 0.166666666666666;
            r2[0] = 0.166666666666666;
            w[0]  = 0.333333333333333;
            r1[1] = 0.666666666666666;
            r2[1] = 0.166666666666666;
            w[1]  = 0.333333333333333;
            r1[2] = 0.166666666666666;
            r2[2] = 0.666666666666666;
            w[2]  = 0.333333333333333;
            break;
    }
}
/*             
  { {0.333333333333333,0.3333333333333333},-0.5625},
  { {0.6,0.2},.520833333333333 },
  { {0.2,0.6},.520833333333333 },
  { {0.2,0.2},.520833333333333 }
};

IntPt2d GQT4[6] = {
  { {0.816847572980459,0.091576213509771},0.109951743655322},
  { {0.091576213509771,0.816847572980459},0.109951743655322},
  { {0.091576213509771,0.091576213509771},0.109951743655322},
  { {0.108103018168070,0.445948490915965},0.223381589678011},
  { {0.445948490915965,0.108103018168070},0.223381589678011},
  { {0.445948490915965,0.445948490915965},0.223381589678011}
};

IntPt2d GQT5[7] = {
  { {0.333333333333333,0.333333333333333},0.225000000000000},
  { {0.797426985353087,0.101286507323456},0.125939180544827},
  { {0.101286507323456,0.797426985353087},0.125939180544827},
  { {0.101286507323456,0.101286507323456},0.125939180544827},
  { {0.470142064105115,0.059715871789770},0.132394152788506},
  { {0.059715871789770,0.470142064105115},0.132394152788506},
  { {0.470142064105115,0.470142064105115},0.132394152788506}
};

IntPt2d GQT6[12] = {
  { {0.873821971016996,0.063089014491502},0.050844906370207},
  { {0.063089014491502,0.873821971016996},0.050844906370207},
  { {0.063089014491502,0.063089014491502},0.050844906370207},
  { {0.501426509658179,0.249286745170910},0.116786275726379},
  { {0.249286745170910,0.501426509658179},0.116786275726379},
  { {0.249286745170910,0.249286745170910},0.116786275726379},
  { {0.636502499121399,0.310352451033784},0.082851075618374},
  { {0.310352451033784,0.636502499121399},0.082851075618374},
  { {0.636502499121399,0.053145049844816},0.082851075618374},
  { {0.310352451033784,0.053145049844816},0.082851075618374},
  { {0.053145049844816,0.310352451033785},0.082851075618374},
  { {0.053145049844816,0.636502499121399},0.082851075618374}
};

IntPt2d GQT7[13] = {
  { {0.333333333333333,0.333333333333333},-0.149570044467682},
  { {0.479308067841920,0.260345966079040},0.175615257433208},
  { {0.260345966079040,0.479308067841920},0.175615257433208},
  { {0.260345966079040,0.260345966079040},0.175615257433208},
  { {0.869739794195568,0.065130102902216},0.053347235608838},
  { {0.065130102902216,0.869739794195568},0.053347235608838},
  { {0.065130102902216,0.065130102902216},0.053347235608838},
  { {0.048690315425316,0.312865496004874},0.077113760890257},
  { {0.312865496004874,0.048690315425316},0.077113760890257},
  { {0.638444188569810,0.048690315425316},0.077113760890257},
  { {0.048690315425316,0.638444188569810},0.077113760890257},
  { {0.312865496004874,0.638444188569810},0.077113760890257},
  { {0.638444188569810,0.312865496004874},0.077113760890257}

};

IntPt2d GQT8[16] = {
  { {0.333333333333333,0.333333333333333},0.144315607677787},
  { {0.081414823414554,0.459292588292723},0.095091634267285},
  { {0.459292588292723,0.081414823414554},0.095091634267285},
  { {0.459292588292723,0.459292588292723},0.095091634267285},
  { {0.658861384496480,0.170569307751760},0.103217370534718},
  { {0.170569307751760,0.658861384496480},0.103217370534718},
  { {0.170569307751760,0.170569307751760},0.103217370534718},
  { {0.898905543365938,0.050547228317031},0.032458497623198},
  { {0.050547228317031,0.898905543365938},0.032458497623198},
  { {0.050547228317031,0.050547228317031},0.032458497623198},  
  { {0.008394777409958,0.728492392955404},0.027230314174435},
  { {0.728492392955404,0.008394777409958},0.027230314174435},
  { {0.263112829634638,0.008394777409958},0.027230314174435},
  { {0.008394777409958,0.263112829634638},0.027230314174435},
  { {0.263112829634638,0.728492392955404},0.027230314174435},
  { {0.728492392955404,0.263112829634638},0.027230314174435}
};

IntPt2d GQT9[19] = {
  { {0.333333333333333,0.333333333333333},0.097135796282799},
  { {0.020634961602525,0.489682519198738},0.031334700227139},
  { {0.489682519198738,0.020634961602525},0.031334700227139},
  { {0.489682519198738,0.489682519198738},0.031334700227139},
  { {0.125820817014127,0.437089591492937},0.077827541004774},
  { {0.437089591492937,0.125820817014127},0.077827541004774},
  { {0.437089591492937,0.437089591492937},0.077827541004774},
  { {0.623592928761935,0.188203535619033},0.079647738927210},
  { {0.188203535619033,0.623592928761935},0.079647738927210},
  { {0.188203535619033,0.188203535619033},0.079647738927210},
  { {0.910540973211095,0.044729513394453},0.025577675658698},
  { {0.044729513394453,0.910540973211095},0.025577675658698},
  { {0.044729513394453,0.044729513394453},0.025577675658698},
  { {0.036838412054736,0.221962989160766},0.043283539377289},
  { {0.221962989160766,0.036838412054736},0.043283539377289},
  { {0.036838412054736,0.741198598784498},0.043283539377289},
  { {0.741198598784498,0.036838412054736},0.043283539377289},
  { {0.741198598784498,0.221962989160766},0.043283539377289},
  { {0.221962989160766,0.741198598784498},0.043283539377289}
};

IntPt2d GQT10[25] = {
  { {0.333333333333333,0.333333333333333},0.090817990382754},
  { {0.028844733232685,0.485577633383657},0.036725957756467},
  { {0.485577633383657,0.028844733232685},0.036725957756467},
  { {0.485577633383657,0.485577633383657},0.036725957756467},
  { {0.781036849029926,0.109481575485037},0.045321059435528},
  { {0.109481575485037,0.781036849029926},0.045321059435528},
  { {0.109481575485037,0.109481575485037},0.045321059435528},
  { {0.141707219414880,0.307939838764121},0.072757916845420},
  { {0.307939838764121,0.141707219414880},0.072757916845420},
  { {0.307939838764121,0.550352941820999},0.072757916845420},
  { {0.550352941820999,0.307939838764121},0.072757916845420},
  { {0.550352941820999,0.141707219414880},0.072757916845420},
  { {0.141707219414880,0.550352941820999},0.072757916845420},
  { {0.025003534762686,0.246672560639903},0.028327242531057},
  { {0.246672560639903,0.025003534762686},0.028327242531057},
  { {0.025003534762686,0.728323904597411},0.028327242531057},
  { {0.728323904597411,0.025003534762686},0.028327242531057},
  { {0.728323904597411,0.246672560639903},0.028327242531057},
  { {0.246672560639903,0.728323904597411},0.028327242531057},
  { {0.009540815400299,0.066803251012200},0.009421666963733},
  { {0.066803251012200,0.009540815400299},0.009421666963733},
  { {0.066803251012200,0.923655933587500},0.009421666963733},
  { {0.923655933587500,0.066803251012200},0.009421666963733},
  { {0.923655933587500,0.009540815400299},0.009421666963733},
  { {0.009540815400299,0.923655933587500},0.009421666963733}
};
*/
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

    int i, j, s1, s2, s3, numsides, which_elem;
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
                right_side_number[j] = 1;
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
                right_side_number[j] = 2;
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
                right_side_number[j] = 3;
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
            left_side_number[numsides] = 1;
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
            left_side_number[numsides] = 2;
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
            left_side_number[numsides] = 3;
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

void time_integrate(float *c, float dt, int n_p, int num_elem, int num_sides) {
    int n_threads = 128;

    int n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);

    int num_rhs = (n_p + 1) * num_elem;

    float *rhs = (float *) malloc(num_elem * (n_p + 1) * sizeof(float));

    // stage 1
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_c, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s1_r1, d_s1_r2,
                     d_s2_r1, d_s2_r2,
                     d_s3_r1, d_s3_r2,
                     d_oned_r, d_oned_w, 
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 1: eval_riemann");

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_c, d_quad_rhs, d_r1, d_r2, d_w, d_J, n_p, num_elem);
    cudaThreadSynchronize();

    /////////////////////
    // GOOD: the quadrature is giving us zeros for n_p = 0;
    //float *rhs = (float *) malloc(num_elem * (n_p + 1) * sizeof(float));
    //cudaMemcpy(rhs, d_rhs, num_elem * (n_p + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < num_elem * (n_p + 1); i++) {
    //    printf(" > %f \n", rhs[i]);
    //}
    /////////////////////

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k1, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, dt, num_elem);
    cudaThreadSynchronize();
    cudaMemcpy(rhs, d_k1, num_elem * (n_p + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_elem * (n_p + 1); i++) {
        printf(" > %f \n", rhs[i]);
    }

    rk4_tempstorage<<<n_blocks_elem, n_threads>>>(d_c, d_kstar, d_k1, 0.5, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 1.");

    // stage 2
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s1_r1, d_s1_r2,
                     d_s2_r1, d_s2_r2,
                     d_s3_r1, d_s3_r2,
                     d_oned_r, d_oned_w, 
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_kstar, d_quad_rhs, d_r1, d_r2, d_w, d_J, n_p, num_elem);
    cudaThreadSynchronize();

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k2, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs,
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, dt, num_elem);
    cudaThreadSynchronize();

    rk4_tempstorage<<<n_blocks_elem, n_threads>>>(d_c, d_kstar, d_k2, 0.5, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 2.");

    // stage 3
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s1_r1, d_s1_r2,
                     d_s2_r1, d_s2_r2,
                     d_s3_r1, d_s3_r2,
                     d_oned_r, d_oned_w, 
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_kstar, d_quad_rhs, d_r1, d_r2, d_w, d_J, n_p, num_elem);
    cudaThreadSynchronize();

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k3, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, dt, num_elem);
    cudaThreadSynchronize();

    rk4_tempstorage<<<n_blocks_elem, n_threads>>>(d_c, d_kstar, d_k3, 1.0, n_p, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 3.");

    // stage 4
    eval_riemann<<<n_blocks_sides, n_threads>>>
                    (d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, d_J, 
                     d_s_length,
                     d_s1_r1, d_s1_r2,
                     d_s2_r1, d_s2_r2,
                     d_s3_r1, d_s3_r2,
                     d_oned_r, d_oned_w, 
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, n_p, num_sides, num_elem);
    cudaThreadSynchronize();

    eval_quad<<<n_blocks_elem, n_threads>>>
                    (d_kstar, d_quad_rhs, d_r1, d_r2, d_w, d_J, n_p, num_elem);
    cudaThreadSynchronize();

    eval_rhs<<<n_blocks_elem, n_threads>>>(d_k4, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                                          d_elem_s1, d_elem_s2, d_elem_s3, 
                                          d_left_elem, dt, num_elem);
    cudaThreadSynchronize();

    checkCudaError("error after stage 4.");
    
    // final stage
    rk4<<<n_blocks_elem, n_threads>>>(d_c, d_k1, d_k2, d_k3, d_k4, n_p, num_elem);
    cudaThreadSynchronize();

    cudaMemcpy(c, d_c, num_elem * (n_p + 1) * sizeof(float), cudaMemcpyDeviceToHost);

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
    cudaMalloc((void **) &d_c  , num_elem * (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_quad_rhs, num_elem * sizeof(float));
    cudaMalloc((void **) &d_left_riemann_rhs, num_sides * sizeof(float));
    cudaMalloc((void **) &d_right_riemann_rhs, num_sides * sizeof(float));

    cudaMalloc((void **) &d_kstar, num_elem * (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_k1, num_elem * (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_k2, num_elem * (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_k3, num_elem * (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_k4, num_elem * (n_p + 1) * sizeof(float));

    cudaMalloc((void **) &d_r1, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_r2, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_w , (n_p + 1) * sizeof(float));

    cudaMalloc((void **) &d_oned_r, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_oned_w, (n_p + 1) * sizeof(float));

    cudaMalloc((void **) &d_J, num_elem * sizeof(float));
    cudaMalloc((void **) &d_s_length, num_sides * sizeof(float));

    cudaMalloc((void **) &d_s_V1x, num_sides * sizeof(float));
    cudaMalloc((void **) &d_s_V2x, num_sides * sizeof(float));
    cudaMalloc((void **) &d_s_V1y, num_sides * sizeof(float));
    cudaMalloc((void **) &d_s_V2y, num_sides * sizeof(float));

    cudaMalloc((void **) &d_elem_s1, num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s2, num_elem * sizeof(int));
    cudaMalloc((void **) &d_elem_s3, num_elem * sizeof(int));

    cudaMalloc((void **) &d_V1x, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V1y, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V2x, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V2y, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V3x, num_elem * sizeof(float));
    cudaMalloc((void **) &d_V3y, num_elem * sizeof(float));

    cudaMalloc((void **) &d_s1_r1, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_s1_r2, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_s2_r1, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_s2_r2, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_s3_r1, (n_p + 1) * sizeof(float));
    cudaMalloc((void **) &d_s3_r2, (n_p + 1) * sizeof(float));
    
    cudaMalloc((void **) &d_left_side_number , num_sides * sizeof(int));
    cudaMalloc((void **) &d_right_side_number, num_sides * sizeof(int));

    cudaMalloc((void **) &d_Nx, num_sides * sizeof(float));
    cudaMalloc((void **) &d_Ny, num_sides * sizeof(float));

    cudaMalloc((void **) &d_right_elem, num_sides * sizeof(int));
    cudaMalloc((void **) &d_left_elem , num_sides * sizeof(int));

    // set quad_rhs to 0
    //cudaMemset(d_quad_rhs, 0., num_elem * sizeof(float));

    // copy over data
    cudaMemcpy(d_s_V1x, sides_x1, num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V1y, sides_y1, num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2x, sides_x2, num_sides * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s_V2y, sides_y2, num_sides * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_left_side_number , left_side_number , num_elem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_side_number, right_side_number, num_elem * sizeof(int), cudaMemcpyHostToDevice);

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

int main() {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int i, n_p;

    float *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;

    float *sides_x1, *sides_x2;
    float *sides_y1, *sides_y2;

    int *left_elem, *right_elem;
    int *elem_s1, *elem_s2, *elem_s3;
    int *left_side_number, *right_side_number;

    n_p = 0;

    FILE *mesh_file;
    mesh_file = fopen("canonical.out", "r");

    // first line should be the number of elements
    char line[100];
    fgets(line, 100, mesh_file);
    sscanf(line, "%i", &num_elem);

    // allocate integration points
    float *r1 = (float *) malloc(1 * sizeof(float));
    float *r2 = (float *) malloc(1 * sizeof(float));
    float *w =  (float *) malloc(1 * sizeof(float));
    float *s1_r1 = (float *) malloc(1 * sizeof(float));
    float *s1_r2 = (float *) malloc(1 * sizeof(float));
    float *s2_r1 = (float *) malloc(1 * sizeof(float));
    float *s2_r2 = (float *) malloc(1 * sizeof(float));
    float *s3_r1 = (float *) malloc(1 * sizeof(float));
    float *s3_r2 = (float *) malloc(1 * sizeof(float));

    float *oned_r = (float *) malloc(1 * sizeof(float));
    float *oned_w = (float *) malloc(1 * sizeof(float));


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

    for (i = 0; i < num_sides; i++) {
        printf(" ? (%i, %i) \n", left_side_number[i], right_side_number[i]);
    }
    //for (i = 0; i < num_elem; i++) {
        //printf(" ? (%i, %i, %i) \n", elem_s1[i], elem_s2[i], elem_s3[i]);
    //}
    
    int n_threads        = 128;
    int n_blocks_elem    = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    int n_blocks_sides   = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);

    // pre computations
    preval_side_length<<<n_blocks_sides, n_threads>>>(d_s_length, d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y, 
                                                      num_sides); 
    preval_jacobian<<<n_blocks_elem, n_threads>>>(d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, num_elem); 
    preval_normals<<<n_blocks_sides, n_threads>>>(d_Nx, d_Ny, d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y,
                                                  d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, 
                                                  d_left_elem, num_sides); 
    checkCudaError("error after prevals.");

    float *Nx = (float *) malloc(num_sides * sizeof(float));
    float *Ny = (float *) malloc(num_sides * sizeof(float)); 

    cudaMemcpy(Nx, d_Nx, num_sides * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ny, d_Ny, num_sides * sizeof(float), cudaMemcpyDeviceToHost);

    for (i = 0; i < num_sides; i++) {
        printf("normals = (%f, %f)\n", Nx[i], Ny[i]);
    }

    float *side_len = (float *)malloc(num_sides * sizeof(float));
    //float *J = (float *)malloc(num_elem * sizeof(float));

    cudaMemcpy(side_len, d_s_length, num_sides * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(J, d_J, num_elem * sizeof(float), cudaMemcpyDeviceToHost);

    //for (i = 0; i < num_sides; i++) {
        //printf("len = %f \n", side_len[i]);
    //}

    //float sum = 0;
    //for (i = 0; i < num_elem; i++) {
        //printf("J %i = %f\n", i, J[i]);
        //sum += J[i];
    //}

    //printf("total area = %f \n", sum);

    //free(side_len);
    //free(J);

    float dt = 0.001;
    int t;

    // get the correct quadrature rules for this scheme
    set_quadrature(n_p, r1, r2, w, 
                   s1_r1, s1_r2, 
                   s2_r1, s2_r2, 
                   s3_r1, s3_r2, 
                   oned_r, oned_w);

    // initial conditions
    printf("num_elem = %i\n", num_elem);
    init_conditions<<<n_blocks_elem, n_threads>>>(d_c, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y,
                    d_r1, d_r2, d_w, n_p, num_elem);
    checkCudaError("error after initial conditions.");

    // no longer need vertices stored on the GPU
    cudaFree(d_V1x);
    cudaFree(d_V1y);
    cudaFree(d_V2x);
    cudaFree(d_V2y);
    cudaFree(d_V3x);
    cudaFree(d_V3y);
    cudaFree(d_s_V1x);
    cudaFree(d_s_V1y);
    cudaFree(d_s_V2x);
    cudaFree(d_s_V2y);

    checkCudaError("error before quadrature copy.");

    cudaMemcpy(d_r1, r1, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r2, r2, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w , w , 1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_s1_r1, s1_r1, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s1_r2, s1_r2, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s2_r1, s2_r1, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s2_r2, s2_r2, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s3_r1, s3_r1, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s3_r2, s3_r2, 1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_oned_r, oned_r, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oned_w, oned_w, 1 * sizeof(float), cudaMemcpyHostToDevice);

    checkCudaError("error before time integration.");

    float *c = (float *) malloc(num_elem * (n_p + 1) * sizeof(float));
    cudaMemcpy(c, d_c, num_elem * (n_p + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    printf("---------\n", c[i]);
    for (i = 0; i < num_elem; i++) {
        printf("%f \n", c[i]);
    }

    // time integration
    for (t = 0; t < 1; t++) {
        time_integrate(c, dt, n_p, num_elem, num_sides);
        printf("---------\n", c[i]);
        for (i = 0; i < num_elem; i++) {
            printf("%f \n", c[i]);
        }
    }

    // free up memory
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

    return 0;
}
