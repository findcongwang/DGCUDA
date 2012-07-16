#include "2dadvec_euler.cu"

int main(int argc, char *argv[]) {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int n_threads, n_blocks_elem, n_blocks_jacobian, n_blocks_sides;
    int i, n, n_p, timesteps, n_quad, n_quad1d, alpha;
    int debug;

    float dt, t;
    float *min_J, min_j;
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

    void (*eval_u_ftn)(float*, float*, float*, float*, int, int) = NULL;
    void (*eval_error_ftn)(float*, 
                       float*, float*,
                       float*, float*,
                       float*, float*,
                       float*, float*, float*, 
                       int, int, float, int) = NULL;

    // get input 
    alpha = 0;
    if (get_input(argc, argv, &n, &debug, &timesteps, &alpha, &mesh_filename, &out_filename)) {
        return 1;
    }

    // set the order of the approximation & timestep
    n_p = (n + 1) * (n + 2) / 2;

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

    checkCudaError("error after gpu init.");
    n_threads         = 128;
    n_blocks_elem     = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    n_blocks_sides    = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);
    n_blocks_jacobian = (num_elem  / 256) + ((num_elem  % 256) ? 1 : 0);

    // pre computations
    preval_jacobian<<<n_blocks_elem, n_threads>>>(d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, num_elem); 
    cudaThreadSynchronize();

    // find the min jacobian. do it on the gpu if there are at least 256 elements
    if (num_elem >= 256) {
        min_jacobian<<<n_blocks_jacobian, 256>>>(d_J, d_min_J, num_elem);

        // each block finds the smallest value, so need to sort through n_blocks_jacobian
        min_J = (float *) malloc(n_blocks_jacobian * sizeof(float));
        cudaMemcpy(min_J, d_min_J, n_blocks_jacobian * sizeof(float), cudaMemcpyDeviceToHost);
        min_j = min_J[0];
        for (i = 0; i < n_blocks_jacobian; i++) {
            min_j = (min_J[i] > min_j) ? min_J[i] : min_j;
        }
        free(min_J);

    } else {
        // just grab all the jacobians and sort them since there are so few of them
        min_J = (float *) malloc(num_elem * sizeof(float));
        cudaMemcpy(min_J, d_J, num_elem * sizeof(float), cudaMemcpyDeviceToHost);
        min_j = min_J[0];
        for (i = 0; i < num_elem; i++) {
            min_j = (min_J[i] > min_j) ? min_J[i] : min_j;
        }
        free(min_J);
    }

    // choose dt to make this scheme stable
    dt  = min_j / 2. / (2. * n + 1.) * sqrt(2.);

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
    init_conditions<<<n_blocks_elem, n_threads>>>(d_c, d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y,
                    n_quad, n_p, num_elem, alpha);
    checkCudaError("error after initial conditions.");

    printf("Computing...\n");
    printf(" ? %i degree polynomial interpolation (n_p = %i)\n", n, n_p);
    printf(" ? %i precomputed basis points\n", n_quad * n_p);
    printf(" ? %i elements\n", num_elem);
    printf(" ? %i sides\n", num_sides);
    printf(" ? %i timesteps\n", timesteps);
    printf(" ? min jacobian = %f\n", min_j);
    printf(" ? dt = %f\n", dt);

    checkCudaError("error before time integration.");
    fprintf(out_file, "View \"Exported field \" {\n");

    time_integrate_rk4(dt, n_quad, n_quad1d, n_p, n, num_elem, num_sides, debug, alpha, timesteps);
    t = timesteps * dt;

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

    switch (n) {
        case 0: eval_u_ftn = eval_u_wrapper0;
                eval_error_ftn = eval_error_wrapper0;
                break;
        case 1: eval_u_ftn = eval_u_wrapper1;
                eval_error_ftn = eval_error_wrapper1;
                break;
        case 2: eval_u_ftn = eval_u_wrapper2;
                eval_error_ftn = eval_error_wrapper2;
                break;
        case 3: eval_u_ftn = eval_u_wrapper3;
                eval_error_ftn = eval_error_wrapper3;
                break;
        case 4: eval_u_ftn = eval_u_wrapper4;
                eval_error_ftn = eval_error_wrapper4;
                break;
        case 5: eval_u_ftn = eval_u_wrapper5;
                eval_error_ftn = eval_error_wrapper5;
                break;
        case 6: eval_u_ftn = eval_u_wrapper6;
                eval_error_ftn = eval_error_wrapper6;
                break;
        case 7: eval_u_ftn = eval_u_wrapper7;
                eval_error_ftn = eval_error_wrapper7;
                break;
    }

    //eval_error_ftn<<<n_blocks_elem, n_threads>>>(d_c, 
                                                 //d_V1x, d_V1y,
                                                 //d_V2x, d_V2y,
                                                 //d_V3x, d_V3y,
                                                 //d_Uv1, d_Uv2, d_Uv3, num_elem, n_p, t * dt);
    eval_u_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
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
