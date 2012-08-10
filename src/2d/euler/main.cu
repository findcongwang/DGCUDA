#include "euler.cu"

int main(int argc, char *argv[]) {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int n_threads, n_blocks_elem, n_blocks_reduction, n_blocks_sides;
    int i, n, n_p, timesteps, n_quad, n_quad1d;

    double dt, t, endtime;
    double *min_radius;
    double min_r;
    double *V1x, *V1y, *V2x, *V2y, *V3x, *V3y;
    double *sides_x1, *sides_x2;
    double *sides_y1, *sides_y2;

    double *r1_local, *r2_local, *w_local;

    double *s_r, *oned_w_local;

    int *left_elem, *right_elem;
    int *elem_s1, *elem_s2, *elem_s3;
    int *left_side_number, *right_side_number;

    FILE *mesh_file, *out_file;

    char line[100];
    char *mesh_filename;
    char *out_filename;
    char *rho_out_filename;
    char *u_out_filename;
    char *v_out_filename;
    char *E_out_filename;
    char *outfile_base;
    int outfile_len;

    double *Uv1, *Uv2, *Uv3;

    void (*eval_rho_ftn)(double*, double*, double*, double*, int, int) = NULL;
    void (*eval_u_ftn)(double*, double*, double*, double*, int, int) = NULL;
    void (*eval_v_ftn)(double*, double*, double*, double*, int, int) = NULL;
    void (*eval_E_ftn)(double*, double*, double*, double*, int, int) = NULL;
    void (*eval_error_ftn)(double*, 
                       double*, double*,
                       double*, double*,
                       double*, double*,
                       double*, double*, double*, 
                       int, int, double, int) = NULL;

    // get input 
    endtime = -1;
    if (get_input(argc, argv, &n, &timesteps, &endtime, &mesh_filename, &out_filename)) {
        return 1;
    }

    // TODO: this should be cleaner, obviously
    rho_out_filename = "output/uniform_rho.out";
    u_out_filename = "output/uniform_u.out";
    v_out_filename = "output/uniform_v.out";
    E_out_filename = "output/uniform_E.out";

    // set the order of the approximation & timestep
    n_p = (n + 1) * (n + 2) / 2;

    // open the mesh to get num_elem for allocations
    mesh_file = fopen(mesh_filename, "r");
    if (!mesh_file) {
        printf("\nERROR: mesh file not found.\n");
        return 1;
    }
    fgets(line, 100, mesh_file);
    sscanf(line, "%i", &num_elem);

    // get the correct functions for this scheme
    switch (n) {
        case 0: eval_rho_ftn = eval_rho_wrapper0;
                eval_u_ftn = eval_u_wrapper0;
                eval_v_ftn = eval_v_wrapper0;
                eval_E_ftn = eval_E_wrapper0;
                break;
        case 1: eval_rho_ftn = eval_rho_wrapper1;
                eval_u_ftn = eval_u_wrapper1;
                eval_v_ftn = eval_v_wrapper1;
                eval_E_ftn = eval_E_wrapper1;
                break;
        case 2: eval_rho_ftn = eval_rho_wrapper2;
                eval_u_ftn = eval_u_wrapper2;
                eval_v_ftn = eval_v_wrapper2;
                eval_E_ftn = eval_E_wrapper2;
                break;
        case 3: eval_rho_ftn = eval_rho_wrapper3;
                eval_u_ftn = eval_u_wrapper3;
                eval_v_ftn = eval_v_wrapper3;
                eval_E_ftn = eval_E_wrapper3;
                break;
        case 4: eval_rho_ftn = eval_rho_wrapper4;
                eval_u_ftn = eval_u_wrapper4;
                eval_v_ftn = eval_v_wrapper4;
                eval_E_ftn = eval_E_wrapper4;
                break;
        case 5: eval_rho_ftn = eval_rho_wrapper5;
                eval_u_ftn = eval_u_wrapper5;
                eval_v_ftn = eval_v_wrapper5;
                eval_E_ftn = eval_E_wrapper5;
                break;
    }


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
    n_threads          = 256;
    n_blocks_elem      = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    n_blocks_sides     = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);
    n_blocks_reduction = (num_elem  / 256) + ((num_elem  % 256) ? 1 : 0);

    // find the min inscribed circle
    preval_inscribed_circles<<<n_blocks_elem, n_threads>>>
                (d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, num_elem);
    min_radius = (double *) malloc(num_elem * sizeof(double));

    /*
    // find the min inscribed circle. do it on the gpu if there are at least 256 elements
    if (num_elem >= 256) {
        //min_reduction<<<n_blocks_reduction, 256>>>(d_J, d_reduction, num_elem);
        cudaThreadSynchronize();
        checkCudaError("error after min_jacobian.");

        // each block finds the smallest value, so need to sort through n_blocks_reduction
        min_radius = (double *) malloc(n_blocks_reduction * sizeof(double));
        cudaMemcpy(min_radius, d_reduction, n_blocks_reduction * sizeof(double), cudaMemcpyDeviceToHost);
        min_r = min_radius[0];
        for (i = 1; i < n_blocks_reduction; i++) {
            min_r = (min_radius[i] < min_r) ? min_radius[i] : min_r;
        }
        free(min_radius);

    } else {
        */
        // just grab all the radii and sort them since there are so few of them
        min_radius = (double *) malloc(num_elem * sizeof(double));
        cudaMemcpy(min_radius, d_J, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        min_r = min_radius[0];
        for (i = 1; i < num_elem; i++) {
            min_r = (min_radius[i] < min_r) ? min_radius[i] : min_r;
        }
        free(min_radius);
    //}

    // pre computations
    preval_jacobian<<<n_blocks_elem, n_threads>>>(d_J, d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y, num_elem); 
    checkCudaError("error after preval_jacobian.");

    cudaThreadSynchronize();

    preval_side_length<<<n_blocks_sides, n_threads>>>(d_s_length, d_s_V1x, d_s_V1y, d_s_V2x, d_s_V2y, 
                                                      num_sides); 
    //cudaThreadSynchronize();
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
                    n_quad, n_p, num_elem);
    checkCudaError("error after initial conditions.");

    printf("Computing...\n");
    printf(" ? %i degree polynomial interpolation (n_p = %i)\n", n, n_p);
    printf(" ? %i precomputed basis points\n", n_quad * n_p);
    printf(" ? %i elements\n", num_elem);
    printf(" ? %i sides\n", num_sides);
    printf(" ? min radius = %lf\n", min_r);
    printf(" ? endtime = %lf\n", endtime);

    checkCudaError("error before time integration.");

    time_integrate_rk4(n_quad, n_quad1d, n_p, n, num_elem, num_sides, endtime, min_r);

    // evaluate at the vertex points and copy over data
    Uv1 = (double *) malloc(num_elem * sizeof(double));
    Uv2 = (double *) malloc(num_elem * sizeof(double));
    Uv3 = (double *) malloc(num_elem * sizeof(double));

    // evaluate rho and write to file 
    eval_rho_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    out_file  = fopen(rho_out_filename , "w");
    fprintf(out_file, "View \"Density \" {\n");
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%lf,%lf,0,%lf,%lf,0,%lf,%lf,0) {%lf,%lf,%lf};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }
    fprintf(out_file,"};");
    fclose(out_file);

    // evaluate u and write to file
    eval_u_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    out_file  = fopen(u_out_filename , "w");
    fprintf(out_file, "View \"u \" {\n");
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%lf,%lf,0,%lf,%lf,0,%lf,%lf,0) {%lf,%lf,%lf};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }
    fprintf(out_file,"};");
    fclose(out_file);

    // evaluate v and write to file
    eval_v_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    out_file  = fopen(v_out_filename , "w");
    fprintf(out_file, "View \"v \" {\n");
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%lf,%lf,0,%lf,%lf,0,%lf,%lf,0) {%lf,%lf,%lf};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }
    fprintf(out_file,"};");
    fclose(out_file);

    // evaluate E and write to file
    eval_E_ftn<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p);
    cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    out_file  = fopen(E_out_filename , "w");
    fprintf(out_file, "View \"E \" {\n");
    for (i = 0; i < num_elem; i++) {
        fprintf(out_file, "ST (%lf,%lf,0,%lf,%lf,0,%lf,%lf,0) {%lf,%lf,%lf};\n", 
                               V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               Uv1[i], Uv2[i], Uv3[i]);
    }
    fprintf(out_file,"};");
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
