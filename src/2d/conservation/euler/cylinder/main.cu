#include <stdlib.h>
#include "../conserv.cu"
#include "euler.cu"

extern int local_N;

int main(int argc, char *argv[]) {
    checkCudaError("error before start.");
    int num_elem, num_sides;
    int n_threads, n_blocks_elem, n_blocks_sides;
    int i, n, n_p, timesteps, n_quad, n_quad1d;

    double endtime;
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

    char out_filename[100];
    char *mesh_filename;

    double *Uu1, *Uu2, *Uu3;
    double *Uv1, *Uv2, *Uv3;
    double *error;

    //void (*eval_rho_ftn)(double*, double*, double*, double*, int, int) = NULL;
    //void (*eval_u_ftn)(double*, double*, double*, double*, int, int) = NULL;
    //void (*eval_v_ftn)(double*, double*, double*, double*, int, int) = NULL;
    //void (*eval_E_ftn)(double*, double*, double*, double*, int, int) = NULL;
    //void (*eval_error_ftn)(double*, 
                       //double*, double*,
                       //double*, double*,
                       //double*, double*,
                       //double*, double*, double*, 
                       //int, int, double, int) = NULL;

    // get input 
    endtime = -1;
    if (get_input(argc, argv, &n, &timesteps, &endtime, &mesh_filename)) {
        return 1;
    }

    // set the order of the approximation & timestep
    n_p = (n + 1) * (n + 2) / 2;

    // open the mesh to get num_elem for allocations
    mesh_file = fopen(mesh_filename, "r");
    if (!mesh_file) {
        printf("\nERROR: mesh file not found.\n");
        return 1;
    }

    // read in the mesh and make all the mappings
    read_mesh(mesh_file, &num_sides, &num_elem,
                         &V1x, &V1y, &V2x, &V2y, &V3x, &V3y,
                         &left_side_number, &right_side_number,
                         &sides_x1, &sides_y1, 
                         &sides_x2, &sides_y2, 
                         &elem_s1, &elem_s2, &elem_s3,
                         &left_elem, &right_elem);

                
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

    // let the GPU know the size of the system
    set_N(local_N);

    checkCudaError("error after gpu init.");
    n_threads          = 256;
    n_blocks_elem      = (num_elem  / n_threads) + ((num_elem  % n_threads) ? 1 : 0);
    n_blocks_sides     = (num_sides / n_threads) + ((num_sides % n_threads) ? 1 : 0);
    //n_blocks_reduction = (num_elem  / 256) + ((num_elem  % 256) ? 1 : 0);

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

    /*
    double *C = (double *) malloc(4 * num_elem * n_p * sizeof(double));
    cudaMemcpy(C, d_c, 4*num_elem*n_p *sizeof(double), cudaMemcpyDeviceToHost);
    for (i = 0; i < 4*num_elem*n_p; i++) {
        printf("%lf\n", C[i]);
    }
    free(C);
    */

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
    Uu1 = (double *) malloc(num_elem * sizeof(double));
    Uu2 = (double *) malloc(num_elem * sizeof(double));
    Uu3 = (double *) malloc(num_elem * sizeof(double));

    Uv1 = (double *) malloc(num_elem * sizeof(double));
    Uv2 = (double *) malloc(num_elem * sizeof(double));
    Uv3 = (double *) malloc(num_elem * sizeof(double));

    // evaluate and write to file
    for (n = 0; n < local_N; n++) {
        eval_u<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, num_elem, n_p, n);
        cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
        sprintf(out_filename, "output/U%d.msh", n);
        out_file  = fopen(out_filename , "w");
        fprintf(out_file, "View \"U%i \" {\n", n);
        for (i = 0; i < num_elem; i++) {
            fprintf(out_file, "ST (%lf,%lf,0,%lf,%lf,0,%lf,%lf,0) {%lf,%lf,%lf};\n", 
                                   V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                                   Uv1[i], Uv2[i], Uv3[i]);
        }
        fprintf(out_file,"};");
        fclose(out_file);
    }

    // evaluate the u and v vectors and write to file
    //measure_error<<<n_blocks_elem, n_threads>>>(d_c, d_Uv1, d_Uv2, d_Uv3, 
                  //d_V1x, d_V1y, d_V2x, d_V2y, d_V3x, d_V3y,
                  //num_elem, n_p);

    //cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Uv2, d_Uv2, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(Uv3, d_Uv3, num_elem * sizeof(double), cudaMemcpyDeviceToHost);
    //out_file  = fopen("output/p_error.out" , "w");
    //fprintf(out_file, "View \"E \" {\n");
    //for (i = 0; i < num_elem; i++) {
        //fprintf(out_file, "ST (%lf,%lf,0,%lf,%lf,0,%lf,%lf,0) {%lf,%lf,%lf};\n", 
                               //V1x[i], V1y[i], V2x[i], V2y[i], V3x[i], V3y[i],
                               //Uv1[i], Uv2[i], Uv3[i]);
    //}
    //fprintf(out_file,"};");
    //fclose(out_file);

    error = (double *) malloc(num_elem * sizeof(double));
    eval_error<<<n_blocks_elem, n_threads>>>(d_c, d_error, 
                                             d_V1x, d_V1y,
                                             d_V2x, d_V2y,
                                             d_V3x, d_V3y,
                                             num_elem, n_p,n_quad, 0);
    cudaMemcpy(error, d_error, num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    //cudaMemcpy(Uv1, d_Uv1, num_elem * sizeof(double), cudaMemcpyDeviceToHost);

    double total_error = 0.;
    for (i = 0; i < num_elem; i++) {
        total_error += error[i];
    }
    total_error = sqrtf(total_error);
    printf("error for rho = %.015lf\n", total_error);

    // free variables
    free_gpu();
    
    free(Uu1);
    free(Uu2);
    free(Uu3);
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
