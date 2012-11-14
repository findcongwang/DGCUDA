/* euler_system.cu
 *
 * This file contains the relevant information for making a system to solve
 *
 * d_t [   rho   ] + d_x [     rho * u    ] + d_y [    rho * v     ] = 0
 * d_t [ rho * u ] + d_x [ rho * u^2 + p  ] + d_y [   rho * u * v  ] = 0
 * d_t [ rho * v ] + d_x [  rho * u * v   ] + d_y [  rho * v^2 + p ] = 0
 * d_t [    E    ] + d_x [ u * ( E +  p ) ] + d_y [ v * ( E +  p ) ] = 0
 *
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH 2.25

// Size of the system
int local_N = 4;

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x,y
 */
__device__ double U0(double x, double y) {
    double r = sqrt(x*x + y*y);
    return powf(1+1.0125*(1.- 1./(r * r)),2.5);
}
__device__ double U1(double x, double y) {
    double r = sqrt(x*x + y*y);
    return sin(atan(y / x)) * MACH / r * U0(x, y);
}
__device__ double U2(double x, double y) {
    double r = sqrt(x*x + y*y);
    return -cos(atan(y / x)) * MACH / r * U0(x, y);
}
__device__ double U3(double x, double y) {
    double r = sqrt(x*x + y*y);
    double p = (1.0 / GAMMA) * powf(U0(x, y), GAMMA);
    return  0.5 * U0(x,y) * (MACH*MACH/(r * r)) + p * (1./(GAMMA - 1.));
}

__device__ void evalU0(double *U, 
                       double v1x, double v1y,
                       double v2x, double v2y,
                       double v3x, double v3y,
                       int i, int n_p, int n_quad) {
    int j;
    double x, y;

    U[0] = 0.;
    U[1] = 0.;
    U[2] = 0.;
    U[3] = 0.;

    for (j = 0; j < n_quad; j++) {
         // get the actual point on the mesh
         x = r1[j] * v2x + r2[j] * v3x + (1 - r1[j] - r2[j]) * v1x;
         y = r1[j] * v2y + r2[j] * v3y + (1 - r1[j] - r2[j]) * v1y;

        // evaluate U at the integration point
        U[0] += w[j] * U0(x, y) * basis[i * n_quad + j];
        U[1] += w[j] * U1(x, y) * basis[i * n_quad + j];
        U[2] += w[j] * U2(x, y) * basis[i * n_quad + j];
        U[3] += w[j] * U3(x, y) * basis[i * n_quad + j];
    }
}

/***********************
 *
 * DEVICE FUNCTIONS
 *
 ***********************/
__device__ double pressure(double *U) {

    double rho, rhou, rhov, E;
    rho  = U[0];
    rhou = U[1];
    rhov = U[2];
    E    = U[3];

    return (GAMMA - 1.) * (E - (rhou*rhou + rhov*rhov) / 2. / rho);
}

/* evaluate c
 *
 * evaulates the speed of sound c
 */
__device__ double eval_c(double *U) {
    double p = pressure(U);
    double rho = U[0];

    return sqrtf(GAMMA * p / rho);
}    

/***********************
 *
 * FLUX
 *
 ***********************/
/* takes the actual values of rho, u, v, and E and returns the flux 
 * x and y components. 
 * NOTE: this needs the ACTUAL values for u and v, NOT rho * u, rho * v.
 */
__device__ void eval_flux(double *U, double *flux_x, double *flux_y) {

    // evaluate pressure
    double rho, rhou, rhov, E;
    double p = pressure(U);
    rho  = U[0];
    rhou = U[1];
    rhov = U[2];
    E    = U[3];

    // flux_1 
    flux_x[0] = rhou;
    flux_y[0] = rhov;

    // flux_2
    flux_x[1] = rhou * rhou / rho + p;
    flux_y[1] = rhou * rhov / rho;

    // flux_3
    flux_x[2] = rhou * rhov / rho;
    flux_y[2] = rhov * rhov / rho + p;

    // flux_4
    flux_x[3] = rhou * (E + p) / rho;
    flux_y[3] = rhov * (E + p) / rho;
}

/***********************
 *
 * RIEMAN SOLVER
 *
 ***********************/
/* finds the max absolute value of the jacobian for F(u).
 *  |u - c|, |u|, |u + c|
 */
__device__ double eval_lambda(double *U_left, double *U_right,
                              double nx,      double ny) {
                              
    double s_left, s_right;
    double c_left, c_right;
    double u_left, v_left;
    double u_right, v_right;
    double left_max, right_max;

    // get c for both sides
    c_left  = eval_c(U_left);
    c_right = eval_c(U_right);

    // find the speeds on each side
    u_left  = U_left[1] / U_left[0];
    v_left  = U_left[2] / U_left[0];
    u_right = U_right[1] / U_right[0];
    v_right = U_right[2] / U_right[0];
    s_left  = nx * u_left  + ny * v_left;
    s_right = nx * u_right + ny * v_right; 
    
    // if speed is positive, want s + c, else s - c
    if (s_left > 0.) {
        left_max = s_left + c_left;
    } else {
        left_max = -s_left + c_left;
    }

    // if speed is positive, want s + c, else s - c
    if (s_right > 0.) {
        right_max = s_right + c_right;
    } else {
        right_max = -s_right + c_right;
    }

    // return the max absolute value of | s +- c |
    if (fabs(left_max) > fabs(right_max)) {
        return fabs(left_max);
    } else { 
        return fabs(right_max);
    }
}

/***********************
 *
 * BOUNDARY CONDITIONS
 *
 ***********************/
/* Put the boundary conditions for the problem in here.
*/
__device__ void inflow_boundary(double *U_left, double *U_right,
                                double v1x, double v1y, 
                                double v2x, double v2y,
                                double v3x, double v3y,
                                double nx, double ny,
                                int j, int left_side, int n_quad1d) {

    double r1_eval, r2_eval;
    double x, y;

    // we need the mapping back to the grid space
    switch (left_side) {
        case 0: 
            r1_eval = 0.5 + 0.5 * r_oned[j];
            r2_eval = 0.;
            break;
        case 1: 
            r1_eval = (1. - r_oned[j]) / 2.;
            r2_eval = (1. + r_oned[j]) / 2.;
            break;
        case 2: 
            r1_eval = 0.;
            r2_eval = 0.5 + 0.5 * r_oned[n_quad1d - 1 - j];
            break;
    }

    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    x = v2x * r1_eval + v3x * r2_eval + v1x * (1 - r1_eval - r2_eval);
    y = v2y * r1_eval + v3y * r2_eval + v1y * (1 - r1_eval - r2_eval);
        
    U_right[0] = U0(x, y);
    U_right[1] = U1(x, y);
    U_right[2] = U2(x, y);
    U_right[3] = U3(x, y);
}

__device__ void outflow_boundary(double *U_left, double *U_right,
                                double v1x, double v1y, 
                                double v2x, double v2y,
                                double v3x, double v3y,
                                double nx, double ny,
                                int j, int left_side, int n_quad1d) {
    // just call inflow_boundary from here
    inflow_boundary(U_left, U_right, v1x, v1y, v2x, v2y, v3x, v3y,
                    nx, ny, j, left_side, n_quad1d);
}



__device__ void reflecting_boundary(double *U_left, double *U_right,
                         double v1x,      double v1y, 
                         double v2x,      double v2y,
                         double v3x,      double v3y,
                         double nx,       double ny,
                         int j, int left_side, int n_quad1d) {

    double r1_eval, r2_eval;
    double x, y;
    double Nx, Ny, dot;
    // we need the mapping back to the grid space
    switch (left_side) {
        case 0: 
            r1_eval = 0.5 + 0.5 * r_oned[j];
            r2_eval = 0.;
            break;
        case 1: 
            r1_eval = (1. + r_oned[j]) / 2.;
            r2_eval = (1. - r_oned[j]) / 2.;
            break;
        case 2: 
            r1_eval = 0.;
            r2_eval = 0.5 + 0.5 * r_oned[n_quad1d - 1 - j];
            break;
    }

    // x = x2 * r + x3 * s + x1 * (1 - r - s)
    x = v2x * r1_eval + v3x * r2_eval + v1x * (1 - r1_eval - r2_eval);
    y = v2y * r1_eval + v3y * r2_eval + v1y * (1 - r1_eval - r2_eval);

    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];


    // taken from algorithm 2 from lilia's code
    dot = sqrtf(x*x + y*y);
    Nx = x / dot;
    Ny = y / dot;

    if (Nx * nx + Ny * ny < 0) {
        Nx *= -1;
        Ny *= -1;
    }

    // set the velocities to reflect
    U_right[1] =  (U_left[1] * Ny - U_left[2] * Nx)*Ny;
    U_right[2] = -(U_left[1] * Ny - U_left[2] * Nx)*Nx;
}

/***********************
 *
 * CFL CONDITION
 *
 ***********************/
/* global lambda evaluation
 *
 * computes the max eigenvalue of |u + c|, |u|, |u - c|.
 */
__global__ void eval_global_lambda(double *C, double *lambda, 
                                   int n_quad, int n_p, int num_elem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) { 
        double c, s;

        double U[4];
        // get cell averages
        U[0] = C[num_elem * n_p * 0 + idx] * basis[0];
        U[1] = C[num_elem * n_p * 1 + idx] * basis[0];
        U[2] = C[num_elem * n_p * 2 + idx] * basis[0];
        U[3] = C[num_elem * n_p * 3 + idx] * basis[0];

        // evaluate c
        c = eval_c(U);

        // speed of the wave
        s = sqrtf(U[1]*U[1] + U[2]*U[2])/U[0];

        // return the max eigenvalue
        if (s > 0) {
            lambda[idx] = s + c;
        } else {
            lambda[idx] = -s + c;
        }
    }
}
