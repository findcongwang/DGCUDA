#include "../euler.cu"

/* supersonic.cu
 *
 * Supersonic flow around a cylinder.
 *
 */

#define PI 3.14159
#define GAMMA 1.4
#define MACH 2.25

int limiter = NO_LIMITER;  // no limiter
int time_integrator = RK4; // time integrator to use
int riemann_solver = LLF;  // local lax friedrichs riemann solver

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x,y
 */

__device__ void U0(double *U, double x, double y) {
    double r = sqrt(x*x + y*y);
    double p = (1.0 / GAMMA) * powf(U0(x, y), GAMMA);

    U[0] = powf(1+1.0125*(1.- 1./(r * r)),2.5);
    U[1] = U[0] * sin(atan(y / x)) * MACH / r;
    U[2] = U[0] * -cos(atan(y / x)) * MACH / r;
    U[3] = 0.5 * U[0] * (MACH*MACH/(r * r)) + p * (1./(GAMMA - 1.));
}

/***********************
*
* INFLOW CONDITIONS
*
************************/

__device__ void U_inflow(double *U, double x, double y, double t) {
    U0(U, x, y);
}

/***********************
*
* OUTFLOW CONDITIONS
*
************************/

__device__ void U_outflow(double *U, double x, double y, double t) {
    U0(U, x, y);
}

/***********************
*
* OUTFLOW CONDITIONS
*
************************/

__device__ void reflecting_boundary(double *U_left, double *U_right,
                         double v1x,      double v1y, 
                         double v2x,      double v2y,
                         double v3x,      double v3y,
                         double nx,       double ny,
                         int j, int left_side, int n_quad1d) {

    double r1_eval, r2_eval;
    double x, y;
    double Nx, Ny, dot;

    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];

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
 * MAIN FUNCTION
 *
 ***********************/

__device__ double get_GAMMA() {
    return GAMMA;
}

int main(int argc, char *argv[]) {
    run_dgcuda(argc, argv);
}
