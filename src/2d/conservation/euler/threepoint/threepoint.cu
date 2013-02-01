#include "../euler.cu"

/* threepoint.cu
 *
 * Use the Euler equations to solve the three point paradox.
 *
 * Create a weak shock and run it into a thin wedge with angle THETA at mach MACH.
 * Use p = 0 to use weak shocks and don't use a limiter.
 * 
 */

#define GAMMA 1.4
#define PI 3.14159
#define MACH 0.25
#define THETA 5

/***********************
 *
 * SHOCK CONDITIONS
 *
 ***********************/

__device__ double U_shock(double *U, double x, double y) {
    U[0] = GAMMA;
    U[1] = MACH * cos(THETA * PI / 180.) * U0_shock(x, y);
    U[2] = MACH * sin(THETA * PI / 180.) * U0_shock(x, y);
    U[3] = 0.5 * U0(x ,y) * MACH * MACH + 1./ (GAMMA - 1.0);
}

/***********************
 *
 * INITIAL CONDITIONS
 *
 ***********************/

/* initial condition function
 *
 * returns the value of the intial condition at point x,y
 */
__device__ double U0(double *U, double x, double y) {
    double angle = atan(x / y);
    if (angle < THETA * PI / 180) {
        U_shock(U, x, y);
    } else {
        U[0] = 0.5 * GAMMA;
        U[1] = 0.;
        U[2] = 0.;
        U[3] = 0.5 * U0(x ,y) * MACH * MACH + 1./ (GAMMA - 1.0);
    }
}

/***********************
*
* INFLOW CONDITIONS
*
************************/

__device__ void U_inflow(double *U, double x, double y, double t) {
    if (x < MACH * t * cos(THETA * PI / 180.) + sin(THETA * PI / 180.)) {
        U_shock(U, x, y);
    } else {
        U0(U, x, y);
    }
}

/***********************
*
* OUTFLOW CONDITIONS
*
************************/

__device__ double U_outflow(double *U, double x, double y, double t) {
    U_shock(U, x, y);
}

/***********************
*
* REFLECTING CONDITIONS
*
************************/

__device__ void reflecting_boundary(double *U_left, double *U_right,
                         double v1x,      double v1y, 
                         double v2x,      double v2y,
                         double v3x,      double v3y,
                         double nx,       double ny,
                         int j, int left_side, int n_quad1d) {

    double dot;

    // set rho and E to be the same in the ghost cell
    U_right[0] = U_left[0];
    U_right[3] = U_left[3];

    // normal reflection
    dot = sqrtf(U_left[1] * nx + U_left[2] * ny);

    if (dot > 0) {
        U_right[1] = -U_left[1];
        U_right[2] = U_left[2];
    } else {
        U_right[1] = U_left[1];
        U_right[2] = -U_left[2];
    }
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
