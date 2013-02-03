__device__ void evalU0(double *U, double v1x, double v1y, double v2x, double v2y, double v3x, double v3y, int i, int n_p, int n_quad);
__device__ void eval_flux(double *U, double *flux_x, double *flux_y);
__device__ double eval_lambda(double *U_left, double *U_right, double nx, double ny);
__device__ void inflow_boundary(double *U_left, double *U_right, double v1x, double v1y, double v2x, double v2y, double v3x, double v3y, double nx, double ny, int j, int left_side, int n_quad1d, double t);
__device__ void outflow_boundary(double *U_left, double *U_right, double v1x, double v1y, double v2x, double v2y, double v3x, double v3y, double nx, double ny, int j, int left_side, int n_quad1d, double t);
__device__ void reflecting_boundary(double *U_left, double *U_right, double v1x, double v1y, double v2x, double v2y, double v3x, double v3y, double nx, double ny, int j, int left_side, int n_quad1d);
__global__ void eval_global_lambda(double *C, double *lambda, int n_quad, int n_p, int idx);
__device__ double U0(double x, double y);
__device__ double U1(double x, double y);
__device__ double U2(double x, double y);
__device__ double U3(double x, double y);
