/* time_integrator.cu
 *
 * time integration functions.
 */
#ifndef TIMEINTEGRATOR_H_GUARD
#define TIMEINTEGRATOR_H_GUARD
#endif

#define TOL 10e-15

/***********************
 * RK4 
 ***********************/

/* tempstorage for RK4
 * 
 * I need to store u + alpha * k_i into some temporary variable called k*.
 */
void rk4_tempstorage(double *c, double *kstar, double*k, double alpha, int n_p, int num_elem) {

    int idx;

    for (idx = 0; idx < 4 * n_p * num_elem; idx++) {
        kstar[idx] = c[idx] + alpha * k[idx];
    }
}

/* rk4
 *
 * computes the runge-kutta solution 
 * u_n+1 = u_n + k1/6 + k2/3 + k3/3 + k4/6
 */
void rk4(double *c, double *k1, double *k2, double *k3, double *k4, int n_p, int num_elem) {
    int idx;

    for (idx = 0; idx < n_p * num_elem; idx++) {
        c[num_elem * n_p * 0 + idx] += k1[num_elem * n_p * 0 + idx]/6. + k2[num_elem * n_p * 0 + idx]/3. + k3[num_elem * n_p * 0 + idx]/3. + k4[num_elem * n_p * 0 + idx]/6.;
        c[num_elem * n_p * 1 + idx] += k1[num_elem * n_p * 1 + idx]/6. + k2[num_elem * n_p * 1 + idx]/3. + k3[num_elem * n_p * 1 + idx]/3. + k4[num_elem * n_p * 1 + idx]/6.;
        c[num_elem * n_p * 2 + idx] += k1[num_elem * n_p * 2 + idx]/6. + k2[num_elem * n_p * 2 + idx]/3. + k3[num_elem * n_p * 2 + idx]/3. + k4[num_elem * n_p * 2 + idx]/6.;
        c[num_elem * n_p * 3 + idx] += k1[num_elem * n_p * 3 + idx]/6. + k2[num_elem * n_p * 3 + idx]/3. + k3[num_elem * n_p * 3 + idx]/3. + k4[num_elem * n_p * 3 + idx]/6.;
    }
}

void sanity_check(double *c, int num_elem, int n_p) {
    double rho_avg, u_avg, v_avg, E_avg, p;

    int idx;

    for (idx = 0; idx < num_elem; idx++) {
        rho_avg = c[num_elem * n_p * 0 + idx] * 1.414213562373095E+00;
        u_avg   = c[num_elem * n_p * 1 + idx] * 1.414213562373095E+00;
        v_avg   = c[num_elem * n_p * 2 + idx] * 1.414213562373095E+00;
        E_avg   = c[num_elem * n_p * 3 + idx] * 1.414213562373095E+00;

        u_avg = u_avg / rho_avg;
        v_avg = v_avg / rho_avg;

        //p = pressure(rho_avg, u_avg, v_avg, E_avg, 30000, idx);
    }
}

/* right hand side
 *
 * computes the sum of the quadrature and the riemann flux for the 
 * coefficients for each element
 * THREADS: num_elem
 */
void eval_rhs_rk4(double *c, double *quad_rhs, double *left_riemann_rhs, double *right_riemann_rhs, 
                  int *elem_s1, int *elem_s2, int *elem_s3,
                  int *left_elem, double *J, 
                  double dt, int n_p, int num_sides, int num_elem) {
    int idx;

    double s1_eqn1, s2_eqn1, s3_eqn1;
    double s1_eqn2, s2_eqn2, s3_eqn2;
    double s1_eqn3, s2_eqn3, s3_eqn3;
    double s1_eqn4, s2_eqn4, s3_eqn4;
    double register_J;
    int i, s1_idx, s2_idx, s3_idx;

    for (idx = 0; idx < num_elem; idx++) {

        register_J = J[idx];

        // get the indicies for the riemann contributions for this element
        s1_idx = elem_s1[idx];
        s2_idx = elem_s2[idx];
        s3_idx = elem_s3[idx];

        for (i = 0; i < n_p; i++) {

            // determine left or right pointing
            if (idx == left_elem[s1_idx]) {
                s1_eqn1 = left_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s1_idx];
                s1_eqn2 = left_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s1_idx];
                s1_eqn3 = left_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s1_idx];
                s1_eqn4 = left_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s1_idx];
            } else {
                s1_eqn1 = right_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s1_idx];
                s1_eqn2 = right_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s1_idx];
                s1_eqn3 = right_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s1_idx];
                s1_eqn4 = right_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s1_idx];
            }

            if (idx == left_elem[s2_idx]) {
                s2_eqn1 = left_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s2_idx];
                s2_eqn2 = left_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s2_idx];
                s2_eqn3 = left_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s2_idx];
                s2_eqn4 = left_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s2_idx];
            } else {
                s2_eqn1 = right_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s2_idx];
                s2_eqn2 = right_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s2_idx];
                s2_eqn3 = right_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s2_idx];
                s2_eqn4 = right_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s2_idx];
            }

            if (idx == left_elem[s3_idx]) {
                s3_eqn1 = left_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s3_idx];
                s3_eqn2 = left_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s3_idx];
                s3_eqn3 = left_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s3_idx];
                s3_eqn4 = left_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s3_idx];
            } else {
                s3_eqn1 = right_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s3_idx];
                s3_eqn2 = right_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s3_idx];
                s3_eqn3 = right_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s3_idx];
                s3_eqn4 = right_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s3_idx];
            }

            // calculate the coefficient c
            c[num_elem * n_p * 0 + i * num_elem + idx] = 1. / register_J * dt * (quad_rhs[num_elem * n_p * 0 + i * num_elem + idx] + s1_eqn1 + s2_eqn1 + s3_eqn1);
            c[num_elem * n_p * 1 + i * num_elem + idx] = 1. / register_J * dt * (quad_rhs[num_elem * n_p * 1 + i * num_elem + idx] + s1_eqn2 + s2_eqn2 + s3_eqn2);
            c[num_elem * n_p * 2 + i * num_elem + idx] = 1. / register_J * dt * (quad_rhs[num_elem * n_p * 2 + i * num_elem + idx] + s1_eqn3 + s2_eqn3 + s3_eqn3);
            c[num_elem * n_p * 3 + i * num_elem + idx] = 1. / register_J * dt * (quad_rhs[num_elem * n_p * 3 + i * num_elem + idx] + s1_eqn4 + s2_eqn4 + s3_eqn4);
        }
    }
}

void time_integrate_rk4(int n_quad, int n_quad1d, int n_p, int n, int num_elem, int num_sides,
                        double endtime, double min_r) {
    int n_threads = 128;
    int i;
    double dt, t;

    double max_l;
    double *max_lambda = (double *) malloc(num_elem * sizeof(double));
    double *c = (double *) malloc(num_elem * n_p * 4 * sizeof(double));

    t = 0;

    double convergence = 1 + TOL;

    while (t < endtime && convergence > TOL) {
        sanity_check(d_c, num_elem, n_p);
        // compute all the lambda values over each cell
        eval_global_lambda(d_c, d_lambda, n_quad, n_p, num_elem);

        // find the max value of lambda
        memcpy(max_lambda, d_lambda, num_elem * sizeof(double));
        max_l = max_lambda[0];
        for (i = 0; i < num_elem; i++) {
            max_l = (max_lambda[i] > max_l) ? max_lambda[i] : max_l;
        }

        // keep CFL condition
        if (t + dt > endtime) {
            dt = endtime - t;
            t = endtime;
        } else {
            dt  = 0.7 * min_r / max_l /  (2. * n + 1.);
            t += dt;
        }

        //printf(" > (%lf), t = %lf\n", max_l, t);

        // stage 1
        eval_surface(d_c, d_left_riemann_rhs, d_right_riemann_rhs, 
                     d_s_length, 
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_quad, n_p, num_sides, num_elem, t);

        eval_volume(d_c, d_quad_rhs, 
                    d_xr, d_yr, d_xs, d_ys,
                    n_quad, n_p, num_elem);

        eval_rhs_rk4(d_k1, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                     d_elem_s1, d_elem_s2, d_elem_s3, 
                     d_left_elem, d_J, dt, n_p, num_sides, num_elem);

        rk4_tempstorage(d_c, d_kstar, d_k1, 0.5, n_p, num_elem);

        // stage 2
        eval_surface(d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, 
                     d_s_length, 
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_quad, n_p, num_sides, num_elem, t);

        eval_volume(d_kstar, d_quad_rhs, 
                    d_xr, d_yr, d_xs, d_ys,
                    n_quad, n_p, num_elem);

        eval_rhs_rk4(d_k2, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs,
                     d_elem_s1, d_elem_s2, d_elem_s3, 
                     d_left_elem, d_J, dt, n_p, num_sides, num_elem);

        rk4_tempstorage(d_c, d_kstar, d_k2, 0.5, n_p, num_elem);


        // stage 3
        eval_surface(d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, 
                     d_s_length, 
                     d_V1x, d_V1y,
                     d_V2x, d_V2y,
                     d_V3x, d_V3y,
                     d_left_elem, d_right_elem,
                     d_left_side_number, d_right_side_number,
                     d_Nx, d_Ny, 
                     n_quad1d, n_quad, n_p, num_sides, num_elem, t);

        eval_volume(d_kstar, d_quad_rhs, 
                    d_xr, d_yr, d_xs, d_ys,
                    n_quad, n_p, num_elem);

        eval_rhs_rk4(d_k3, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                     d_elem_s1, d_elem_s2, d_elem_s3, 
                     d_left_elem, d_J, dt, n_p, num_sides, num_elem);

        rk4_tempstorage(d_c, d_kstar, d_k3, 1.0, n_p, num_elem);


        // stage 4
        eval_surface(d_kstar, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_quad, n_p, num_sides, num_elem, t);

        eval_volume(d_kstar, d_quad_rhs, 
                    d_xr, d_yr, d_xs, d_ys,
                    n_quad, n_p, num_elem);

        eval_rhs_rk4(d_k4, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                     d_elem_s1, d_elem_s2, d_elem_s3, 
                     d_left_elem, d_J, dt, n_p, num_sides, num_elem);

        // combine them all
        rk4(d_c, d_k1, d_k2, d_k3, d_k4, n_p, num_elem);

        //if (t - dt > 0.) {
            //check_convergence(d_c_prev, d_c, num_elem, n_p);
            //memcpy(c, d_c_prev, num_elem * n_p * 4 * sizeof(double));

            //convergence = 0.;
            //for (i = 0; i < num_elem * n_p * 4; i++) {
                //convergence += c[i];
            //}

            //convergence = sqrtf(convergence);

            //printf(" > convergence = %.015lf\n", convergence);
        //}

        //memcpy(d_c_prev, d_c, num_elem * n_p * 4 * sizeof(double));
    }

    free(c);
    free(max_lambda);
}

/***********************
 * FORWARD EULER
 ***********************/

void eval_rhs_fe(double *c, double *quad_rhs, double *left_riemann_rhs, double *right_riemann_rhs, 
                 int *elem_s1, int *elem_s2, int *elem_s3,
                 int *left_elem, double *J, 
                 double dt, int n_p, int num_sides, int num_elem) {
    int idx;
    double s1_eqn1, s2_eqn1, s3_eqn1;
    double s1_eqn2, s2_eqn2, s3_eqn2;
    double s1_eqn3, s2_eqn3, s3_eqn3;
    double s1_eqn4, s2_eqn4, s3_eqn4;
    double register_J;
    int i, s1_idx, s2_idx, s3_idx;

    for (idx = 0; idx < num_elem; idx++) {

        register_J = J[idx];

        // get the indicies for the riemann contributions for this element
        s1_idx = elem_s1[idx];
        s2_idx = elem_s2[idx];
        s3_idx = elem_s3[idx];

        for (i = 0; i < n_p; i++) {

            // determine left or right pointing
            if (idx == left_elem[s1_idx]) {
                s1_eqn1 = left_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s1_idx];
                s1_eqn2 = left_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s1_idx];
                s1_eqn3 = left_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s1_idx];
                s1_eqn4 = left_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s1_idx];
            } else {
                s1_eqn1 = right_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s1_idx];
                s1_eqn2 = right_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s1_idx];
                s1_eqn3 = right_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s1_idx];
                s1_eqn4 = right_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s1_idx];
            }

            if (idx == left_elem[s2_idx]) {
                s2_eqn1 = left_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s2_idx];
                s2_eqn2 = left_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s2_idx];
                s2_eqn3 = left_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s2_idx];
                s2_eqn4 = left_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s2_idx];
            } else {
                s2_eqn1 = right_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s2_idx];
                s2_eqn2 = right_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s2_idx];
                s2_eqn3 = right_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s2_idx];
                s2_eqn4 = right_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s2_idx];
            }

            if (idx == left_elem[s3_idx]) {
                s3_eqn1 = left_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s3_idx];
                s3_eqn2 = left_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s3_idx];
                s3_eqn3 = left_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s3_idx];
                s3_eqn4 = left_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s3_idx];
            } else {
                s3_eqn1 = right_riemann_rhs[num_sides * n_p * 0 + i * num_sides + s3_idx];
                s3_eqn2 = right_riemann_rhs[num_sides * n_p * 1 + i * num_sides + s3_idx];
                s3_eqn3 = right_riemann_rhs[num_sides * n_p * 2 + i * num_sides + s3_idx];
                s3_eqn4 = right_riemann_rhs[num_sides * n_p * 3 + i * num_sides + s3_idx];
            }

            // calculate the coefficient c
            c[num_elem * n_p * 0 + i * num_elem + idx] += 1. / register_J * dt * (quad_rhs[num_elem * n_p * 0 + i * num_elem + idx] + s1_eqn1 + s2_eqn1 + s3_eqn1);
            c[num_elem * n_p * 1 + i * num_elem + idx] += 1. / register_J * dt * (quad_rhs[num_elem * n_p * 1 + i * num_elem + idx] + s1_eqn2 + s2_eqn2 + s3_eqn2);
            c[num_elem * n_p * 2 + i * num_elem + idx] += 1. / register_J * dt * (quad_rhs[num_elem * n_p * 2 + i * num_elem + idx] + s1_eqn3 + s2_eqn3 + s3_eqn3);
            c[num_elem * n_p * 3 + i * num_elem + idx] += 1. / register_J * dt * (quad_rhs[num_elem * n_p * 3 + i * num_elem + idx] + s1_eqn4 + s2_eqn4 + s3_eqn4);
        }
    }
}

// forward eulers
void time_integrate_fe(int n_quad, int n_quad1d, int n_p, int n, 
              int num_elem, int num_sides, double endtime, double min_r) {
    int i;
    double t, dt;
    double *max_lambda;
    double max_l;

    t = 0;
    while (t < endtime) {
        eval_global_lambda(d_c, d_lambda, n_quad, n_p, num_elem);

        // find the max value of lambda
        max_lambda = (double *) malloc(num_elem * sizeof(double));
        memcpy(max_lambda, d_lambda, num_elem * sizeof(double));
        max_l = max_lambda[0];
        for (i = 0; i < num_elem; i++) {
            max_l = (max_lambda[i] > max_l) ? max_lambda[i] : max_l;
        }
        free(max_lambda);

        // keep CFL condition
        dt  = 0.7 * min_r / max_l /  (2. * n + 1.);

        // add to total time
        t += dt;
        printf(" > (%lf), t = %lf\n", max_l, t);

        eval_surface(d_c, d_left_riemann_rhs, d_right_riemann_rhs, 
                         d_s_length, 
                         d_V1x, d_V1y,
                         d_V2x, d_V2y,
                         d_V3x, d_V3y,
                         d_left_elem, d_right_elem,
                         d_left_side_number, d_right_side_number,
                         d_Nx, d_Ny, 
                         n_quad1d, n_quad, n_p, num_sides, num_elem, t);

        eval_volume(d_c, d_quad_rhs, 
                        d_xr, d_yr, d_xs, d_ys,
                        n_quad, n_p, num_elem);

        eval_rhs_fe(d_c, d_quad_rhs, d_left_riemann_rhs, d_right_riemann_rhs, 
                    d_elem_s1, d_elem_s2, d_elem_s3, 
                    d_left_elem, d_J, dt, n_p, num_sides, num_elem);
    }
}
