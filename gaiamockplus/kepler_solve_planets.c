/* Fast routines for fitting Gaia epoch astrometry and/or radial velocities of systems with any number of planets
(dark companions).
These are used by the fitting functions in gaiamockplus.py, and are kept separate from gaiamock's
kepler_solve_astrometry.c so that the upstream code can be updated independently. Compile with, e.g.:
 gcc -O2 -shared -fPIC -o kepler_solve_planets.so kepler_solve_planets.c -I$(brew --prefix gsl)/include -L$(brew --prefix gsl)/lib -lgsl -lgslcblas -lm
or, where GSL is already linked by default:
 gcc -O2 -shared -fPIC -o kepler_solve_planets.so kepler_solve_planets.c -lgsl -lgslcblas -lm

Model: for fixed nonlinear parameters (period P, phase phi_p = 2 pi Tp/P, and eccentricity e of each planet), the
along-scan astrometry is linear in 5 + 4N parameters, ordered as in gaiamock:
    ra_off, pmra, dec_off, pmdec, parallax, then (B, G, A, F) for each planet,
with design-matrix columns sin(psi), t sin(psi), cos(psi), t cos(psi), plx_factor, X sin(psi), Y sin(psi),
X cos(psi), Y cos(psi), where X = cos(E) - e and Y = sqrt(1 - e^2) sin(E). Likewise, the RVs are linear in
n_sets + 2N parameters: one systemic velocity per RV dataset (instrument), then (K cos(w), K sin(w)) for each planet,
with columns cos(nu) + e and -sin(nu) (nu: true anomaly; w: argument of periastron of the star's orbit). The linear
parameters are always solved exactly (weighted least squares); only the 3N nonlinear parameters are searched.

Options shared by most functions:
    plx_mu, plx_sig: Gaussian parallax prior parallax = plx_mu +/- plx_sig (in mas); none if plx_sig <= 0
    reject: if 1, the astrometric epoch with the largest weighted residual of the full solution is dropped and the
        linear solution is recomputed without it (the same "drop-one" objective as gaiamock's reject_outlier option)
    rv_set, n_sets: the RV dataset (0 to n_sets - 1) of each RV, and the number of RV datasets; rv_err should include
        any jitter, added in quadrature
    ecc_a, ecc_b: Beta(ecc_a, ecc_b) prior on each eccentricity; none if ecc_a <= 0 or ecc_b <= 0
    ast_skip: bit mask of the planets (bit i for planet i) with no astrometric signal ("RV-only" planets, e.g. a planet
        whose astrometric signal is negligible): they have no astrometric columns (their B, G, A, F are returned as
        0), and enter the RVs only. Search functions never skip the new planet. */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>

#define PI 3.14159265358979 /* the same value as in gaiamock's kepler_solve_astrometry.c */
#define GMP_MAX_PLANETS 10
#define GMP_MAX_SETS 20 /* RV datasets; GMP_MAX_LIN must be >= GMP_MAX_SETS + 2*GMP_MAX_PLANETS */
#define GMP_MAX_LIN (5 + 4*GMP_MAX_PLANETS)

/* ---------------------------------------------------------------- Kepler's equation (as in gaiamock) */

static double gmp_eps3(double ecc, double M, double x){
    double t1 = cos(x);
    double t2 = -1 + ecc*t1;
    double t3 = sin(x);
    double t4 = ecc*t3;
    double t5 = -x + t4 + M;
    double t6 = t5/(0.5*t5*t4/t2 + t2);
    return t5/((0.5*t3 - (1.0/6.0)*t1*t6)*ecc*t6 + t2);
}

static double gmp_kepler_start3(double ecc, double M){
    double t34 = ecc*ecc;
    double t35 = ecc*t34;
    double t33 = cos(M);
    return M + (-0.5*t35 + ecc + (t34 + 1.5*t33*t35)*t33)*sin(M);
}

static double gmp_solve_kepler(double Mi, double ecc){
    double xtol = 1e-10;
    double eps = 1;
    double Mnorm = fmod(Mi, 2*PI);
    if (Mnorm < 0.0) {
        Mnorm += 2*PI;
    }
    double EE = gmp_kepler_start3(ecc, Mnorm);
    int n_iter = 0;
    while (eps > xtol && n_iter < 100){
        double EE0 = EE;
        EE = EE0 - gmp_eps3(ecc, Mnorm, EE0);
        eps = fabs(EE0 - EE);
        n_iter++;
    }
    return EE;
}

/* ---------------------------------------------------------------- data and design matrices */

#define GMP_AST 0
#define GMP_RV 1

/* One dataset: Gaia along-scan astrometry or radial velocities. For fixed nonlinear parameters of the planets, the
model of each dataset is linear in n_base + n_pp*N parameters:
  astrometry: n_base = 5 (ra_off, pmra, dec_off, pmdec, parallax), n_pp = 4 per planet (B, G, A, F), as in gaiamock;
  RVs: n_base = n_sets (one systemic velocity per RV dataset), n_pp = 2 per planet (K cos(w), K sin(w)), since
       K [cos(nu + w) + e cos(w)] = K cos(w) [cos(nu) + e] - K sin(w) sin(nu). */
typedef struct {
    int type, n;
    const double *t, *plx, *obs;
    double *spsi, *cpsi, *w; /* sin(psi), cos(psi) (astrometry only), and weights 1/err^2 */
    double plx_mu, plx_sig;  /* parallax prior (astrometry only; none if plx_sig <= 0) */
    int reject;              /* drop the worst epoch (astrometry only) */
    const int *set;          /* RV dataset index of each epoch (RVs only) */
    int n_sets, n_base, n_pp;
    int skip;                /* bit mask of the planets with no columns (astrometry only; see ast_skip) */
} gmp_data;

static int gmp_skipped(const gmp_data *d, int i){
    return (d->skip >> i) & 1;
}

/* first column of planet i (also the number of columns of planets 0 to i - 1, after the base columns) */
static int gmp_col(const gmp_data *d, int i){
    int c = d->n_base;
    for (int j = 0; j < i; j++) if (!gmp_skipped(d, j)) c += d->n_pp;
    return c;
}

static int gmp_data_init(gmp_data *d, int n, const double *t, const double *psi, const double *plx, const double *obs, const double *err, double plx_mu, double plx_sig, int reject){
    d->type = GMP_AST; d->n_base = 5; d->n_pp = 4; d->set = NULL; d->n_sets = 0; d->skip = 0;
    d->n = n; d->t = t; d->plx = plx; d->obs = obs;
    d->plx_mu = plx_mu; d->plx_sig = plx_sig; d->reject = reject;
    d->spsi = (double*)malloc(3*(size_t)n*sizeof(double));
    if (d->spsi == NULL) return 1;
    d->cpsi = d->spsi + n;
    d->w = d->spsi + 2*n;
    for (int j = 0; j < n; j++){
        d->spsi[j] = sin(psi[j]);
        d->cpsi[j] = cos(psi[j]);
        d->w[j] = 1.0/(err[j]*err[j]);
    }
    return 0;
}

static int gmp_data_init_rv(gmp_data *d, int n, const double *t, const double *rv, const double *rv_err, const int *set, int n_sets){
    d->type = GMP_RV; d->n_base = n_sets; d->n_pp = 2; d->set = set; d->n_sets = n_sets; d->skip = 0;
    d->n = n; d->t = t; d->plx = NULL; d->obs = rv;
    d->plx_mu = 0.0; d->plx_sig = -1.0; d->reject = 0;
    d->spsi = (double*)malloc((size_t)n*sizeof(double));
    if (d->spsi == NULL) return 1;
    d->cpsi = NULL;
    d->w = d->spsi;
    for (int j = 0; j < n; j++){
        if (set[j] < 0 || set[j] >= n_sets) { free(d->spsi); return 1; }
        d->w[j] = 1.0/(rv_err[j]*rv_err[j]);
    }
    return 0;
}

static void gmp_data_free(gmp_data *d){
    free(d->spsi);
}

/* columns are stored column-major: column k is A[k*n ... k*n + n - 1] */
static void gmp_fill_base_columns(const gmp_data *d, double *A){
    int n = d->n;
    if (d->type == GMP_RV){
        for (int k = 0; k < d->n_sets; k++)
            for (int j = 0; j < n; j++) A[k*n + j] = (d->set[j] == k) ? 1.0 : 0.0;
        return;
    }
    for (int j = 0; j < n; j++){
        A[j] = d->spsi[j];
        A[n + j] = d->t[j]*d->spsi[j];
        A[2*n + j] = d->cpsi[j];
        A[3*n + j] = d->t[j]*d->cpsi[j];
        A[4*n + j] = d->plx[j];
    }
}

static void gmp_fill_planet_columns(const gmp_data *d, double *A, int col0, double P, double phi, double e){
    int n = d->n;
    double sq = sqrt(1 - e*e);
    for (int j = 0; j < n; j++){
        double E = gmp_solve_kepler(2*PI*d->t[j]*365.25/P - phi, e);
        if (d->type == GMP_RV){
            double denom = 1 - e*cos(E);
            double cos_nu = (cos(E) - e)/denom, sin_nu = sq*sin(E)/denom;
            A[col0*n + j] = cos_nu + e;
            A[(col0 + 1)*n + j] = -sin_nu;
            continue;
        }
        double X = cos(E) - e;
        double Y = sq*sin(E);
        A[col0*n + j] = X*d->spsi[j];
        A[(col0 + 1)*n + j] = Y*d->spsi[j];
        A[(col0 + 2)*n + j] = X*d->cpsi[j];
        A[(col0 + 3)*n + j] = Y*d->cpsi[j];
    }
}

/* circular orbit: the phase is absorbed by the linear coefficients (as in gaiamock's periodogram; for RVs this is
the generalized Lomb-Scargle periodogram with one offset per RV dataset) */
static void gmp_fill_circular_columns(const gmp_data *d, double *A, int col0, double P){
    int n = d->n;
    for (int j = 0; j < n; j++){
        double phase = 2*PI*d->t[j]*365.25/P;
        double cp = cos(phase), sp = sin(phase);
        if (d->type == GMP_RV){
            A[col0*n + j] = cp;
            A[(col0 + 1)*n + j] = sp;
            continue;
        }
        A[col0*n + j] = cp*d->spsi[j];
        A[(col0 + 1)*n + j] = sp*d->spsi[j];
        A[(col0 + 2)*n + j] = cp*d->cpsi[j];
        A[(col0 + 3)*n + j] = sp*d->cpsi[j];
    }
}

/* ---------------------------------------------------------------- weighted linear least squares */

/* Solve the symmetric positive (semi)definite system Nm x = b. Tries a Cholesky factorization first; if the
matrix is not numerically positive definite, falls back to a truncated SVD (minimum-norm solution).
Nm is not modified. Returns 0 on success. */
static int gmp_spd_solve(int p, const double *Nm, const double *b, double *x){
    double L[GMP_MAX_LIN*GMP_MAX_LIN];
    int ok = 1;
    for (int i = 0; i < p && ok; i++){
        for (int k = 0; k <= i; k++){
            double s = Nm[i*p + k];
            for (int m = 0; m < k; m++) s -= L[i*p + m]*L[k*p + m];
            if (i == k){
                if (!(s > 1e-13*Nm[i*p + i]) || !isfinite(s)) { ok = 0; break; }
                L[i*p + i] = sqrt(s);
            } else {
                L[i*p + k] = s/L[k*p + k];
            }
        }
    }
    if (ok){
        double y[GMP_MAX_LIN];
        for (int i = 0; i < p; i++){
            double s = b[i];
            for (int m = 0; m < i; m++) s -= L[i*p + m]*y[m];
            y[i] = s/L[i*p + i];
        }
        for (int i = p - 1; i >= 0; i--){
            double s = y[i];
            for (int m = i + 1; m < p; m++) s -= L[m*p + i]*x[m];
            x[i] = s/L[i*p + i];
        }
        return 0;
    }
    /* fallback: truncated SVD */
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    gsl_matrix *U = gsl_matrix_alloc(p, p);
    gsl_matrix *V = gsl_matrix_alloc(p, p);
    gsl_vector *S = gsl_vector_alloc(p);
    gsl_vector *work = gsl_vector_alloc(p);
    for (int i = 0; i < p; i++) for (int k = 0; k < p; k++) gsl_matrix_set(U, i, k, Nm[i*p + k]);
    int status = gsl_linalg_SV_decomp(U, V, S, work);
    if (status == GSL_SUCCESS){
        double smax = gsl_vector_get(S, 0);
        for (int i = 0; i < p; i++) x[i] = 0.0;
        for (int k = 0; k < p; k++){
            double sk = gsl_vector_get(S, k);
            if (!(sk > 1e-13*smax)) continue;
            double ub = 0.0;
            for (int i = 0; i < p; i++) ub += gsl_matrix_get(U, i, k)*b[i];
            for (int i = 0; i < p; i++) x[i] += gsl_matrix_get(V, i, k)*ub/sk;
        }
    }
    gsl_matrix_free(U); gsl_matrix_free(V); gsl_vector_free(S); gsl_vector_free(work);
    gsl_set_error_handler(old_handler);
    return status == GSL_SUCCESS ? 0 : 1;
}

/* Weighted least squares with the p columns in A. Writes the solution to mu and returns the data chi2
(excluding the rejected epoch, if any). *prior_term is the parallax-prior contribution, (mu_plx - plx_mu)^2/plx_sig^2,
and *worst is the index of the rejected epoch (-1 if none). Returns INFINITY if the solve fails. */
static double gmp_linear_solve(const gmp_data *d, int p, const double *A, double *mu, double *prior_term, int *worst){
    int n = d->n;
    double Nm[GMP_MAX_LIN*GMP_MAX_LIN], b[GMP_MAX_LIN];
    *prior_term = 0.0;
    *worst = -1;
    for (int a = 0; a < p; a++){
        double s = 0.0;
        for (int j = 0; j < n; j++) s += d->w[j]*A[a*n + j]*d->obs[j];
        b[a] = s;
        for (int c = 0; c <= a; c++){
            double s2 = 0.0;
            for (int j = 0; j < n; j++) s2 += d->w[j]*A[a*n + j]*A[c*n + j];
            Nm[a*p + c] = s2;
            Nm[c*p + a] = s2;
        }
    }
    if (d->plx_sig > 0){
        Nm[4*p + 4] += 1.0/(d->plx_sig*d->plx_sig);
        b[4] += d->plx_mu/(d->plx_sig*d->plx_sig);
    }
    if (gmp_spd_solve(p, Nm, b, mu)) return INFINITY;

    double chi2 = 0.0, worst_chi2 = -1.0;
    int jw = -1;
    for (int j = 0; j < n; j++){
        double pred = 0.0;
        for (int a = 0; a < p; a++) pred += A[a*n + j]*mu[a];
        double c2 = d->w[j]*(d->obs[j] - pred)*(d->obs[j] - pred);
        chi2 += c2;
        if (c2 > worst_chi2) { worst_chi2 = c2; jw = j; }
    }

    if (d->reject && n > p + 1 && jw >= 0){
        /* remove the worst epoch from the normal equations and re-solve */
        for (int a = 0; a < p; a++){
            b[a] -= d->w[jw]*A[a*n + jw]*d->obs[jw];
            for (int c = 0; c < p; c++) Nm[a*p + c] -= d->w[jw]*A[a*n + jw]*A[c*n + jw];
        }
        if (gmp_spd_solve(p, Nm, b, mu)) return INFINITY;
        chi2 = 0.0;
        for (int j = 0; j < n; j++){
            if (j == jw) continue;
            double pred = 0.0;
            for (int a = 0; a < p; a++) pred += A[a*n + j]*mu[a];
            chi2 += d->w[j]*(d->obs[j] - pred)*(d->obs[j] - pred);
        }
        *worst = jw;
    }
    if (d->plx_sig > 0) *prior_term = (mu[4] - d->plx_mu)*(mu[4] - d->plx_mu)/(d->plx_sig*d->plx_sig);
    return chi2;
}

/* Cholesky factorization Nm = L L^T of a p x p symmetric positive definite matrix (row-major; L lower triangular).
Returns 0 on success, 1 if the matrix is not numerically positive definite. */
static int gmp_cholesky(int p, const double *Nm, double *L){
    for (int i = 0; i < p; i++){
        for (int k = 0; k <= i; k++){
            double s = Nm[i*p + k];
            for (int m = 0; m < k; m++) s -= L[i*p + m]*L[k*p + m];
            if (i == k){
                if (!(s > 1e-13*Nm[i*p + i]) || !isfinite(s)) return 1;
                L[i*p + i] = sqrt(s);
            } else {
                L[i*p + k] = s/L[k*p + k];
            }
        }
    }
    return 0;
}

/* Solve L L^T x = b, with L from gmp_cholesky() */
static void gmp_cholesky_solve(int p, const double *L, const double *b, double *x){
    double y[GMP_MAX_LIN];
    for (int i = 0; i < p; i++){
        double s = b[i];
        for (int m = 0; m < i; m++) s -= L[i*p + m]*y[m];
        y[i] = s/L[i*p + i];
    }
    for (int i = p - 1; i >= 0; i--){
        double s = y[i];
        for (int m = i + 1; m < p; m++) s -= L[m*p + i]*x[m];
        x[i] = s/L[i*p + i];
    }
}

/* Weighted least squares as gmp_linear_solve() (with the same rejection of the worst epoch), also returning ln det of
the normal matrix (for the marginal likelihood) and, if cov is not NULL, its inverse (the covariance of the linear
parameters, p x p, row-major). Returns the chi2 at the solution, or INFINITY if the normal matrix is not positive definite. */
static double gmp_linear_solve_cov(const gmp_data *d, int p, const double *A, double *mu, double *logdet, double *cov, int *worst){
    int n = d->n;
    double Nm[GMP_MAX_LIN*GMP_MAX_LIN], b[GMP_MAX_LIN], L[GMP_MAX_LIN*GMP_MAX_LIN];
    *worst = -1;
    for (int a = 0; a < p; a++){
        double s = 0.0;
        for (int j = 0; j < n; j++) s += d->w[j]*A[a*n + j]*d->obs[j];
        b[a] = s;
        for (int c = 0; c <= a; c++){
            double s2 = 0.0;
            for (int j = 0; j < n; j++) s2 += d->w[j]*A[a*n + j]*A[c*n + j];
            Nm[a*p + c] = s2;
            Nm[c*p + a] = s2;
        }
    }
    if (d->plx_sig > 0){
        Nm[4*p + 4] += 1.0/(d->plx_sig*d->plx_sig);
        b[4] += d->plx_mu/(d->plx_sig*d->plx_sig);
    }
    if (gmp_cholesky(p, Nm, L)) return INFINITY;
    gmp_cholesky_solve(p, L, b, mu);
    if (d->reject && n > p + 1){
        /* the epoch with the largest weighted residual of the full solution is dropped, as in gmp_linear_solve() */
        double worst_chi2 = -1.0;
        int jw = -1;
        for (int j = 0; j < n; j++){
            double pred = 0.0;
            for (int a = 0; a < p; a++) pred += A[a*n + j]*mu[a];
            double c2 = d->w[j]*(d->obs[j] - pred)*(d->obs[j] - pred);
            if (c2 > worst_chi2) { worst_chi2 = c2; jw = j; }
        }
        for (int a = 0; a < p; a++){
            b[a] -= d->w[jw]*A[a*n + jw]*d->obs[jw];
            for (int c = 0; c < p; c++) Nm[a*p + c] -= d->w[jw]*A[a*n + jw]*A[c*n + jw];
        }
        if (gmp_cholesky(p, Nm, L)) return INFINITY;
        gmp_cholesky_solve(p, L, b, mu);
        *worst = jw;
    }
    double chi2 = 0.0;
    for (int j = 0; j < n; j++){
        if (j == *worst) continue;
        double pred = 0.0;
        for (int a = 0; a < p; a++) pred += A[a*n + j]*mu[a];
        chi2 += d->w[j]*(d->obs[j] - pred)*(d->obs[j] - pred);
    }
    *logdet = 0.0;
    for (int i = 0; i < p; i++) *logdet += 2*log(L[i*p + i]);
    if (cov != NULL){
        double unit[GMP_MAX_LIN], col[GMP_MAX_LIN];
        for (int c = 0; c < p; c++){
            for (int i = 0; i < p; i++) unit[i] = (i == c) ? 1.0 : 0.0;
            gmp_cholesky_solve(p, L, unit, col);
            for (int i = 0; i < p; i++) cov[i*p + c] = col[i];
        }
    }
    return chi2;
}

/* ---------------------------------------------------------------- the fitting problem: astrometry and/or RVs */

/* A problem is an astrometric dataset, an RV dataset, or both. For fixed nonlinear parameters (P, phi, e) of each
planet, the linear parameters of each dataset are solved separately, and the objective is the sum of the chi2 of the
datasets (plus the parallax-prior term). Every exported function below takes both datasets; set n_ast = 0 or
n_rv = 0 to leave one out. */
typedef struct {
    int nd;
    gmp_data d[2];
    double *A[2];
} gmp_problem;

#define GMP_DATA_ARGS int n_ast, const double *t, const double *psi, const double *plx, const double *obs, const double *err, \
                      double plx_mu, double plx_sig, int reject, \
                      int n_rv, const double *t_rv, const double *rv, const double *rv_err, const int *rv_set, int n_sets, int ast_skip
#define GMP_DATA_PASS n_ast, t, psi, plx, obs, err, plx_mu, plx_sig, reject, n_rv, t_rv, rv, rv_err, rv_set, n_sets, ast_skip

static void gmp_problem_free(gmp_problem *pr){
    for (int k = 0; k < pr->nd; k++){
        free(pr->A[k]);
        gmp_data_free(&pr->d[k]);
    }
    pr->nd = 0;
}

/* Set up the datasets, with design matrices for n_pl planets (base columns filled). Returns 0 on success. */
static int gmp_problem_init(gmp_problem *pr, GMP_DATA_ARGS, int n_pl){
    pr->nd = 0;
    if (n_pl < 0 || n_pl > GMP_MAX_PLANETS || n_sets > GMP_MAX_SETS) return 1;
    if (n_ast > 0){
        if (gmp_data_init(&pr->d[pr->nd], n_ast, t, psi, plx, obs, err, plx_mu, plx_sig, reject)) return 1;
        pr->d[pr->nd].skip = ast_skip;
        pr->A[pr->nd] = NULL;
        pr->nd++;
    }
    if (n_rv > 0){
        if (n_sets < 1 || gmp_data_init_rv(&pr->d[pr->nd], n_rv, t_rv, rv, rv_err, rv_set, n_sets)) { gmp_problem_free(pr); return 1; }
        pr->A[pr->nd] = NULL;
        pr->nd++;
    }
    if (pr->nd == 0) return 1;
    for (int k = 0; k < pr->nd; k++){
        gmp_data *d = &pr->d[k];
        pr->A[k] = (double*)malloc((size_t)(d->n_base + d->n_pp*n_pl)*d->n*sizeof(double));
        if (pr->A[k] == NULL) { gmp_problem_free(pr); return 1; }
        gmp_fill_base_columns(d, pr->A[k]);
    }
    return 0;
}

static void gmp_problem_fill_planet(gmp_problem *pr, int i, double P, double phi, double e){
    for (int k = 0; k < pr->nd; k++)
        if (!gmp_skipped(&pr->d[k], i)) gmp_fill_planet_columns(&pr->d[k], pr->A[k], gmp_col(&pr->d[k], i), P, phi, e);
}

static void gmp_problem_fill_circular(gmp_problem *pr, int i, double P){
    for (int k = 0; k < pr->nd; k++)
        if (!gmp_skipped(&pr->d[k], i)) gmp_fill_circular_columns(&pr->d[k], pr->A[k], gmp_col(&pr->d[k], i), P);
}

/* the new planet of a search (index i) is never skipped */
static void gmp_problem_unskip(gmp_problem *pr, int i){
    for (int k = 0; k < pr->nd; k++) pr->d[k].skip &= ~(1 << i);
}

/* Solve the linear parameters of each dataset for n_pl planets. parts (if not NULL) receives the astrometric chi2,
the parallax-prior term and the RV chi2; worst (if not NULL) the index of the rejected astrometric epoch (-1 if none);
mu_ast and mu_rv (if not NULL) the linear parameters. Returns the sum of the chi2 and the prior term. */
static double gmp_problem_solve(gmp_problem *pr, int n_pl, double *parts, int *worst, double *mu_ast, double *mu_rv){
    double total = 0.0, mu[GMP_MAX_LIN];
    if (parts != NULL) parts[0] = parts[1] = parts[2] = 0.0;
    if (worst != NULL) *worst = -1;
    for (int k = 0; k < pr->nd; k++){
        gmp_data *d = &pr->d[k];
        int p = gmp_col(d, n_pl), w;
        double prior;
        double chi2 = gmp_linear_solve(d, p, pr->A[k], mu, &prior, &w);
        total += chi2 + prior;
        if (d->type == GMP_AST){
            if (parts != NULL) { parts[0] = chi2; parts[1] = prior; }
            if (worst != NULL) *worst = w;
            if (mu_ast != NULL){ /* 5 + 4 n_pl values, with zeros for skipped planets */
                memcpy(mu_ast, mu, d->n_base*sizeof(double));
                for (int i = 0; i < n_pl; i++)
                    for (int c = 0; c < d->n_pp; c++)
                        mu_ast[d->n_base + d->n_pp*i + c] = gmp_skipped(d, i) ? 0.0 : mu[gmp_col(d, i) + c];
            }
        } else {
            if (parts != NULL) parts[2] = chi2;
            if (mu_rv != NULL) memcpy(mu_rv, mu, p*sizeof(double));
        }
    }
    return total;
}

/* ---------------------------------------------------------------- exported: chi2 at fixed nonlinear parameters */

/* out[0] = astrometric chi2, out[1] = parallax-prior term, out[2] = RV chi2, out[3] = index of the rejected
astrometric epoch (-1 if none). mu_ast (length 5 + 4*n_pl) receives ra_off, pmra, dec_off, pmdec, parallax, then
(B, G, A, F) for each planet; mu_rv (length n_sets + 2*n_pl) the systemic velocities, then (K cos(w), K sin(w)) for
each planet. Unused outputs are left untouched (pass any array). */
void gmp_chi2(GMP_DATA_ARGS, int n_pl, const double *P, const double *phi, const double *e, double *out, double *mu_ast, double *mu_rv){
    out[0] = out[1] = out[2] = NAN; out[3] = -1;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_pl)) return;
    for (int i = 0; i < n_pl; i++) gmp_problem_fill_planet(&pr, i, P[i], phi[i], e[i]);
    int worst;
    gmp_problem_solve(&pr, n_pl, out, &worst, mu_ast, mu_rv);
    out[3] = worst;
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- exported: marginal likelihood terms */

/* For fixed nonlinear parameters, the terms of the likelihood marginalized over the linear parameters of each dataset
(with flat priors): out[0] = astrometric chi2 at the best linear parameters, out[1] = ln det of the astrometric normal
matrix, out[2] = index of the rejected astrometric epoch (-1 if none), out[3] = RV chi2, out[4] = ln det of the RV
normal matrix (NAN for a dataset that is not given, or INFINITY for the chi2 if a normal matrix is singular).
mu_ast (5 + 4 n_pl) and mu_rv (n_sets + 2 n_pl) receive the best linear parameters, in the order of gmp_chi2() (planets
skipped with ast_skip have no astrometric entries); if want_cov, cov_ast and cov_rv receive their covariance matrices
(row-major). Unused outputs are left untouched (pass any array). */
void gmp_marginal(GMP_DATA_ARGS, int n_pl, const double *P, const double *phi, const double *e, int want_cov,
                  double *out, double *mu_ast, double *cov_ast, double *mu_rv, double *cov_rv){
    out[0] = out[3] = INFINITY; out[1] = out[4] = NAN; out[2] = -1;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_pl)) return;
    for (int i = 0; i < n_pl; i++) gmp_problem_fill_planet(&pr, i, P[i], phi[i], e[i]);
    for (int k = 0; k < pr.nd; k++){
        gmp_data *d = &pr.d[k];
        int p = gmp_col(d, n_pl), worst;
        double logdet = NAN;
        int ast = d->type == GMP_AST;
        double chi2 = gmp_linear_solve_cov(d, p, pr.A[k], ast ? mu_ast : mu_rv, &logdet, want_cov ? (ast ? cov_ast : cov_rv) : NULL, &worst);
        if (ast) { out[0] = chi2; out[1] = logdet; out[2] = worst; }
        else { out[3] = chi2; out[4] = logdet; }
    }
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- exported: conditional periodogram */

/* For each trial period, the objective (chi2 + parallax prior) of a model with the n_fixed planets (at fixed P, phi, e)
plus one circular orbit at that period, with all linear parameters re-solved. */
void gmp_periodogram(GMP_DATA_ARGS, int n_fixed, const double *P, const double *phi, const double *e,
                     int n_trial, const double *trial_P, double *chi2_out){
    for (int k = 0; k < n_trial; k++) chi2_out[k] = NAN;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_fixed + 1)) return;
    gmp_problem_unskip(&pr, n_fixed);
    for (int i = 0; i < n_fixed; i++) gmp_problem_fill_planet(&pr, i, P[i], phi[i], e[i]);
    for (int k = 0; k < n_trial; k++){
        gmp_problem_fill_circular(&pr, n_fixed, trial_P[k]);
        chi2_out[k] = gmp_problem_solve(&pr, n_fixed + 1, NULL, NULL, NULL, NULL);
    }
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- exported: eccentricity/phase scan for a new planet */

/* The objective (chi2 + parallax prior) for one new planet on a grid of candidate periods, eccentricities and phases
phi_p = 2 pi k/n_phase, with the n_fixed planets held at fixed (P, phi, e). chi2_out has shape (n_cand, n_ecc, n_phase). */
void gmp_scan(GMP_DATA_ARGS, int n_fixed, const double *P, const double *phi, const double *e,
              int n_cand, const double *cand_P, int n_ecc, const double *ecc_grid, int n_phase, double *chi2_out){
    for (int k = 0; k < n_cand*n_ecc*n_phase; k++) chi2_out[k] = NAN;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_fixed + 1)) return;
    gmp_problem_unskip(&pr, n_fixed);
    for (int i = 0; i < n_fixed; i++) gmp_problem_fill_planet(&pr, i, P[i], phi[i], e[i]);
    for (int ic = 0; ic < n_cand; ic++){
        for (int ie = 0; ie < n_ecc; ie++){
            for (int ip = 0; ip < n_phase; ip++){
                gmp_problem_fill_planet(&pr, n_fixed, cand_P[ic], 2*PI*ip/n_phase, ecc_grid[ie]);
                chi2_out[(ic*n_ecc + ie)*n_phase + ip] = gmp_problem_solve(&pr, n_fixed + 1, NULL, NULL, NULL, NULL);
            }
        }
    }
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- Nelder-Mead polish of the nonlinear parameters */

static double gmp_log_beta(double x, double a, double b){
    if (x <= 0.0 || x >= 1.0) return -INFINITY;
    return (a - 1.0)*log(x) + (b - 1.0)*log1p(-x) - (lgamma(a) + lgamma(b) - lgamma(a + b));
}

typedef struct {
    gmp_problem *pr;
    int n_pl, n_free;
    int free_idx[GMP_MAX_PLANETS];
    double logP[GMP_MAX_PLANETS], phi[GMP_MAX_PLANETS], e[GMP_MAX_PLANETS];
    double L[3], U[3]; /* bounds on (P, phi, e), as in gaiamock */
    int smooth;        /* coordinates: 0 = (log P, phi, e) as in gaiamock; 1 = (log P, e cos(phi), e sin(phi)) */
    double ecc_a, ecc_b;
    double last_parts[3], last_ecc_prior; /* astrometric chi2, parallax-prior term, RV chi2; eccentricity-prior term */
} gmp_obj;

/* The optimizer works in one of two coordinate systems for each free planet:
 - (log P, phi, e), as in gaiamock's Nelder-Mead polish (smooth = 0): a vertex is kept inside the bounds
   L <= (P, phi, e) <= U by clamping each coordinate. This is used in gaiamock's search for each new planet.
 - (log P, e cos(phi), e sin(phi)) (smooth = 1), used for the joint polish of several planets. These coordinates are
   smooth through e = 0 (where phi is undefined) and periodic in phi, so a polish cannot get stuck at e = 0 or at
   the phi = 0 / 2 pi boundary. log P is clamped, and e is kept within [L[2], U[2]] by rescaling radially. */
static void gmp_constrain(const gmp_obj *o, double *x){
    double log_Pmin = log(o->L[0]), log_Pmax = log(o->U[0]);
    for (int k = 0; k < o->n_free; k++){
        double *v = x + 3*k;
        if (v[0] < log_Pmin) v[0] = log_Pmin;
        if (v[0] > log_Pmax) v[0] = log_Pmax;
        if (o->smooth){
            double ecc = hypot(v[1], v[2]);
            if (ecc > o->U[2]) { v[1] *= o->U[2]/ecc; v[2] *= o->U[2]/ecc; }
            else if (ecc < o->L[2]){
                if (ecc > 0) { v[1] *= o->L[2]/ecc; v[2] *= o->L[2]/ecc; }
                else v[1] = o->L[2];
            }
        } else {
            for (int c = 1; c < 3; c++){
                if (v[c] < o->L[c]) v[c] = o->L[c];
                if (v[c] > o->U[c]) v[c] = o->U[c];
            }
        }
    }
}

static void gmp_x_to_planet(const gmp_obj *o, const double *v, double *logP, double *phi, double *e){
    *logP = v[0];
    if (o->smooth){
        *e = hypot(v[1], v[2]);
        *phi = atan2(v[2], v[1]);
        if (*phi < 0) *phi += 2*PI;
    } else {
        *phi = v[1]; *e = v[2];
    }
}

/* objective = chi2 of all datasets + parallax-prior term - 2 sum ln Beta(e_i) (if an eccentricity prior is used) */
static double gmp_objective(gmp_obj *o, const double *x){
    for (int k = 0; k < o->n_free; k++){
        int i = o->free_idx[k];
        gmp_x_to_planet(o, x + 3*k, &o->logP[i], &o->phi[i], &o->e[i]);
        gmp_problem_fill_planet(o->pr, i, exp(o->logP[i]), o->phi[i], o->e[i]);
    }
    double ecc_prior = 0.0;
    if (o->ecc_a > 0 && o->ecc_b > 0){
        for (int i = 0; i < o->n_pl; i++) ecc_prior += -2.0*gmp_log_beta(o->e[i], o->ecc_a, o->ecc_b);
        if (!isfinite(ecc_prior)) return INFINITY;
    }
    double f = gmp_problem_solve(o->pr, o->n_pl, o->last_parts, NULL, NULL, NULL) + ecc_prior;
    o->last_ecc_prior = ecc_prior;
    return isfinite(f) ? f : INFINITY;
}

/* Nelder-Mead minimization of the objective over x (length n = 3*n_free), started from a simplex with the given
step sizes. Returns the best objective value and leaves the best vertex in x. */
static double gmp_nelder_mead(gmp_obj *o, double *x, int n, const double *step, int max_iter, double ftol, double xtol){
    int m = n + 1;
    double *S = (double*)malloc((size_t)m*n*sizeof(double));
    double *f = (double*)malloc((size_t)m*sizeof(double));
    double *cen = (double*)malloc((size_t)n*sizeof(double));
    double *xr = (double*)malloc((size_t)n*sizeof(double));
    double *xe = (double*)malloc((size_t)n*sizeof(double));
    double *xc = (double*)malloc((size_t)n*sizeof(double));
    double *tmp = (double*)malloc((size_t)n*sizeof(double));

    gmp_constrain(o, x);
    for (int i = 0; i < m; i++){
        memcpy(S + i*n, x, n*sizeof(double));
        if (i > 0){
            S[i*n + i - 1] += step[i - 1];
            gmp_constrain(o, S + i*n);
            if (S[i*n + i - 1] == x[i - 1]) { /* stuck at a bound: step the other way */
                S[i*n + i - 1] -= step[i - 1];
                gmp_constrain(o, S + i*n);
            }
        }
        f[i] = gmp_objective(o, S + i*n);
    }

    for (int iter = 0; iter < max_iter; iter++){
        /* sort vertices by objective (insertion sort) */
        for (int i = 1; i < m; i++){
            double fi = f[i];
            memcpy(tmp, S + i*n, n*sizeof(double));
            int k = i - 1;
            while (k >= 0 && f[k] > fi){
                f[k + 1] = f[k];
                memcpy(S + (k + 1)*n, S + k*n, n*sizeof(double));
                k--;
            }
            f[k + 1] = fi;
            memcpy(S + (k + 1)*n, tmp, n*sizeof(double));
        }
        double fspread = fabs(f[m - 1] - f[0]);
        double xspread = 0.0;
        for (int i = 1; i < m; i++) for (int k = 0; k < n; k++){
            double dx = fabs(S[i*n + k] - S[k]);
            if (dx > xspread) xspread = dx;
        }
        if (isfinite(f[0]) && fspread < ftol && xspread < xtol) break;

        for (int k = 0; k < n; k++){
            cen[k] = 0.0;
            for (int i = 0; i < m - 1; i++) cen[k] += S[i*n + k];
            cen[k] /= (m - 1);
        }
        double *worst = S + (m - 1)*n;
        for (int k = 0; k < n; k++) xr[k] = cen[k] + (cen[k] - worst[k]);
        gmp_constrain(o, xr);
        double fr = gmp_objective(o, xr);
        if (fr < f[0]){
            for (int k = 0; k < n; k++) xe[k] = cen[k] + 2.0*(xr[k] - cen[k]);
            gmp_constrain(o, xe);
            double fe = gmp_objective(o, xe);
            if (fe < fr) { memcpy(worst, xe, n*sizeof(double)); f[m - 1] = fe; }
            else { memcpy(worst, xr, n*sizeof(double)); f[m - 1] = fr; }
        } else if (fr < f[m - 2]){
            memcpy(worst, xr, n*sizeof(double)); f[m - 1] = fr;
        } else {
            if (fr < f[m - 1]) { for (int k = 0; k < n; k++) xc[k] = cen[k] + 0.5*(xr[k] - cen[k]); }
            else { for (int k = 0; k < n; k++) xc[k] = cen[k] + 0.5*(worst[k] - cen[k]); }
            gmp_constrain(o, xc);
            double fc = gmp_objective(o, xc);
            if (fc < f[m - 1]) { memcpy(worst, xc, n*sizeof(double)); f[m - 1] = fc; }
            else {
                for (int i = 1; i < m; i++){
                    for (int k = 0; k < n; k++) S[i*n + k] = S[k] + 0.5*(S[i*n + k] - S[k]);
                    gmp_constrain(o, S + i*n);
                    f[i] = gmp_objective(o, S + i*n);
                }
            }
        }
    }
    int best = 0;
    for (int i = 1; i < m; i++) if (f[i] < f[best]) best = i;
    memcpy(x, S + best*n, n*sizeof(double));
    double fbest = f[best];
    free(S); free(f); free(cen); free(xr); free(xe); free(xc); free(tmp);
    return fbest;
}

/* Set up the objective for n_pl planets of an initialized problem, with the planets flagged in free_flags free and the
others fixed (their columns are filled once). */
static void gmp_obj_init(gmp_obj *o, gmp_problem *pr, int n_pl, const double *P, const double *phi, const double *e, const int *free_flags,
                         const double *L, const double *U, double ecc_a, double ecc_b, int smooth){
    o->pr = pr; o->n_pl = n_pl; o->n_free = 0; o->smooth = smooth;
    for (int c = 0; c < 3; c++) { o->L[c] = L[c]; o->U[c] = U[c]; }
    o->ecc_a = ecc_a; o->ecc_b = ecc_b;
    for (int i = 0; i < n_pl; i++){
        o->logP[i] = log(P[i]); o->phi[i] = phi[i]; o->e[i] = e[i];
        if (free_flags[i]) o->free_idx[o->n_free++] = i;
        else gmp_problem_fill_planet(pr, i, P[i], phi[i], e[i]);
    }
}

/* Run Nelder-Mead on the free planets from their current (P, phi, e), with gaiamock's initial step sizes
(0.05 in log P, 0.25 in phi, 0.08 in e; 0.08 in e cos(phi) and e sin(phi) for smooth coordinates); repeat from the
best vertex (with halved steps) up to max_restarts times while the objective improves by more than 10*ftol.
Returns the objective at the result, stored in o. */
static double gmp_obj_polish(gmp_obj *o, int max_iter, double ftol, double xtol, int max_restarts){
    int n = 3*o->n_free;
    double x[3*GMP_MAX_PLANETS], step[3*GMP_MAX_PLANETS];
    for (int k = 0; k < o->n_free; k++){
        int i = o->free_idx[k];
        x[3*k] = o->logP[i];
        if (o->smooth){
            x[3*k + 1] = o->e[i]*cos(o->phi[i]); x[3*k + 2] = o->e[i]*sin(o->phi[i]);
            step[3*k] = 0.05; step[3*k + 1] = 0.08; step[3*k + 2] = 0.08;
        } else {
            x[3*k + 1] = o->phi[i]; x[3*k + 2] = o->e[i];
            step[3*k] = 0.05; step[3*k + 1] = 0.25; step[3*k + 2] = 0.08;
        }
    }
    double fbest = INFINITY;
    if (n > 0){
        for (int restart = 0; restart <= max_restarts; restart++){
            double fnew = gmp_nelder_mead(o, x, n, step, max_iter, ftol, xtol);
            int done = !(fnew < fbest - 10*ftol);
            if (fnew < fbest) fbest = fnew;
            for (int k = 0; k < n; k++) step[k] *= 0.5;
            if (done) break;
        }
    }
    return gmp_objective(o, x); /* also stores the parameters and chi2 components of the result in o */
}

/* Polish the nonlinear parameters (P, phi, e) of the planets flagged in free_flags, holding the others fixed, by
Nelder-Mead within the bounds L <= (P, phi, e) <= U, in (log P, phi, e) as in gaiamock (smooth = 0) or in
(log P, e cos(phi), e sin(phi)) (smooth = 1; see gmp_constrain), restarted up to max_restarts times. Each run
stops when the objective and coordinates of the simplex vertices agree to within ftol and xtol, or after max_iter
iterations. P, phi and e are updated in place (fixed planets are returned untouched).
out[0] = objective, out[1] = astrometric chi2, out[2] = parallax-prior term, out[3] = eccentricity-prior term,
out[4] = RV chi2. */
void gmp_polish(GMP_DATA_ARGS, int n_pl, double *P, double *phi, double *e, const int *free_flags, const double *L, const double *U,
                double ecc_a, double ecc_b, int max_iter, double ftol, double xtol, int max_restarts, int smooth, double *out){
    out[0] = out[1] = out[2] = out[3] = out[4] = NAN;
    if (n_pl < 1) return;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_pl)) return;
    gmp_obj o;
    gmp_obj_init(&o, &pr, n_pl, P, phi, e, free_flags, L, U, ecc_a, ecc_b, smooth);
    out[0] = gmp_obj_polish(&o, max_iter, ftol, xtol, max_restarts);
    out[1] = o.last_parts[0]; out[2] = o.last_parts[1]; out[3] = o.last_ecc_prior; out[4] = o.last_parts[2];
    for (int k = 0; k < o.n_free; k++){
        int i = o.free_idx[k];
        P[i] = exp(o.logP[i]); phi[i] = o.phi[i]; e[i] = o.e[i];
    }
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- exported: gaiamock's multistart search for one more planet */

typedef struct { double chi2, P, phi, e; } gmp_cand;

static int gmp_cand_compare(const void *a, const void *b){
    double da = ((const gmp_cand*)a)->chi2, db = ((const gmp_cand*)b)->chi2;
    return (da > db) - (da < db);
}

/* keep the max_n best candidates, sorted by chi2 (as gaiamock's insert_astfit_candidate) */
static void gmp_insert_candidate(gmp_cand *c, int *n, int max_n, double chi2, double P, double phi, double e){
    if (!isfinite(chi2)) return;
    if (*n >= max_n && chi2 >= c[*n - 1].chi2) return;
    int pos = *n;
    if (pos < max_n) *n += 1; else pos = max_n - 1;
    while (pos > 0 && chi2 < c[pos - 1].chi2) { c[pos] = c[pos - 1]; pos--; }
    c[pos].chi2 = chi2; c[pos].P = P; c[pos].phi = phi; c[pos].e = e;
}

static int gmp_period_is_distinct(double P, const double *periods, int n, double log_separation){
    for (int i = 0; i < n; i++) if (fabs(log(P/periods[i])) < log_separation) return 0;
    return 1;
}

static double gmp_phase_distance(double a, double b){
    double d = fabs(a - b);
    while (d > 2.0*PI) d -= 2.0*PI;
    if (d > PI) d = 2.0*PI - d;
    return d;
}

static int gmp_start_is_distinct(gmp_cand c, const gmp_cand *starts, int n){
    for (int i = 0; i < n; i++){
        if (fabs(log(c.P/starts[i].P)) < 0.003 && gmp_phase_distance(c.phi, starts[i].phi) < 0.05 && fabs(c.e - starts[i].e) < 0.05) return 0;
    }
    return 1;
}

/* Search for one more planet, given n_fixed planets whose (P, phi, e) are held fixed. This is gaiamock's
"periodogram_multistart" optimizer (run_astfit_periodogram_multistart_config in kepler_solve_astrometry.c), or, if
n_circ_phase > 0, its "multistart" optimizer (run_astfit_grid_multistart), with the fixed planets included in every
chi2 evaluation (and all linear parameters re-solved), and with the objective of the problem (astrometry, RVs, or
both); for astrometry with n_fixed = 0 it is the same algorithm:
  1. a circular-orbit periodogram on log-uniform period grids (n_period periods between L[0] and U[0], plus
     n_short_period periods between L[0] and min(U[0], 300 d)); with n_circ_phase > 0, each period is instead scored
     by the best chi2 of circular orbits at n_circ_phase phases (as gaiamock's scan_circular_period_grid);
  2. the best periods that are distinct by 2.5% (up to max_periods_keep), plus 8 long periods U[0] exp(-i/7);
  3. candidates: the 128 best periodogram rows (as circular orbits) and an eccentricity/phase scan of the kept
     periods (e = 0, 0.2, 0.4, 0.6, 0.8, 0.95 within the bounds; n_ecc_phase phases), keeping the best 256;
  4. Nelder-Mead polish of the new planet from the best max_starts distinct candidates, with gaiamock's default
     tolerances (full_nm = 0: 250 iterations, 1e-4, 3e-6) or those of its "full" optimizer (full_nm = 1: 400, 1e-6, 1e-7).
The objective is the chi2 + parallax-prior term (+ the eccentricity-prior term in the polish).
out (4 x max_starts) receives (P, phi, e, objective) of every polished start, sorted by objective; *n_out their number. */
void gmp_multistart(GMP_DATA_ARGS, int n_fixed, const double *P, const double *phi, const double *e, const double *L, const double *U,
                    double ecc_a, double ecc_b, int n_period, int n_short_period, int n_ecc_phase, int max_periods_keep, int max_starts,
                    int full_nm, int n_circ_phase, double *out, int *n_out){
    *n_out = 0;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_fixed + 1)) return;
    gmp_problem_unskip(&pr, n_fixed);
    int n_pl = n_fixed + 1;
    int max_periods = 32, max_candidates = 256;
    double ecc_grid[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 0.95};
    double short_period_max = fmin(U[0], 300.0);
    int max_rows = n_period + ((short_period_max > L[0]) ? n_short_period : 0);

    gmp_cand *rows = (gmp_cand*)malloc((size_t)max_rows*sizeof(gmp_cand));
    gmp_cand *cand = (gmp_cand*)malloc((size_t)max_candidates*sizeof(gmp_cand));
    gmp_cand *starts = (gmp_cand*)malloc((size_t)max_starts*sizeof(gmp_cand));
    double periods[32];
    if (rows == NULL || cand == NULL || starts == NULL) goto cleanup;
    for (int i = 0; i < n_fixed; i++) gmp_problem_fill_planet(&pr, i, P[i], phi[i], e[i]);

    /* 1. circular-orbit periodogram */
    int n_rows = 0;
    for (int grid = 0; grid < 2; grid++){
        int n_grid = grid == 0 ? n_period : n_short_period;
        double Pmax = grid == 0 ? U[0] : short_period_max;
        if (grid == 1 && !(short_period_max > L[0])) break;
        for (int ip = 0; ip < n_grid; ip++){
            double frac = (n_grid == 1) ? 0.0 : ((double)ip/(double)(n_grid - 1));
            double Ptrial = exp(log(L[0]) + frac*(log(Pmax) - log(L[0])));
            double chi2 = INFINITY, phi_row = (n_circ_phase > 0) ? L[1] : 0.0;
            if (n_circ_phase > 0){
                for (int iph = 0; iph < n_circ_phase; iph++){
                    double ph = L[1] + ((double)iph/(double)n_circ_phase)*(U[1] - L[1]);
                    gmp_problem_fill_planet(&pr, n_fixed, Ptrial, ph, 0.0);
                    double c = gmp_problem_solve(&pr, n_pl, NULL, NULL, NULL, NULL);
                    if (c < chi2) { chi2 = c; phi_row = ph; }
                }
            } else {
                gmp_problem_fill_circular(&pr, n_fixed, Ptrial);
                chi2 = gmp_problem_solve(&pr, n_pl, NULL, NULL, NULL, NULL);
            }
            rows[n_rows].chi2 = isfinite(chi2) ? chi2 : INFINITY;
            rows[n_rows].P = Ptrial; rows[n_rows].phi = phi_row; rows[n_rows].e = 0.0;
            n_rows++;
        }
    }
    qsort(rows, n_rows, sizeof(gmp_cand), gmp_cand_compare);

    /* 2. distinct periods to scan */
    int n_periods = 0;
    for (int i = 0; i < n_rows && n_periods < max_periods_keep; i++){
        if (gmp_period_is_distinct(rows[i].P, periods, n_periods, 0.025)) periods[n_periods++] = rows[i].P;
    }
    for (int i = 0; i < 8 && n_periods < max_periods; i++){
        double Plong = U[0]*exp(-((double)i/7.0));
        if (Plong >= L[0] && Plong <= U[0] && gmp_period_is_distinct(Plong, periods, n_periods, 0.01)) periods[n_periods++] = Plong;
    }

    /* 3. candidates: best periodogram rows and an eccentricity/phase scan of the kept periods */
    int n_cand = 0;
    for (int i = 0; i < n_rows && i < 128; i++) gmp_insert_candidate(cand, &n_cand, max_candidates, rows[i].chi2, rows[i].P, rows[i].phi, rows[i].e);
    for (int ip = 0; ip < n_periods; ip++){
        for (int ie = 0; ie < 6; ie++){
            if (ecc_grid[ie] < L[2] || ecc_grid[ie] > U[2]) continue;
            for (int iph = 0; iph < n_ecc_phase; iph++){
                double ph = L[1] + ((double)iph/(double)n_ecc_phase)*(U[1] - L[1]);
                gmp_problem_fill_planet(&pr, n_fixed, periods[ip], ph, ecc_grid[ie]);
                double chi2 = gmp_problem_solve(&pr, n_pl, NULL, NULL, NULL, NULL);
                gmp_insert_candidate(cand, &n_cand, max_candidates, chi2, periods[ip], ph, ecc_grid[ie]);
            }
        }
    }

    /* 4. polish the new planet from the best distinct starts */
    int n_starts = 0;
    for (int i = 0; i < n_cand && n_starts < max_starts; i++){
        if (gmp_start_is_distinct(cand[i], starts, n_starts)) starts[n_starts++] = cand[i];
    }
    double Pall[GMP_MAX_PLANETS], phiall[GMP_MAX_PLANETS], eall[GMP_MAX_PLANETS];
    int free_flags[GMP_MAX_PLANETS];
    for (int i = 0; i < n_fixed; i++) { Pall[i] = P[i]; phiall[i] = phi[i]; eall[i] = e[i]; free_flags[i] = 0; }
    free_flags[n_fixed] = 1;
    gmp_cand *results = cand; /* reuse the candidate buffer for the polished results */
    int n_res = 0;
    for (int k = 0; k < n_starts; k++){
        Pall[n_fixed] = starts[k].P; phiall[n_fixed] = starts[k].phi; eall[n_fixed] = starts[k].e;
        gmp_obj o;
        gmp_obj_init(&o, &pr, n_pl, Pall, phiall, eall, free_flags, L, U, ecc_a, ecc_b, 0); /* gaiamock's coordinates */
        double f = full_nm ? gmp_obj_polish(&o, 400, 1e-6, 1e-7, 0) : gmp_obj_polish(&o, 250, 1e-4, 3e-6, 0);
        results[n_res].chi2 = isfinite(f) ? f : INFINITY;
        results[n_res].P = exp(o.logP[n_fixed]); results[n_res].phi = o.phi[n_fixed]; results[n_res].e = o.e[n_fixed];
        n_res++;
    }
    qsort(results, n_res, sizeof(gmp_cand), gmp_cand_compare);
    for (int k = 0; k < n_res; k++){
        out[4*k] = results[k].P; out[4*k + 1] = results[k].phi; out[4*k + 2] = results[k].e; out[4*k + 3] = results[k].chi2;
    }
    *n_out = n_res;

cleanup:
    free(rows); free(cand); free(starts);
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- exported: gaiamock's simulated annealing for one more planet */

static double gmp_random_double(void){
    return (double)rand() / (double)RAND_MAX; /* as gaiamock's random_double() */
}

/* as gaiamock's generate_xnew_from_generation_temperatures() */
static void gmp_generate_xnew(const double *x, const double *L, const double *U, const double *Tgen, int len, double *xnew){
    for (int i = 0; i < len; i++) xnew[i] = x[i];
    for (int g = 0; g < len; g++){
        while (1){
            double r = gmp_random_double();
            int sgn = (r - 0.5) >= 0 ? 1 : -1;
            double q = sgn*Tgen[g]*(pow((1.0 + 1.0/Tgen[g]), fabs(2*r - 1)) - 1);
            xnew[g] = x[g] + q*(U[g] - L[g]);
            if (L[g] <= xnew[g] && xnew[g] <= U[g]) break;
        }
    }
}

/* the objective at (P, phi, e) of the new planet (linear coordinates, as in gaiamock's annealing) */
static double gmp_anneal_objective(gmp_obj *o, const double *xl){
    double x[3] = {log(xl[0]), xl[1], xl[2]};
    return gmp_objective(o, x);
}

/* Search for one more planet, given n_fixed planets whose (P, phi, e) are held fixed, with gaiamock's "annealing"
optimizer: adaptive simulated annealing of (P, phi, e) within L <= (P, phi, e) <= U, started at (L + U)/2, with the
tunable parameters of RVFIT (Iglesias-Marzoa et al. 2015). This is a line-by-line port of run_astfit() in
kepler_solve_astrometry.c (using rand() in the same order, so that for astrometry with n_fixed = 0 and the same
srand() seed it follows the same path), with the objective of the problem (chi2 of all datasets + prior terms). The
reheating threshold chislimit = sum of (mean err/err)^2 is computed for each dataset (astrometry, RVs) and summed.
If seed > 0, srand(seed) is called first. out = (P, phi, e, objective) of the best point. */
void gmp_anneal(GMP_DATA_ARGS, int n_fixed, const double *P, const double *phi, const double *e, const double *L, const double *U,
                double ecc_a, double ecc_b, unsigned int seed, double *out){
    out[0] = out[1] = out[2] = out[3] = NAN;
    gmp_problem pr;
    if (gmp_problem_init(&pr, GMP_DATA_PASS, n_fixed + 1)) return;
    gmp_problem_unskip(&pr, n_fixed);
    if (seed > 0) srand(seed);
    int n_pl = n_fixed + 1;
    double Pall[GMP_MAX_PLANETS], phiall[GMP_MAX_PLANETS], eall[GMP_MAX_PLANETS];
    int free_flags[GMP_MAX_PLANETS];
    for (int i = 0; i < n_fixed; i++) { Pall[i] = P[i]; phiall[i] = phi[i]; eall[i] = e[i]; free_flags[i] = 0; }
    Pall[n_fixed] = (L[0] + U[0])/2.0; phiall[n_fixed] = (L[1] + U[1])/2.0; eall[n_fixed] = (L[2] + U[2])/2.0;
    free_flags[n_fixed] = 1;
    gmp_obj o;
    gmp_obj_init(&o, &pr, n_pl, Pall, phiall, eall, free_flags, L, U, ecc_a, ecc_b, 0);

    double eps = 1e-5; /* all the tunable parameters are taken from RVFIT, as in gaiamock */
    int Neps = 5, Nterm = 20;
    enum { npar = 3 };
    double fbestlist[5];
    for (int i = 0; i < Neps; i++) fbestlist[i] = 1.0;
    int nrean = 0;

    double chislimit = 0.0;
    for (int k = 0; k < pr.nd; k++){
        const gmp_data *d = &pr.d[k];
        double mean_err = 0.0;
        for (int j = 0; j < d->n; j++) mean_err += 1.0/sqrt(d->w[j]);
        mean_err /= d->n;
        for (int j = 0; j < d->n; j++) chislimit += mean_err*mean_err*d->w[j];
    }

    double x[npar], xbest[npar];
    for (int i = 0; i < npar; i++) x[i] = (U[i] + L[i])/2.0;
    double f = gmp_anneal_objective(&o, x);
    double fbest = f;
    for (int i = 0; i < npar; i++) xbest[i] = x[i];

    int ka = 0;
    double Ta0 = f, Ta = Ta0;
    int nacep = 0;
    double kgen[npar] = {0.0, 0.0, 0.0}, Tgen0[npar], Tgen[npar], delta[npar];
    for (int i = 0; i < npar; i++) { Tgen0[i] = 1.0; Tgen[i] = Tgen0[i]; delta[i] = fabs(U[i] - L[i])*1e-8; }
    double c = 20.0;
    int Na = 1000, Ngen = 10000;
    double acep = 0.25;
    enum { ntest = 100 };
    double ftest[ntest];

    for (int j = 0; j < ntest; j++){
        double xnew[npar];
        gmp_generate_xnew(x, L, U, Tgen, npar, xnew);
        ftest[j] = gmp_anneal_objective(&o, xnew);
        if (ftest[j] < fbest){
            fbest = ftest[j];
            for (int i = 0; i < npar; i++) xbest[i] = xnew[i];
        }
    }
    double avdftest = 0.0;
    for (int i = 0; i < ntest - 1; i++) avdftest += fabs(ftest[i + 1] - ftest[i]);
    avdftest /= (ntest - 1);
    Ta0 = avdftest/log(1.0/acep - 1.0);

    int repeat = 1;
    while (repeat){
        for (int j = 0; j < Ngen; j++){
            int flag_acceptance = 0;
            double xnew[npar];
            gmp_generate_xnew(x, L, U, Tgen, npar, xnew);
            double fnew = gmp_anneal_objective(&o, xnew);
            if (fnew <= f){
                flag_acceptance = 1;
            } else {
                double test = (fnew - f)/Ta;
                double Pa = (test > 20) ? 0 : 1.0/(1.0 + exp(test));
                if (gmp_random_double() <= Pa) flag_acceptance = 1;
            }
            if (flag_acceptance){
                if (fnew < fbest){
                    fbest = fnew;
                    for (int i = 0; i < npar; i++) xbest[i] = xnew[i];
                    nrean = 0;
                }
                nacep++;
                ka++;
                for (int i = 0; i < npar; i++) x[i] = xnew[i];
                f = fnew;
            }
            if (nacep >= Na){
                double s[npar];
                for (int g = 0; g < npar; g++){
                    double xdelta[npar];
                    for (int i = 0; i < npar; i++) xdelta[i] = xbest[i];
                    xdelta[g] += delta[g];
                    s[g] = fabs((gmp_anneal_objective(&o, xdelta) - fbest)/delta[g]);
                }
                for (int i = 0; i < npar; i++){
                    if (s[i] == 0.0){
                        double min_nonzero = 1e10;
                        for (int k = 0; k < npar; k++) if (s[k] != 0.0 && s[k] < min_nonzero) min_nonzero = s[k];
                        s[i] = min_nonzero;
                    }
                }
                double smax = 0.0;
                for (int i = 0; i < npar; i++) if (s[i] > smax) smax = s[i];
                for (int i = 0; i < npar; i++){
                    Tgen[i] = Tgen[i]*(smax/s[i]);
                    kgen[i] = fabs(pow(log(Tgen0[i]/Tgen[i])/c, npar));
                }
                Ta0 = f;
                Ta = fbest;
                ka = pow(log(Ta0/Ta)/c, npar); /* truncated to an integer, as in gaiamock */
                int stop = 0;
                for (int i = 0; i < npar; i++) if (Tgen[i] == 0) { stop = 1; break; }
                if (stop) { repeat = 0; break; }
                nacep = 0;
                nrean++;
            }
        }
        for (int i = 0; i < npar; i++){
            kgen[i] += 1;
            Tgen[i] = Tgen0[i]*exp(-c*pow(kgen[i], 1.0/npar));
        }
        ka += 1;
        Ta = Ta0*exp(-c*pow(ka, 1.0/npar));
        for (int i = 1; i < Neps; i++) fbestlist[i - 1] = fbestlist[i];
        fbestlist[Neps - 1] = fbest;
        int count = 0;
        for (int i = 0; i < Neps - 1; i++) if (fabs(fbestlist[i + 1] - fbestlist[i]) < eps) count++;
        if (count == Neps - 1){
            if (fbest < chislimit) repeat = 0;
            else Ta = Ta0;
        }
        if (nrean >= Nterm) repeat = 0;
    }
    out[0] = xbest[0]; out[1] = xbest[1]; out[2] = xbest[2]; out[3] = fbest;
    gmp_problem_free(&pr);
}

/* ---------------------------------------------------------------- exported: residuals for the Jacobian */

/* Normalized residuals (model - obs)/err for the full parameter vector
theta = (ra_off, dec_off, parallax, pmra, pmdec, then for each planet: P, e, phi_p, A, B, F, G),
which for one planet is the same ordering as gaiamock's 12-parameter solution. */
void gmp_astrometry_residuals(int n_obs, const double *t, const double *psi, const double *plx, const double *obs, const double *err,
                              int n_pl, const double *theta, double *resid){
    for (int j = 0; j < n_obs; j++){
        double s = sin(psi[j]), c = cos(psi[j]);
        double ra = theta[0] + theta[3]*t[j], dec = theta[1] + theta[4]*t[j];
        for (int i = 0; i < n_pl; i++){
            const double *q = theta + 5 + 7*i;
            double E = gmp_solve_kepler(2*PI*t[j]*365.25/q[0] - q[2], q[1]);
            double X = cos(E) - q[1], Y = sqrt(1 - q[1]*q[1])*sin(E);
            ra += q[4]*X + q[6]*Y;  /* B X + G Y */
            dec += q[3]*X + q[5]*Y; /* A X + F Y */
        }
        resid[j] = (ra*s + dec*c + theta[2]*plx[j] - obs[j])/err[j];
    }
}

/* Normalized RV residuals (model - rv)/rv_err for the full parameter vector
theta = (gamma_0, ..., gamma_{n_sets - 1}, then for each planet: P, e, phi_p, K cos(w), K sin(w)),
where the model is gamma_{set} + sum over planets of K [cos(nu + w) + e cos(w)]. */
void gmp_rv_residuals(int n_rv, const double *t_rv, const double *rv, const double *rv_err, const int *rv_set, int n_sets,
                      int n_pl, const double *theta, double *resid){
    for (int j = 0; j < n_rv; j++){
        double model = theta[rv_set[j]];
        for (int i = 0; i < n_pl; i++){
            const double *q = theta + n_sets + 5*i;
            double E = gmp_solve_kepler(2*PI*t_rv[j]*365.25/q[0] - q[2], q[1]);
            double denom = 1 - q[1]*cos(E);
            double cos_nu = (cos(E) - q[1])/denom, sin_nu = sqrt(1 - q[1]*q[1])*sin(E)/denom;
            model += q[3]*(cos_nu + q[1]) - q[4]*sin_nu;
        }
        resid[j] = (model - rv[j])/rv_err[j];
    }
}
