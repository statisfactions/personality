// Variant 2: tightened kappa AND lognormal loading prior.
// Changes vs base:
//   - kappa ~ N(0, 0.5)        (was N(0, 1.5))  — restrict cutpoint wandering
//   - a_pos ~ Lognormal(0, 0.5) (was HalfNormal(0, 0.5)) — mode at 1 instead
//     of 0, no shrinkage toward zero
//
// Rationale (W11, 2026-05-13): same as variant 1, plus removing the
// loading-shrinkage toward zero. Half-normal prior has mode at 0,
// actively pulling loadings small. Lognormal(0, 0.5) has mode at exp(-0.25)
// ≈ 0.78, median at 1, allows larger loadings without shrinkage.

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> J;
  int<lower=1> D;
  int<lower=1> K;

  array[J] int<lower=1, upper=D> trait;
  array[J] int<lower=-1, upper=1> g;
  array[P] int<lower=1, upper=J> L;
  array[P] int<lower=1, upper=J> R;
  array[N, P] int<lower=1, upper=K> y;
}

parameters {
  matrix[N, D] theta;
  vector<lower=0>[J] a_pos;
  array[P] ordered[K - 1] kappa;
}

transformed parameters {
  vector[J] a;
  for (j in 1:J) a[j] = g[j] * a_pos[j];
}

model {
  to_vector(theta) ~ std_normal();
  a_pos ~ lognormal(0, 0.5);                  // CHANGED: mode at 1, no zero-shrink
  for (p in 1:P) kappa[p] ~ normal(0, 0.5);   // TIGHTENED from N(0, 1.5)

  for (i in 1:N) {
    for (p in 1:P) {
      real mu_L = a[L[p]] * theta[i, trait[L[p]]];
      real mu_R = a[R[p]] * theta[i, trait[R[p]]];
      real eta  = (mu_R - mu_L) / sqrt2();
      y[i, p] ~ ordered_logistic(eta, kappa[p]);
    }
  }
}
