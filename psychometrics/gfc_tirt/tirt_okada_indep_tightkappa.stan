// Variant 1: same as tirt_okada_indep.stan but with TIGHTER kappa prior.
// Changes vs base:
//   - kappa ~ N(0, 0.5)  (was N(0, 1.5))  — restrict cutpoint wandering
// a_pos prior UNCHANGED (HalfNormal(0, 0.5)).
//
// Rationale (W11, 2026-05-13): TIRT recovery on IPIP-NEO-GFC-60 was
// near-zero with sign flips despite per-pair preferences looking
// credible. Hypothesis: broad kappa prior allows cutpoints to absorb
// data variance that should flow through to theta. Tightening kappa
// forces more signal into theta.

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
  a_pos ~ normal(0, 0.5);                     // UNCHANGED
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
