// Ordinal Thurstonian IRT, EXACT Okada Appendix D specification.
//
// Differences vs results/tirt_okada.stan:
//   - theta_i ~ N(0, I_5) (independent prior; NOT estimating latent
//     correlation matrix). This matches Appendix D's explicit choice:
//     "we do not estimate a latent correlation matrix among traits in
//     the IRT scoring step."
//   - kappa_p ~ N(0, 1.5) (Okada's prior; was N(0, 5) in tirt_okada.stan)
//   - a_j+ ~ HalfNormal(0, 0.5) (Okada's; was Lognormal(0, 0.5))
//
// All other choices unchanged: signed loadings a_j = g_j * a_j+ with
// keying anchoring, ordered thresholds, ordered_logistic likelihood with
// eta_ip = (mu_R - mu_L) / sqrt(2).

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
  // Independent latent traits (Okada Appendix D)
  matrix[N, D] theta;

  // Positive discrimination magnitudes
  vector<lower=0>[J] a_pos;

  // Per-pair ordered thresholds
  array[P] ordered[K - 1] kappa;
}

transformed parameters {
  vector[J] a;
  for (j in 1:J) a[j] = g[j] * a_pos[j];
}

model {
  // Priors (Okada Appendix D)
  to_vector(theta) ~ std_normal();
  a_pos ~ normal(0, 0.5);                     // half-normal via <lower=0>
  for (p in 1:P) kappa[p] ~ normal(0, 1.5);

  // Likelihood
  for (i in 1:N) {
    for (p in 1:P) {
      real mu_L = a[L[p]] * theta[i, trait[L[p]]];
      real mu_R = a[R[p]] * theta[i, trait[R[p]]];
      real eta  = (mu_R - mu_L) / sqrt2();
      y[i, p] ~ ordered_logistic(eta, kappa[p]);
    }
  }
}
