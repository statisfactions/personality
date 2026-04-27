// Ordinal Thurstonian IRT for graded forced-choice (GFC) data.
// Mirrors Okada et al. (2026) Section 3.3 GFC formulation:
//
//   mu_ij  = a_j * (q_j' theta_i)              statement utility
//   eta_ip = (mu_{i,R(p)} - mu_{i,L(p)}) / sqrt(2)
//   P(Y_ip >= k | theta_i) = inv_logit(eta_ip - kappa_{p,k-1})
//
// Identifiability: signed discrimination a_j = g_j * a_j+, with a_j+ > 0
// (lognormal prior). The sign is fixed by item keying g_j in {-1, +1},
// which anchors the latent direction of each trait and prevents the
// global sign-flip pathology seen in the off-the-shelf thurstonianIRT
// package. Trait correlation matrix Omega has an LKJ(2) prior; trait
// scales are fixed at 1 (Sigma = Omega).

data {
  int<lower=1> N;                 // n respondents
  int<lower=1> P;                 // n GFC pairs (blocks)
  int<lower=1> J;                 // n unique statements (= 2P)
  int<lower=1> D;                 // n latent traits (= 5 for Big Five)
  int<lower=1> K;                 // n response categories (= 7)

  // Per-statement metadata
  array[J] int<lower=1, upper=D> trait;  // trait index for each statement
  array[J] int<lower=-1, upper=1> g;     // keying sign (-1 or +1)

  // Per-pair structure: which statements are LEFT and RIGHT
  array[P] int<lower=1, upper=J> L;      // left  statement index
  array[P] int<lower=1, upper=J> R;      // right statement index

  // Responses: y[n, p] in 1..K
  array[N, P] int<lower=1, upper=K> y;
}

parameters {
  // Latent traits (z-form, transformed via Cholesky for correlated theta)
  matrix[D, N] z;
  cholesky_factor_corr[D] L_Omega;

  // Discrimination magnitudes (positive); signed by g_j in transformed params
  vector<lower=0>[J] a_pos;

  // Per-pair ordered thresholds
  array[P] ordered[K - 1] kappa;
}

transformed parameters {
  matrix[N, D] theta;             // latent traits, theta[i, d]
  vector[J] a;                    // signed discriminations

  theta = (L_Omega * z)';         // theta_i ~ MVN(0, L_Omega L_Omega')
  for (j in 1:J) {
    a[j] = g[j] * a_pos[j];
  }
}

model {
  // Priors
  to_vector(z) ~ std_normal();
  L_Omega ~ lkj_corr_cholesky(2);
  a_pos ~ lognormal(0, 0.5);
  for (p in 1:P) {
    kappa[p] ~ normal(0, 5);
  }

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

generated quantities {
  corr_matrix[D] Omega = multiply_lower_tri_self_transpose(L_Omega);
}
