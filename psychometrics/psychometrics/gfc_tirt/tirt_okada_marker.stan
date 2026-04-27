// Ordinal Thurstonian IRT, marker-item identification.
//
// Difference vs results/tirt_okada.stan: instead of forcing every item's
// loading sign by keying (a_j = g_j * a_pos with a_pos > 0), this version
// lets all loadings be free-signed and uses ONE positively-keyed marker
// item per trait to anchor the latent direction. The marker's loading is
// constrained > 0; all other items have signed loadings determined by the
// data.
//
// This is the standard Bayesian TIRT identifiability convention
// (Brown & Maydeu-Olivares; Bürkner). It tests whether Haiku's apparent
// {A, C} sign flip reflects (i) genuine counter-keying behavior in the
// model's response data (signs stay flipped under marker anchoring) or
// (ii) an artifact of forcing all loadings to obey the published keying
// (signs flip back to positive).

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> J;
  int<lower=1> D;
  int<lower=1> K;

  array[J] int<lower=1, upper=D> trait;
  array[J] int<lower=-1, upper=1> g;     // keying sign (used only as a prior)

  array[P] int<lower=1, upper=J> L;
  array[P] int<lower=1, upper=J> R;

  array[N, P] int<lower=1, upper=K> y;

  // Marker item index per trait: marker[d] is the J-index of the marker
  // statement for trait d. Its loading a[marker[d]] is constrained > 0.
  array[D] int<lower=1, upper=J> marker;
}

transformed data {
  // Identify which items are markers (1) vs free (0)
  array[J] int is_marker;
  for (j in 1:J) is_marker[j] = 0;
  for (d in 1:D) is_marker[marker[d]] = 1;
}

parameters {
  matrix[D, N] z;
  cholesky_factor_corr[D] L_Omega;

  // Marker items: positive magnitude
  vector<lower=0>[D] a_marker;

  // Free items: signed loadings (no sign constraint)
  vector[J - D] a_free;

  array[P] ordered[K - 1] kappa;
}

transformed parameters {
  matrix[N, D] theta;
  vector[J] a;

  theta = (L_Omega * z)';

  // Assemble loading vector: marker entries get a_marker, others get a_free
  {
    int free_idx = 1;
    for (j in 1:J) {
      if (is_marker[j] == 1) {
        // find which trait this marker belongs to
        for (d in 1:D) {
          if (marker[d] == j) {
            a[j] = a_marker[d];
          }
        }
      } else {
        a[j] = a_free[free_idx];
        free_idx += 1;
      }
    }
  }
}

model {
  // Priors
  to_vector(z) ~ std_normal();
  L_Omega ~ lkj_corr_cholesky(2);

  // Marker loadings: positive, lognormal-ish magnitude
  a_marker ~ lognormal(0, 0.5);

  // Free loadings: keying-informed prior. Mean = g_j (so positively-keyed
  // items have prior mean +1, negatively-keyed -1) but the data can
  // override the sign if it disagrees. SD wide enough to permit flipping.
  for (j in 1:J) {
    if (is_marker[j] == 0) {
      // figure out which entry of a_free this is
      int idx = 0;
      for (k in 1:j) if (is_marker[k] == 0) idx += 1;
      a_free[idx] ~ normal(g[j] * 1.0, 1.5);
    }
  }

  for (p in 1:P) {
    kappa[p] ~ normal(0, 5);
  }

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
