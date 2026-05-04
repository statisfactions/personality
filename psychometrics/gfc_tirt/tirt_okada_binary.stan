// Binary (true forced-choice) Thurstonian IRT, Okada Appendix D structure.
//
// Same generative model as tirt_okada_indep.stan EXCEPT the response is
// binary (which item describes you more, LEFT or RIGHT) instead of a 7-point
// graded rating. Drops the kappa thresholds and ordered_logistic likelihood;
// uses bernoulli_logit on the latent utility difference.
//
// y[i, p] = 1  ⇔  LEFT-as-shown chosen
// y[i, p] = 0  ⇔  RIGHT-as-shown chosen
//
// (The data preprocessing layer must already have unswapped responses to
// the canonical instrument-defined LEFT, so eta below uses canonical L/R.)
//
// eta_ip = (mu_L - mu_R) / sqrt(2),  P(y=1) = inv_logit(eta_ip)
//
// Priors: theta ~ N(0, I_5), a_pos ~ HalfNormal(0, 0.5), with signed
// loadings a[j] = g[j] * a_pos[j] (keying anchoring).

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> J;
  int<lower=1> D;

  array[J] int<lower=1, upper=D> trait;
  array[J] int<lower=-1, upper=1> g;
  array[P] int<lower=1, upper=J> L;
  array[P] int<lower=1, upper=J> R;
  array[N, P] int<lower=0, upper=1> y;
}

parameters {
  matrix[N, D] theta;
  vector<lower=0>[J] a_pos;
}

transformed parameters {
  vector[J] a;
  for (j in 1:J) a[j] = g[j] * a_pos[j];
}

model {
  to_vector(theta) ~ std_normal();
  a_pos ~ normal(0, 0.5);                     // half-normal via <lower=0>

  for (i in 1:N) {
    for (p in 1:P) {
      real mu_L = a[L[p]] * theta[i, trait[L[p]]];
      real mu_R = a[R[p]] * theta[i, trait[R[p]]];
      real eta  = (mu_L - mu_R) / sqrt2();
      y[i, p] ~ bernoulli_logit(eta);
    }
  }
}
