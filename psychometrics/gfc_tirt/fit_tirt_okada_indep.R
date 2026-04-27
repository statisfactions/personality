#!/usr/bin/env Rscript
# Fit the Okada-Appendix-D-exact TIRT (results/tirt_okada_indep.stan) to one
# model's GFC-30 responses.
#
# Differences vs fit_tirt_okada.R:
#   - Uses tirt_okada_indep.stan (independent theta prior, Okada priors)
#   - No Omega in output (model doesn't estimate it)
#
# Usage:
#   Rscript results/fit_tirt_okada_indep.R <responses.json> <output.rds> [n_personas] [iter] [chains]

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: fit_tirt_okada_indep.R <responses.json> <output.rds>")
responses_path <- args[1]
output_path    <- args[2]
n_personas     <- if (length(args) >= 3) as.integer(args[3]) else 50L
n_iter         <- if (length(args) >= 4) as.integer(args[4]) else 700L  # Okada: 200 warmup + 500 sampling
n_chains       <- if (length(args) >= 5) as.integer(args[5]) else 4L

stan_file <- "results/tirt_okada_indep.stan"
if (!file.exists(stan_file)) stop("Stan file missing: ", stan_file)

message("Stan model: ", stan_file)
message("Responses:  ", responses_path)
message("Output:     ", output_path)

inst <- fromJSON("instruments/okada_gfc30.json")
pairs <- inst$pairs
P <- nrow(pairs)
D <- 5L
K <- 7L
trait_names <- c("A", "C", "E", "N", "O")
trait_idx <- setNames(seq_along(trait_names), trait_names)

stmt_df <- tibble(
  block = rep(seq_len(P), each = 2),
  side  = rep(c("L", "R"), times = P),
  trait = c(rbind(pairs$left$trait,   pairs$right$trait)),
  key   = c(rbind(pairs$left$keying,  pairs$right$keying))
) %>%
  mutate(g = ifelse(key == "+", 1L, -1L),
         trait_id = trait_idx[trait])
stmt_df$stmt_index <- seq_len(nrow(stmt_df))
J <- nrow(stmt_df)

L_idx <- stmt_df$stmt_index[stmt_df$side == "L"]
R_idx <- stmt_df$stmt_index[stmt_df$side == "R"]

# Load responses
raw <- fromJSON(responses_path, flatten = TRUE)
results_df <- as_tibble(raw$results)
target_ids <- paste0("s", seq_len(n_personas))
results_df <- results_df %>%
  filter(persona_id %in% target_ids, !is.na(response_argmax)) %>%
  mutate(response = as.integer(response_argmax))

wide <- results_df %>%
  select(persona_id, block, response) %>%
  pivot_wider(names_from = block, values_from = response, names_prefix = "b") %>%
  filter(persona_id %in% target_ids) %>%
  arrange(match(persona_id, target_ids))

keep <- complete.cases(wide[, -1])
wide <- wide[keep, ]
N <- nrow(wide)
message("Complete respondents: ", N)

y_mat <- as.matrix(wide[, paste0("b", seq_len(P))])
mode(y_mat) <- "integer"

personas_json <- fromJSON("instruments/synthetic_personas.json")
gt <- tibble(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O
)
gt_aligned <- gt %>%
  filter(persona_id %in% wide$persona_id) %>%
  arrange(match(persona_id, wide$persona_id))

stan_data <- list(
  N = N, P = P, J = J, D = D, K = K,
  trait = stmt_df$trait_id,
  g     = stmt_df$g,
  L     = L_idx, R = R_idx,
  y     = y_mat
)

message("\nCompiling indep-prior model...")
model <- stan_model(stan_file)

message("Sampling: ", n_chains, " x ", n_iter, " (", n_iter %/% 3, " warmup)")
fit <- sampling(
  model, data = stan_data,
  chains = n_chains, iter = n_iter, warmup = n_iter %/% 3,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  refresh = 100,
  pars = c("theta", "a_pos", "kappa")
)

theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
colnames(theta_mean) <- trait_names

recovery <- sapply(trait_names, function(t) {
  cor(theta_mean[, t], gt_aligned[[t]])
})
message("\nGround-truth recovery (Pearson r):")
for (t in trait_names) message(sprintf("  %s: %+.3f", t, recovery[t]))

# Even though we didn't model Omega, compute sample correlation of theta_mean
sample_omega <- cor(theta_mean)
dimnames(sample_omega) <- list(trait_names, trait_names)
message("\nSample correlation of theta_mean (NOT estimated by model):")
print(round(sample_omega, 2))

a_summary <- summary(fit, pars = "a_pos")$summary
message("\na_pos summary (head):")
print(round(head(a_summary[, c("mean", "sd", "n_eff", "Rhat")], 6), 3))

out <- list(
  fit = fit,
  theta_mean = theta_mean,
  sample_omega = sample_omega,
  recovery = recovery,
  persona_ids = wide$persona_id,
  responses_path = responses_path
)
saveRDS(out, output_path)
message("\nSaved: ", output_path)

diagnostics <- rstan::summary(fit)$summary
high_rhat <- sum(diagnostics[, "Rhat"] > 1.05, na.rm = TRUE)
low_neff  <- sum(diagnostics[, "n_eff"] < 100, na.rm = TRUE)
message(sprintf("\nConvergence: %d Rhat>1.05, %d n_eff<100",
                high_rhat, low_neff))
