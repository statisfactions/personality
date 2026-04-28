#!/usr/bin/env Rscript
# Fit the custom Okada-style Stan TIRT (results/tirt_okada.stan) to one
# model's GFC-30 responses on the s1..s50 persona subset.
#
# Usage:
#   Rscript results/fit_tirt_okada.R <responses.json> <output.rds>
# Example:
#   Rscript results/fit_tirt_okada.R \
#     results/gemma3-12b_gfc30_synthetic.json \
#     results/gemma3_okada_stan_fit.rds

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: fit_tirt_okada.R <responses.json> <output.rds>")
}
responses_path <- args[1]
output_path    <- args[2]
n_personas     <- if (length(args) >= 3) as.integer(args[3]) else 50L
n_iter         <- if (length(args) >= 4) as.integer(args[4]) else 1500L
n_chains       <- if (length(args) >= 5) as.integer(args[5]) else 4L

stan_file <- "results/tirt_okada.stan"
if (!file.exists(stan_file)) stop("Stan file not found at ", stan_file)

message("Stan model: ", stan_file)
message("Responses:  ", responses_path)
message("Output:     ", output_path)
message("Personas:   ", n_personas, "  Chains: ", n_chains, "  Iter: ", n_iter)

# --- Load instrument and ground truth ---
inst <- fromJSON("instruments/okada_gfc30.json")
pairs <- inst$pairs
P <- nrow(pairs)
D <- 5L
K <- 7L
trait_names <- c("A", "C", "E", "N", "O")
trait_idx <- setNames(seq_along(trait_names), trait_names)

# Build per-statement metadata: 60 unique statements (2 per pair)
statement_id <- function(p_row, side) {
  paste0(p_row[[side]]$trait, ifelse(p_row[[side]]$keying == "+", "p", "n"),
         p_row$block)
}
stmt_df <- tibble(
  block = rep(seq_len(P), each = 2),
  side  = rep(c("L", "R"), times = P),
  trait = c(rbind(pairs$left$trait,   pairs$right$trait)),
  key   = c(rbind(pairs$left$keying,  pairs$right$keying))
) %>%
  mutate(stmt_name = paste0(trait, ifelse(key == "+", "p", "n"), block),
         g = ifelse(key == "+", 1L, -1L),
         trait_id = trait_idx[trait])
# Each (block, side) is its own statement (Okada treats them as 2P unique)
stmt_df$stmt_index <- seq_len(nrow(stmt_df))
J <- nrow(stmt_df)

L_idx <- stmt_df$stmt_index[stmt_df$side == "L"]
R_idx <- stmt_df$stmt_index[stmt_df$side == "R"]

# --- Load responses ---
raw <- fromJSON(responses_path, flatten = TRUE)
results_df <- as_tibble(raw$results)

# Keep s1..s50, complete cases only
target_ids <- paste0("s", seq_len(n_personas))
results_df <- results_df %>%
  filter(persona_id %in% target_ids,
         !is.na(response_argmax)) %>%
  mutate(response_raw = as.integer(response_argmax),
         # Inference randomized L/R per prompt; un-swap so response is in
         # instrument-canonical L/R coords (matches stmt_df construction).
         response = ifelse(swapped, 8L - response_raw, response_raw))

# Wide matrix: rows = personas (preserving order in target_ids), cols = blocks
wide <- results_df %>%
  select(persona_id, block, response) %>%
  pivot_wider(names_from = block, values_from = response,
              names_prefix = "b") %>%
  filter(persona_id %in% target_ids) %>%
  arrange(match(persona_id, target_ids))

# Drop rows with missing blocks
keep <- complete.cases(wide[, -1])
wide <- wide[keep, ]
N <- nrow(wide)
message("Complete respondents: ", N, " / ", n_personas)

y_mat <- as.matrix(wide[, paste0("b", seq_len(P))])
mode(y_mat) <- "integer"
stopifnot(all(y_mat >= 1 & y_mat <= K))

# --- Ground truth, aligned to wide order ---
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

# --- Stan data ---
stan_data <- list(
  N = N, P = P, J = J, D = D, K = K,
  trait = stmt_df$trait_id,
  g     = stmt_df$g,
  L     = L_idx,
  R     = R_idx,
  y     = y_mat
)

message("\nStan data sanity:")
message("  N=", N, " P=", P, " J=", J, " D=", D, " K=", K)
message("  trait counts: ",
        paste(names(table(trait_names[stmt_df$trait_id])),
              table(trait_names[stmt_df$trait_id]), sep = "=",
              collapse = " "))
message("  keying counts (L items): +=", sum(stmt_df$g[stmt_df$side=="L"] == 1),
        " -=", sum(stmt_df$g[stmt_df$side=="L"] == -1))
message("  response category usage:")
print(table(y_mat))

# --- Compile + fit ---
message("\nCompiling Stan model...")
model <- stan_model(stan_file)

message("Sampling: ", n_chains, " chains x ", n_iter, " iter (", n_iter/2, " warmup)")
fit <- sampling(
  model, data = stan_data,
  chains = n_chains, iter = n_iter, warmup = n_iter %/% 2,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  refresh = 200,
  pars = c("theta", "a_pos", "Omega", "kappa")
)

# --- Recovery summary ---
theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
colnames(theta_mean) <- trait_names

recovery <- sapply(trait_names, function(t) {
  cor(theta_mean[, t], gt_aligned[[t]])
})
message("\nGround-truth recovery (Pearson r):")
for (t in trait_names) {
  message(sprintf("  %s: %+.3f", t, recovery[t]))
}

Omega_mean <- apply(rstan::extract(fit, "Omega")$Omega, c(2, 3), mean)
dimnames(Omega_mean) <- list(trait_names, trait_names)
message("\nEstimated trait correlations (posterior mean Omega):")
print(round(Omega_mean, 2))

gt_omega <- cor(as.matrix(gt_aligned[, trait_names]))
message("\nGround-truth correlations on this 50-respondent subset:")
print(round(gt_omega, 2))

a_summary <- summary(fit, pars = "a_pos")$summary
message("\na_pos summary (head):")
print(round(head(a_summary[, c("mean", "sd", "n_eff", "Rhat")], 6), 3))

# --- Save ---
out <- list(
  fit = fit,
  theta_mean = theta_mean,
  Omega_mean = Omega_mean,
  gt_omega = gt_omega,
  recovery = recovery,
  persona_ids = wide$persona_id,
  responses_path = responses_path,
  n_personas = N
)
saveRDS(out, output_path)
message("\nSaved: ", output_path)

# Convergence sanity
diagnostics <- rstan::summary(fit)$summary
high_rhat <- sum(diagnostics[, "Rhat"] > 1.05, na.rm = TRUE)
low_neff  <- sum(diagnostics[, "n_eff"] < 100, na.rm = TRUE)
message(sprintf("\nConvergence: %d params with Rhat>1.05, %d with n_eff<100",
                high_rhat, low_neff))
