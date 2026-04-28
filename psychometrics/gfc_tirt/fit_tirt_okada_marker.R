#!/usr/bin/env Rscript
# Fit the marker-anchored TIRT (results/tirt_okada_marker.stan).
# Same data wrangling as fit_tirt_okada.R; differs only in the Stan file
# and in passing the marker[] vector.
#
# Usage:
#   Rscript results/fit_tirt_okada_marker.R <responses.json> <output.rds> [n_personas] [iter] [chains]

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: fit_tirt_okada_marker.R <responses.json> <output.rds>")
responses_path <- args[1]
output_path    <- args[2]
n_personas     <- if (length(args) >= 3) as.integer(args[3]) else 50L
n_iter         <- if (length(args) >= 4) as.integer(args[4]) else 1500L
n_chains       <- if (length(args) >= 5) as.integer(args[5]) else 4L

stan_file <- "results/tirt_okada_marker.stan"
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

# Identify marker per trait: the first positively-keyed statement (lowest
# stmt_index) for each trait
marker_idx <- integer(D)
for (d in seq_len(D)) {
  candidates <- stmt_df %>%
    filter(trait_id == d, g == 1L) %>%
    arrange(stmt_index)
  if (nrow(candidates) == 0) {
    stop(sprintf("Trait %s has no positively-keyed item; cannot pick marker",
                 trait_names[d]))
  }
  marker_idx[d] <- candidates$stmt_index[1]
}
message("Markers (statement indices):")
for (d in seq_len(D)) {
  s <- stmt_df[marker_idx[d], ]
  side_text <- if (s$side == "L") pairs$left$text[s$block] else pairs$right$text[s$block]
  message(sprintf("  %s: stmt %d  block %d %s  \"%s\"",
                  trait_names[d], marker_idx[d], s$block, s$side, side_text))
}

# --- Load responses ---
raw <- fromJSON(responses_path, flatten = TRUE)
results_df <- as_tibble(raw$results)
target_ids <- paste0("s", seq_len(n_personas))
results_df <- results_df %>%
  filter(persona_id %in% target_ids, !is.na(response_argmax)) %>%
  mutate(response_raw = as.integer(response_argmax),
         # Inference randomized L/R per prompt; un-swap so response is in
         # instrument-canonical L/R coords (matches stmt_df construction).
         response = ifelse(swapped, 8L - response_raw, response_raw))

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
  y     = y_mat,
  marker = marker_idx
)

message("\nCompiling marker model...")
model <- stan_model(stan_file)

message("Sampling: ", n_chains, " x ", n_iter)
fit <- sampling(
  model, data = stan_data,
  chains = n_chains, iter = n_iter, warmup = n_iter %/% 2,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  refresh = 200,
  pars = c("theta", "a", "Omega", "kappa")
)

theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
colnames(theta_mean) <- trait_names

recovery <- sapply(trait_names, function(t) {
  cor(theta_mean[, t], gt_aligned[[t]])
})
message("\nGround-truth recovery (Pearson r):")
for (t in trait_names) message(sprintf("  %s: %+.3f", t, recovery[t]))

Omega_mean <- apply(rstan::extract(fit, "Omega")$Omega, c(2, 3), mean)
dimnames(Omega_mean) <- list(trait_names, trait_names)
message("\nEstimated Omega (posterior mean):")
print(round(Omega_mean, 2))

a_summary <- summary(fit, pars = "a")$summary
# Show signs of marker vs non-marker items per trait
message("\nLoading sign summary by trait (mean across items):")
for (d in seq_len(D)) {
  trait_items <- stmt_df %>% filter(trait_id == d)
  a_vals <- a_summary[trait_items$stmt_index, "mean"]
  marker_val <- a_summary[marker_idx[d], "mean"]
  pos_keyed <- trait_items$g == 1L
  message(sprintf("  %s: marker(+)=%.2f | other +keyed mean=%.2f (n=%d) | -keyed mean=%.2f (n=%d)",
                  trait_names[d], marker_val,
                  mean(a_vals[pos_keyed & seq_along(a_vals) != which(trait_items$stmt_index == marker_idx[d])]),
                  sum(pos_keyed) - 1,
                  mean(a_vals[!pos_keyed]),
                  sum(!pos_keyed)))
}

out <- list(
  fit = fit,
  theta_mean = theta_mean,
  Omega_mean = Omega_mean,
  recovery = recovery,
  persona_ids = wide$persona_id,
  marker_idx = marker_idx,
  marker_info = lapply(seq_len(D), function(d) list(
    trait = trait_names[d],
    stmt_index = marker_idx[d],
    block = stmt_df$block[marker_idx[d]],
    side = stmt_df$side[marker_idx[d]]
  )),
  responses_path = responses_path
)
saveRDS(out, output_path)
message("\nSaved: ", output_path)

diagnostics <- rstan::summary(fit)$summary
high_rhat <- sum(diagnostics[, "Rhat"] > 1.05, na.rm = TRUE)
low_neff  <- sum(diagnostics[, "n_eff"] < 100, na.rm = TRUE)
message(sprintf("\nConvergence: %d Rhat>1.05, %d n_eff<100",
                high_rhat, low_neff))
