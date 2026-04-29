#!/usr/bin/env Rscript
# Post-hoc analysis of the 4-model pooled fit:
#   - per-model recovered inter-trait correlation matrix (theta_hat, honest)
#   - true inter-trait correlation matrix (synthetic personas)
#   - neutral placement (bare, respondent) per model on the latent θ scale
#
# Output: prints tables; saves a tidy summary tibble for the writeup.

suppressMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(tibble)
})

fit_path <- "psychometrics/gfc_tirt/pooled_haiku_gemma27b_llama70b_qwen72b_fit.rds"
out      <- readRDS(fit_path)
trait_names <- c("A", "C", "E", "N", "O")
hat_names   <- paste0(trait_names, "_hat")

# Ground-truth correlation among the 50 target personas
personas_json <- fromJSON("instruments/synthetic_personas.json")
gt_full <- tibble(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O
)
target_ids <- paste0("s", seq_len(50))
gt <- gt_full %>% filter(persona_id %in% target_ids)
true_cor <- cor(gt %>% select(all_of(trait_names)))

cat("=== TRUE inter-trait correlation (50 target personas) ===\n")
print(round(true_cor, 2))

# Helper to print a cor matrix
print_cor <- function(label, mat) {
  cat(sprintf("\n=== %s ===\n", label))
  print(round(mat, 2))
}

# Recovered correlation per model (honest condition only)
theta_df <- out$theta_df
models <- unique(theta_df$model)

scored_cors <- list()
for (m in models) {
  sub <- theta_df %>% filter(model == m, condition == "honest")
  if (nrow(sub) < 5) next
  scored <- cor(sub %>% select(all_of(hat_names)))
  colnames(scored) <- trait_names
  rownames(scored) <- trait_names
  scored_cors[[m]] <- scored
  print_cor(sprintf("Recovered cor(theta_hat) — %s, honest, N=%d", m, nrow(sub)),
            scored)
}

# Off-diagonal Frobenius distance from true_cor
cat("\n=== Distance to TRUE inter-trait correlation (off-diag only) ===\n")
offdiag <- function(M) M[upper.tri(M)]
true_off <- offdiag(true_cor)
dist_tbl <- tibble(
  model = names(scored_cors),
  rmsd  = sapply(scored_cors, function(M) sqrt(mean((offdiag(M) - true_off)^2))),
  max_abs_dev = sapply(scored_cors, function(M) max(abs(offdiag(M) - true_off))),
  mean_off_diag_recovered = sapply(scored_cors, function(M) mean(offdiag(M))),
  mean_off_diag_true      = mean(true_off)
)
print(dist_tbl)

# Neutral placement
cat("\n=== Neutral placement (pooled latent space) ===\n")
print(out$neutral, n = 100)

# Save tidy outputs for the writeup
saveRDS(list(
  true_cor = true_cor,
  scored_cors = scored_cors,
  dist_tbl = dist_tbl,
  neutral = out$neutral,
  recovery = out$recovery,
  recovery_summary = out$recovery_summary
), "psychometrics/gfc_tirt/pooled_haiku_gemma27b_llama70b_qwen72b_summary.rds")
cat("\nSaved: psychometrics/gfc_tirt/pooled_haiku_gemma27b_llama70b_qwen72b_summary.rds\n")
