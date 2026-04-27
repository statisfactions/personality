#!/usr/bin/env Rscript
# Render a markdown summary of the pooled TIRT fit, suitable for inclusion
# in ecb-reports/. Reads results/pooled_tirt_fit.rds and emits markdown
# tables on stdout.
#
# Usage:
#   Rscript results/render_pooled_report.R > ecb-reports/pooled_tirt_summary.md

suppressMessages({
  library(dplyr)
  library(tidyr)
  library(knitr)
})

fit_path <- if (length(commandArgs(trailingOnly = TRUE)) >= 1)
              commandArgs(trailingOnly = TRUE)[1]
            else "results/pooled_tirt_fit.rds"
out <- readRDS(fit_path)

trait_names <- c("A", "C", "E", "N", "O")

cat("# Pooled Okada-style TIRT — replication summary\n\n")
cat(sprintf("Fit: `%s`\n\n", fit_path))
cat(sprintf("Pooled response matrix: **N = %d rows** across LLMs and conditions, P = 30 GFC blocks.\n\n", out$N))

cat("## Rows by (model, condition)\n\n")
counts <- out$wide_meta %>% count(model, condition) %>% arrange(model, condition)
print(kable(counts))
cat("\n\n")

cat("## Per-(model, condition) ground-truth recovery (Pearson r)\n\n")
print(kable(out$recovery, digits = 3))
cat("\n\n")

cat("## Recovery summary (mean signed r vs. mean |r|)\n\n")
print(kable(out$recovery_summary %>%
              select(model, condition, n_personas, mean_signed_r, mean_abs_r),
            digits = 3))
cat("\n\n")

cat("## Neutral placement (model-default θ̂ on persona-fitted scale)\n\n")
print(kable(out$neutral, digits = 3))
cat("\n\n")

# Difference between conditions per model (HONEST vs FAKE-GOOD on θ̂ means)
if (any(out$theta_df$condition == "fakegood")) {
  cat("## Instruction-induced shift (fake-good − honest mean θ̂, per trait, per model)\n\n")
  cat("Positive = movement in direction of socially desirable convention (high A/C/E/O, low N).\n")
  cat("Sign-corrected as `(fake-good − honest) × g_t` with g_N = -1.\n\n")
  shift <- out$theta_df %>%
    filter(condition %in% c("honest", "fakegood")) %>%
    pivot_longer(ends_with("_hat"),
                 names_to = "trait",
                 values_to = "theta_hat",
                 names_pattern = "(.)_hat") %>%
    group_by(model, condition, trait) %>%
    summarise(mean_theta = mean(theta_hat), .groups = "drop") %>%
    pivot_wider(names_from = condition, values_from = mean_theta) %>%
    mutate(
      raw_shift = fakegood - honest,
      g_t = ifelse(trait == "N", -1, 1),
      direction_corrected = raw_shift * g_t
    ) %>%
    select(model, trait, honest, fakegood, raw_shift, direction_corrected)
  print(kable(shift, digits = 3))
  cat("\n")
}
