#!/usr/bin/env Rscript
# W12 §5e: per-trait theta shift under FG-prefix vs FG-suffix.
#
# Discriminates three hypotheses for FG-prefix cohort Δ = +0.040:
#   (a) "Forgetting": FG fades over distance, model reverts to honest.
#       Predicts per-trait shifts ≈ 0 across all traits.
#   (b) "Task-framing prime": FG-prefix primes careful engagement;
#       persona encoding strengthens; FG itself ignored. Predicts
#       per-trait shifts ≈ 0 but with overall recovery improved.
#   (c) "Partial FG influence": attenuated FG signal. Predicts shifts
#       in same direction as FG-suffix (N↓, E↑, C↑) but smaller.
#
# Reference §5b FG-suffix per-trait shifts:
#   A +0.007, C +0.012, E +0.031, N -0.036, O -0.016

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
})

MODELS <- c("Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7")
FORMS  <- c("description", "ipip_raw", "ipip_reflowed")
trait_names <- c("A", "C", "E", "N", "O")

gt_json <- fromJSON("instruments/synthetic_personas.json")
gt <- data.frame(
  persona_id = gt_json$personas$persona_id,
  A = gt_json$personas$z_scores$A,
  C = gt_json$personas$z_scores$C,
  E = gt_json$personas$z_scores$E,
  N = gt_json$personas$z_scores$N,
  O = gt_json$personas$z_scores$O,
  stringsAsFactors = FALSE
)

load_theta <- function(model, form, condition_suffix) {
  if (condition_suffix == "") {
    rds <- sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s_indep_fit.rds",
                   model, form)
  } else {
    rds <- sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s_%s_indep_fit.rds",
                   model, form, condition_suffix)
  }
  if (!file.exists(rds)) return(NULL)
  fit_obj <- readRDS(rds)
  list(theta = fit_obj$theta_mean, pids = fit_obj$persona_ids,
       recovery = fit_obj$recovery)
}

per_trait_shifts <- function(condition_suffix, label) {
  rows <- list()
  for (m in MODELS) for (f in FORMS) {
    h <- load_theta(m, f, "")
    fg <- load_theta(m, f, condition_suffix)
    if (is.null(h) || is.null(fg)) next
    ord_h <- match(fg$pids, h$pids)
    if (any(is.na(ord_h))) next
    th <- h$theta[ord_h, ]
    tf <- fg$theta
    signs <- ifelse(h$recovery < 0, -1, 1)
    th <- sweep(th, 2, signs, `*`)
    tf <- sweep(tf, 2, signs, `*`)
    for (t in trait_names) {
      rows[[length(rows) + 1]] <- data.frame(
        model = m, form = f, trait = t,
        shift = mean(tf[, t]) - mean(th[, t]),
        stringsAsFactors = FALSE
      )
    }
  }
  df <- do.call(rbind, rows)
  summary <- df %>% group_by(trait) %>%
    summarise(mean_shift = mean(shift),
              median_shift = median(shift),
              sd_shift = sd(shift),
              n_cells = n(),
              .groups = "drop")
  cat(sprintf("\n=== %s per-trait theta shift (sign-aligned, cohort mean) ===\n", label))
  print(summary, digits = 3)
  cat(sprintf("  mean |trait shift|: %.3f\n",
              mean(abs(summary$mean_shift))))
  invisible(summary)
}

cat("============================================================\n")
cat("W12 §5e: FG-suffix vs FG-prefix per-trait theta shift\n")
cat("============================================================\n")
suf <- per_trait_shifts("fake_good", "FG-SUFFIX")
pfx <- per_trait_shifts("fake_good_fgpfx", "FG-PREFIX")

cat("\n=== Side-by-side ===\n")
cat(sprintf("  %-5s  %12s  %12s  %12s\n", "trait", "Δ (suffix)", "Δ (prefix)", "pfx / suf"))
for (t in trait_names) {
  ds <- suf$mean_shift[suf$trait == t]
  dp <- pfx$mean_shift[pfx$trait == t]
  ratio <- if (abs(ds) > 1e-6) dp / ds else NA
  cat(sprintf("  %-5s  %+12.4f  %+12.4f  %12s\n",
              t, ds, dp,
              if (is.na(ratio)) "NA" else sprintf("%+.2f", ratio)))
}

cat("\n=== Hypothesis test ===\n")
fg_predicted_dirs <- c(A = +1, C = +1, E = +1, N = -1, O = NA)  # NA = ambivalent
cat("FG-suffix matches predicted FG direction (N↓, E/C/A↑): ")
matches_suf <- 0; total <- 0
for (t in names(fg_predicted_dirs)) {
  pred <- fg_predicted_dirs[[t]]
  if (is.na(pred)) next
  observed <- sign(suf$mean_shift[suf$trait == t])
  if (observed == pred) matches_suf <- matches_suf + 1
  total <- total + 1
}
cat(sprintf("%d/%d traits\n", matches_suf, total))

matches_pfx <- 0
for (t in names(fg_predicted_dirs)) {
  pred <- fg_predicted_dirs[[t]]
  if (is.na(pred)) next
  observed <- sign(pfx$mean_shift[pfx$trait == t])
  if (observed == pred) matches_pfx <- matches_pfx + 1
}
cat(sprintf("FG-prefix matches predicted FG direction (N↓, E/C/A↑): %d/%d traits\n",
            matches_pfx, total))

dir(file.path("results", "persona"), full.names = FALSE)  # silence
write(toJSON(list(fg_suffix = suf, fg_prefix = pfx),
             pretty = TRUE, auto_unbox = TRUE),
      "results/persona/persona_fg_per_trait_suffix_vs_prefix.json")
message("\nWrote results/persona/persona_fg_per_trait_suffix_vs_prefix.json")
