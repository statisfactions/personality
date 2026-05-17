#!/usr/bin/env Rscript
# Aggregate W12 ablation results: compare top30 / bot30 / rand30 / full60
# TIRT recovery across the 7-model cohort x 3 persona forms.
#
# Metrics:
#   - diagonal r (across-person, per-trait), cohort-mean |r|
#   - profile r (within-person, across-traits) Spearman, sign-corrected

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})

MODELS <- c("Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7")
FORMS  <- c("description", "ipip_raw", "ipip_reflowed")
SUBSETS <- c("top30", "bot30", "rand30")
trait_names <- c("A", "C", "E", "N", "O")

# Full-P=60 sidecars live in results/persona/persona_gfc_tirt_<M>_ipipneogfc60_hf_<F>.json
# Ablation sidecars in results/persona/ablation/persona_gfc_tirt_<M>_ipipneogfc60_hf_<F>_<sub>.json

sidecar_path <- function(model, form, subset) {
  if (subset == "full60") {
    sprintf("results/persona/persona_gfc_tirt_%s_ipipneogfc60_hf_%s.json", model, form)
  } else {
    sprintf("results/persona/ablation/persona_gfc_tirt_%s_ipipneogfc60_hf_%s_%s.json",
            model, form, subset)
  }
}

rds_path <- function(model, form, subset) {
  if (subset == "full60") {
    sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s_indep_fit.rds", model, form)
  } else {
    sprintf("psychometrics/gfc_tirt/ablation_subsets/%s_ipipneogfc60_hf_%s_%s_indep_fit.rds",
            model, form, subset)
  }
}

# Ground truth
personas_json <- fromJSON("instruments/synthetic_personas.json")
gt <- data.frame(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O,
  stringsAsFactors = FALSE
)

rows <- list()
profile_pool <- list()
for (m in MODELS) for (f in FORMS) for (s in c("full60", SUBSETS)) {
  sc <- sidecar_path(m, f, s)
  rd <- rds_path(m, f, s)
  if (!file.exists(sc) || !file.exists(rd)) {
    message("Missing: ", sc); next
  }
  d <- fromJSON(sc)
  diag_r <- unlist(d$diagonal_correlations)
  diag_abs <- mean(abs(diag_r))

  fit_obj <- readRDS(rd)
  theta <- fit_obj$theta_mean
  pids  <- fit_obj$persona_ids
  g <- gt[match(pids, gt$persona_id), trait_names]
  signs <- ifelse(diag_r < 0, -1, 1)
  theta_signed <- sweep(theta, 2, signs, `*`)
  profile_sp <- suppressWarnings(sapply(seq_len(nrow(theta_signed)), function(i) {
    cor(theta_signed[i, ], as.numeric(g[i, ]), method = "spearman")
  }))

  rows[[length(rows) + 1]] <- data.frame(
    model = m, form = f, subset = s,
    diag_abs = diag_abs,
    profile_sp_mean = mean(profile_sp, na.rm = TRUE),
    profile_sp_median = median(profile_sp, na.rm = TRUE),
    pct_profile_gt_0 = mean(profile_sp > 0, na.rm = TRUE) * 100,
    n_sign_flips = sum(signs == -1),
    stringsAsFactors = FALSE
  )
  profile_pool[[length(profile_pool) + 1]] <- data.frame(
    model = m, form = f, subset = s,
    profile_sp = profile_sp, stringsAsFactors = FALSE
  )
}

df <- do.call(rbind, rows)
pool <- do.call(rbind, profile_pool)

df$subset <- factor(df$subset, levels = c("full60", "top30", "rand30", "bot30"))

cat("\n=== Per-fit recovery ===\n")
print(df %>% arrange(form, model, subset), row.names = FALSE, digits = 3)

cat("\n=== Cohort mean |r| (diagonal) by subset x form ===\n")
diag_tbl <- df %>% group_by(form, subset) %>%
  summarise(diag_abs = mean(diag_abs), .groups = "drop") %>%
  pivot_wider(names_from = subset, values_from = diag_abs)
print(diag_tbl, digits = 3)

cat("\n=== Cohort grand mean |r| by subset ===\n")
grand <- df %>% group_by(subset) %>%
  summarise(diag_abs = mean(diag_abs),
            profile_sp = mean(profile_sp_mean),
            pct_gt_0 = mean(pct_profile_gt_0), .groups = "drop")
print(grand, digits = 3)

cat("\n=== Profile spearman mean by subset x form ===\n")
prof_tbl <- df %>% group_by(form, subset) %>%
  summarise(profile_sp = mean(profile_sp_mean), .groups = "drop") %>%
  pivot_wider(names_from = subset, values_from = profile_sp)
print(prof_tbl, digits = 3)

# Per-model x subset cohort means (description form only, the strongest)
cat("\n=== Per-model diag |r| (description form) by subset ===\n")
desc_tbl <- df %>% filter(form == "description") %>%
  select(model, subset, diag_abs) %>%
  pivot_wider(names_from = subset, values_from = diag_abs)
print(desc_tbl, digits = 3)

# Save
out <- list(per_fit = df, cohort_grand = grand,
            diag_by_form = diag_tbl, profile_by_form = prof_tbl,
            desc_per_model = desc_tbl)
dir.create("results/persona/ablation", showWarnings = FALSE, recursive = TRUE)
write(toJSON(out, pretty = TRUE, auto_unbox = TRUE),
      "results/persona/ablation/ablation_summary.json")
message("\nWrote results/persona/ablation/ablation_summary.json")
