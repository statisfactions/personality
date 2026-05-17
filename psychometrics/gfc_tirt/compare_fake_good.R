#!/usr/bin/env Rscript
# W12 §X: Compare HONEST vs FAKE-GOOD across two readouts.
#
# Predictions from the W12 loading-as-filter mechanism:
#   (1) Likert shift on each pair = mean(response_FG) - mean(response_HONEST),
#       averaged across personas. Should correlate with item desirability
#       (FG-aligned items shift; FG-misaligned items don't) and with
#       INVERSE loading (low-loading items shift most because they have
#       the most "room" toward the assistant default).
#   (2) TIRT theta shift = mean(theta_FG[i, t]) - mean(theta_HONEST[i, t]),
#       per (model, form, trait). Should be SMALLER than the corresponding
#       Likert-aggregate shift, because the loading-weighted scoring
#       suppresses the FG-sensitive items (low-loading items) and the
#       FG-insensitive items (high-loading items) contribute most of theta.
#
# Output: results/persona/persona_fake_good_comparison.json + console table.

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})

MODELS <- c("Gemma", "Gemma12", "Llama", "Llama8", "Phi4", "Qwen", "Qwen7")
FORMS  <- c("description", "ipip_raw", "ipip_reflowed")
trait_names <- c("A", "C", "E", "N", "O")

# Per-item averaged loading (from W12 P=60 honest fits) for cross-referencing
# Likert shifts to loading.
loading_path <- "psychometrics/gfc_tirt/ablation_subsets.json"
loadings <- if (file.exists(loading_path)) {
  d <- fromJSON(loading_path)
  bi <- as.data.frame(d$block_info)
  bi  # block, a_L, a_R, info, a_mean_pair
} else stop("Missing ", loading_path, " — run rank_pairs_and_subset.R first")

# IPIP-NEO-GFC pair desirability
instr <- fromJSON("instruments/ipip_neo_gfc_P60.json", flatten = FALSE)
pair_meta <- data.frame(
  block = instr$pairs$block,
  left_sd = instr$pairs$left$sd,
  right_sd = instr$pairs$right$sd,
  pair_sd = (instr$pairs$left$sd + instr$pairs$right$sd) / 2,
  stringsAsFactors = FALSE
)

# Persona ground truth (for sign-corrected theta direction)
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

# --- 1. Likert (per-pair) shift ---
load_response_records <- function(path) {
  d <- fromJSON(path, flatten = FALSE)
  r <- d$results
  r$response_argmax <- as.integer(r$response_argmax)
  # Un-swap: if swapped, response 1<->7, 2<->6, ... (8 - x)
  r$response_canonical <- ifelse(r$swapped, 8L - r$response_argmax,
                                  r$response_argmax)
  # Also EV un-swap
  r$ev_canonical <- ifelse(r$swapped, 8 - r$response_ev, r$response_ev)
  r
}

likert_rows <- list()
for (m in MODELS) for (f in FORMS) {
  honest_path <- sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s.json", m, f)
  fg_path     <- sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s_fake_good.json", m, f)
  if (!file.exists(honest_path) || !file.exists(fg_path)) next
  rh <- load_response_records(honest_path)
  rf <- load_response_records(fg_path)
  # Per-block: mean response across personas, in canonical L/R
  mh <- rh %>% group_by(block) %>%
    summarise(mean_resp_honest = mean(response_canonical, na.rm = TRUE),
              .groups = "drop")
  mf <- rf %>% group_by(block) %>%
    summarise(mean_resp_fg = mean(response_canonical, na.rm = TRUE),
              .groups = "drop")
  j <- inner_join(mh, mf, by = "block") %>%
    mutate(shift_likert = mean_resp_fg - mean_resp_honest,
           model = m, form = f)
  likert_rows[[length(likert_rows) + 1]] <- j
}
likert_df <- do.call(rbind, likert_rows) %>%
  left_join(loadings %>% select(block, info, a_mean_pair), by = "block") %>%
  left_join(pair_meta %>% select(block, pair_sd), by = "block")

cat("\n=== Per-block Likert shift correlation with loading & desirability ===\n")
likert_cors <- likert_df %>% group_by(model, form) %>%
  summarise(
    r_shift_vs_info = cor(abs(shift_likert), info, use = "pairwise"),
    r_shift_vs_loading = cor(abs(shift_likert), a_mean_pair, use = "pairwise"),
    r_shift_vs_sd = cor(abs(shift_likert), pair_sd, use = "pairwise"),
    mean_abs_shift = mean(abs(shift_likert)),
    mean_shift = mean(shift_likert),
    .groups = "drop"
  )
print(likert_cors, digits = 3)

cat("\n=== Pooled correlation (Likert shift vs. loading/desirability) ===\n")
pooled_likert_cor <- with(likert_df, {
  data.frame(
    r_abs_shift_vs_info = cor(abs(shift_likert), info, use = "pairwise"),
    r_abs_shift_vs_a_mean = cor(abs(shift_likert), a_mean_pair, use = "pairwise"),
    r_abs_shift_vs_sd = cor(abs(shift_likert), pair_sd, use = "pairwise")
  )
})
print(pooled_likert_cor, digits = 3)

# --- 2. TIRT theta shift ---
load_theta <- function(model, form, condition) {
  rds_path <- if (condition == "honest") {
    sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s_indep_fit.rds", model, form)
  } else {
    sprintf("psychometrics/gfc_tirt/%s_ipipneogfc60_hf_%s_fake_good_indep_fit.rds", model, form)
  }
  if (!file.exists(rds_path)) return(NULL)
  fit_obj <- readRDS(rds_path)
  list(theta = fit_obj$theta_mean, pids = fit_obj$persona_ids,
       recovery = fit_obj$recovery)
}

theta_rows <- list()
for (m in MODELS) for (f in FORMS) {
  h <- load_theta(m, f, "honest")
  fg <- load_theta(m, f, "fake_good")
  if (is.null(h) || is.null(fg)) next
  # Align personas
  ord_h <- match(fg$pids, h$pids)
  if (any(is.na(ord_h))) next
  th <- h$theta[ord_h, ]
  tf <- fg$theta
  # Sign-correct using HONEST diagonal-r signs (so we measure shift in
  # the trait-aligned direction)
  signs <- ifelse(h$recovery < 0, -1, 1)
  th <- sweep(th, 2, signs, `*`)
  tf <- sweep(tf, 2, signs, `*`)
  # Per-trait shift in cohort means
  for (t in trait_names) {
    shift <- mean(tf[, t]) - mean(th[, t])
    theta_rows[[length(theta_rows) + 1]] <- data.frame(
      model = m, form = f, trait = t,
      theta_honest = mean(th[, t]),
      theta_fg     = mean(tf[, t]),
      shift_theta  = shift,
      stringsAsFactors = FALSE
    )
  }
}
theta_df <- do.call(rbind, theta_rows)

cat("\n=== Per-trait TIRT theta cohort-mean shift (HONEST -> FG, sign-aligned) ===\n")
theta_summary <- theta_df %>% group_by(trait) %>%
  summarise(mean_shift = mean(shift_theta),
            median_shift = median(shift_theta),
            sd_shift = sd(shift_theta),
            .groups = "drop")
print(theta_summary, digits = 3)

cat("\n=== Per-(model,form) TIRT mean |theta shift| over 5 traits ===\n")
theta_by_cell <- theta_df %>% group_by(model, form) %>%
  summarise(mean_abs_theta_shift = mean(abs(shift_theta)),
            .groups = "drop")
print(theta_by_cell, digits = 3)

# --- 3. Direct comparison: Likert shift magnitude vs TIRT shift magnitude ---
cat("\n=== Cohort comparison: Likert mean|shift| vs TIRT mean|theta shift| ===\n")
likert_by_cell <- likert_df %>% group_by(model, form) %>%
  summarise(likert_abs_shift_per_pt = mean(abs(shift_likert)),
            .groups = "drop")
cohort_compare <- likert_by_cell %>%
  inner_join(theta_by_cell, by = c("model", "form")) %>%
  mutate(ratio = mean_abs_theta_shift / likert_abs_shift_per_pt)
print(cohort_compare, digits = 3)

cat("\n=== Cohort grand means ===\n")
cat(sprintf("  Likert mean |shift| per pair (7-pt scale): %.3f\n",
            mean(likert_by_cell$likert_abs_shift_per_pt)))
cat(sprintf("  TIRT mean |theta shift| per trait (z scale): %.3f\n",
            mean(theta_by_cell$mean_abs_theta_shift)))

# --- 4. Save ---
dir.create("results/persona", showWarnings = FALSE, recursive = TRUE)
out <- list(
  pooled_likert_correlations = as.list(pooled_likert_cor),
  per_cell_likert_correlations = likert_cors,
  per_trait_theta_shift = theta_summary,
  per_cell_theta_shift = theta_by_cell,
  cohort_comparison = cohort_compare,
  cohort_grand_likert_abs_shift = mean(likert_by_cell$likert_abs_shift_per_pt),
  cohort_grand_theta_abs_shift = mean(theta_by_cell$mean_abs_theta_shift)
)
write(toJSON(out, pretty = TRUE, auto_unbox = TRUE),
      "results/persona/persona_fake_good_comparison.json")
message("\nWrote results/persona/persona_fake_good_comparison.json")
