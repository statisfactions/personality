#!/usr/bin/env Rscript
# W12 §5c.4: Re-run SDR vs assistant-default partial correlation on
# asst_top30 (assistant-matched at the pair level) and asst_bot30
# (assistant-mismatched). Compare to the full-P60 baseline from §5c.2.
#
# Prediction: if the W11 instrument's SDR-matching constraint determines
# framing, then in the asst_top30 subset (where assistant-default is
# instead controlled at the pair level), SDR should LOSE unique variance
# and assistant-default should GAIN it.

suppressMessages({
  library(jsonlite)
  library(rstan)
  library(dplyr)
})

instr <- fromJSON("instruments/ipip_neo_gfc_P60.json", flatten = FALSE)
self <- fromJSON("results/persona/cohort_self_rating_P60.json", flatten = FALSE)
models <- names(self$ratings)

# Per-item: cohort EV and selfrating_dist (within-model |EV-4| then averaged)
ipip_ids_all <- unique(c(instr$pairs$left$ipip_id, instr$pairs$right$ipip_id))
ev_per_item <- list()
for (m in models) {
  ev_per_item[[m]] <- sapply(ipip_ids_all, function(id) self$ratings[[m]][[id]]$ev)
}
selfrating_dist_by_id <- sapply(ipip_ids_all, function(id) {
  mean(sapply(models, function(m) abs(ev_per_item[[m]][id] - 4)))
})

# Per-item SDR distance
sd_by_id <- sapply(ipip_ids_all, function(id) {
  rows <- which(instr$pairs$left$ipip_id == id | instr$pairs$right$ipip_id == id)
  for (i in rows) {
    if (instr$pairs$left$ipip_id[i] == id) return(instr$pairs$left$sd[i])
    if (instr$pairs$right$ipip_id[i] == id) return(instr$pairs$right$sd[i])
  }
  NA
})
desirability_dist_by_id <- abs(sd_by_id - 5)

# Subsets
subsets <- fromJSON("psychometrics/gfc_tirt/ablation_assistant_subsets.json",
                    flatten = FALSE)

run_analysis <- function(rds_pattern, subset_name, blocks_in_subset) {
  rds_files <- list.files(
    path = dirname(rds_pattern),
    pattern = basename(rds_pattern),
    full.names = TRUE
  )
  if (length(rds_files) == 0) {
    cat(sprintf("\n[%s] no rds files matched pattern %s\n", subset_name, rds_pattern))
    return(NULL)
  }
  cat(sprintf("\n[%s] %d fits\n", subset_name, length(rds_files)))

  # Which original blocks are in this subset, in sorted order (used by
  # filter_asst_match_responses to remap to 1..30).
  sorted_orig_blocks <- sort(blocks_in_subset)
  # Per fit, extract a_pos and assign to (item, fit). The filtered file's
  # block IDs go 1..30; orig block = sorted_orig_blocks[k]. For each block
  # in the subset, L statement is at j=2k-1, R at j=2k.
  per_item_loadings <- matrix(NA, nrow = 2 * length(sorted_orig_blocks),
                              ncol = length(rds_files))
  per_item_ipip <- character(2 * length(sorted_orig_blocks))
  for (i in seq_along(sorted_orig_blocks)) {
    orig_block <- sorted_orig_blocks[i]
    pair_row <- instr$pairs[instr$pairs$block == orig_block, ]
    per_item_ipip[2*i - 1] <- pair_row$left$ipip_id
    per_item_ipip[2*i]     <- pair_row$right$ipip_id
  }
  for (k in seq_along(rds_files)) {
    fit_obj <- readRDS(rds_files[k])
    a <- colMeans(rstan::extract(fit_obj$fit, pars = "a_pos")$a_pos)
    if (length(a) != nrow(per_item_loadings)) {
      cat(sprintf("  skip (J mismatch): %s\n", basename(rds_files[k])))
      next
    }
    per_item_loadings[, k] <- a
  }
  a_pos_mean <- rowMeans(per_item_loadings, na.rm = TRUE)

  df <- data.frame(
    ipip_id = per_item_ipip,
    a_pos_mean = a_pos_mean,
    desirability_dist = desirability_dist_by_id[per_item_ipip],
    selfrating_dist   = selfrating_dist_by_id[per_item_ipip],
    stringsAsFactors = FALSE
  )

  # Partials
  partial_cor <- function(a, b, c) {
    r_ab <- cor(a, b); r_ac <- cor(a, c); r_bc <- cor(b, c)
    (r_ab - r_ac * r_bc) / sqrt((1 - r_ac^2) * (1 - r_bc^2))
  }
  result <- list(
    subset = subset_name,
    n_items = nrow(df),
    r_sdr_assistant = cor(df$desirability_dist, df$selfrating_dist),
    cor_loading_sdr = cor(df$a_pos_mean, df$desirability_dist),
    cor_loading_assistant = cor(df$a_pos_mean, df$selfrating_dist),
    partial_loading_sdr_given_assistant =
      partial_cor(df$a_pos_mean, df$desirability_dist, df$selfrating_dist),
    partial_loading_assistant_given_sdr =
      partial_cor(df$a_pos_mean, df$selfrating_dist, df$desirability_dist)
  )
  cat(sprintf("  cor(SDR, assistant)                 = %+.3f\n", result$r_sdr_assistant))
  cat(sprintf("  univariate r(load, SDR)             = %+.3f\n", result$cor_loading_sdr))
  cat(sprintf("  univariate r(load, assistant)       = %+.3f\n", result$cor_loading_assistant))
  cat(sprintf("  partial r(load, SDR | assistant)    = %+.3f\n", result$partial_loading_sdr_given_assistant))
  cat(sprintf("  partial r(load, assistant | SDR)    = %+.3f\n", result$partial_loading_assistant_given_sdr))
  result
}

cat("============================================================\n")
cat("W12 §5c.4: SDR-vs-assistant partial correlations by subset\n")
cat("============================================================\n")

baseline <- run_analysis(
  rds_pattern = "psychometrics/gfc_tirt/.*_ipipneogfc60_hf_.*_indep_fit\\.rds$",
  subset_name = "full_p60 (baseline)",
  blocks_in_subset = instr$pairs$block
)
# baseline runner gets confused by globbing; rebuild explicitly
all_p60 <- list.files(
  "psychometrics/gfc_tirt",
  pattern = "_ipipneogfc60_hf_.*_indep_fit\\.rds$",
  full.names = TRUE
)
all_p60 <- all_p60[!grepl("_n25_|_fake_good_", all_p60)]
cat("\n--- recomputing full-P60 baseline directly ---\n")
# Build per-item loadings from full-P60 fits (each fit has J=120 items)
j_to_ipip <- character(2 * nrow(instr$pairs))
for (i in seq_len(nrow(instr$pairs))) {
  j_to_ipip[2*i - 1] <- instr$pairs$left$ipip_id[i]
  j_to_ipip[2*i]     <- instr$pairs$right$ipip_id[i]
}
mat <- matrix(NA, nrow = length(j_to_ipip), ncol = length(all_p60))
for (k in seq_along(all_p60)) {
  fit_obj <- readRDS(all_p60[k])
  a <- colMeans(rstan::extract(fit_obj$fit, pars = "a_pos")$a_pos)
  mat[, k] <- a
}
df_full <- data.frame(
  ipip_id = j_to_ipip,
  a_pos_mean = rowMeans(mat),
  desirability_dist = desirability_dist_by_id[j_to_ipip],
  selfrating_dist   = selfrating_dist_by_id[j_to_ipip],
  stringsAsFactors = FALSE
)
partial_cor <- function(a, b, c) {
  r_ab <- cor(a, b); r_ac <- cor(a, c); r_bc <- cor(b, c)
  (r_ab - r_ac * r_bc) / sqrt((1 - r_ac^2) * (1 - r_bc^2))
}
baseline_result <- list(
  subset = "full_p60",
  n_items = nrow(df_full),
  cor_loading_sdr = cor(df_full$a_pos_mean, df_full$desirability_dist),
  cor_loading_assistant = cor(df_full$a_pos_mean, df_full$selfrating_dist),
  partial_loading_sdr_given_assistant =
    partial_cor(df_full$a_pos_mean, df_full$desirability_dist, df_full$selfrating_dist),
  partial_loading_assistant_given_sdr =
    partial_cor(df_full$a_pos_mean, df_full$selfrating_dist, df_full$desirability_dist)
)
cat(sprintf("  univariate r(load, SDR)             = %+.3f\n", baseline_result$cor_loading_sdr))
cat(sprintf("  univariate r(load, assistant)       = %+.3f\n", baseline_result$cor_loading_assistant))
cat(sprintf("  partial r(load, SDR | assistant)    = %+.3f\n", baseline_result$partial_loading_sdr_given_assistant))
cat(sprintf("  partial r(load, assistant | SDR)    = %+.3f\n", baseline_result$partial_loading_assistant_given_sdr))

top_result <- run_analysis(
  rds_pattern = "psychometrics/gfc_tirt/ablation_assistant_subsets/.*_asst_top30_indep_fit\\.rds$",
  subset_name = "asst_top30 (assistant-matched at pair level)",
  blocks_in_subset = subsets$asst_top30
)
bot_result <- run_analysis(
  rds_pattern = "psychometrics/gfc_tirt/ablation_assistant_subsets/.*_asst_bot30_indep_fit\\.rds$",
  subset_name = "asst_bot30 (assistant-mismatched at pair level)",
  blocks_in_subset = subsets$asst_bot30
)

cat("\n============================================================\n")
cat("Summary: did the partial correlations flip under asst-matched?\n")
cat("============================================================\n")
fmt <- function(x) if (is.null(x)) "NA" else sprintf("%+.3f", x)
cat(sprintf("\n%-40s %12s %12s %12s\n",
            "metric", "full_p60", "asst_top30", "asst_bot30"))
cat(sprintf("%-40s %12s %12s %12s\n",
            "univariate r(load, SDR)",
            fmt(baseline_result$cor_loading_sdr),
            fmt(top_result$cor_loading_sdr),
            fmt(bot_result$cor_loading_sdr)))
cat(sprintf("%-40s %12s %12s %12s\n",
            "univariate r(load, assistant)",
            fmt(baseline_result$cor_loading_assistant),
            fmt(top_result$cor_loading_assistant),
            fmt(bot_result$cor_loading_assistant)))
cat(sprintf("%-40s %12s %12s %12s\n",
            "partial r(load, SDR | assistant)",
            fmt(baseline_result$partial_loading_sdr_given_assistant),
            fmt(top_result$partial_loading_sdr_given_assistant),
            fmt(bot_result$partial_loading_sdr_given_assistant)))
cat(sprintf("%-40s %12s %12s %12s\n",
            "partial r(load, assistant | SDR)",
            fmt(baseline_result$partial_loading_assistant_given_sdr),
            fmt(top_result$partial_loading_assistant_given_sdr),
            fmt(bot_result$partial_loading_assistant_given_sdr)))

out <- list(
  full_p60 = baseline_result,
  asst_top30 = top_result,
  asst_bot30 = bot_result
)
dir.create("results/persona", showWarnings = FALSE, recursive = TRUE)
write(toJSON(out, pretty = TRUE, auto_unbox = TRUE),
      "results/persona/persona_asst_match_partial.json")
message("\nWrote results/persona/persona_asst_match_partial.json")
