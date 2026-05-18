#!/usr/bin/env Rscript
# W12 §6.6: Does loading depend on item desirability (SDR), assistant-default
# (no-persona self-rating distance from neutral), or both?
#
# Three per-item measures:
#   - a_pos: TIRT loading (averaged across 21 P=60 fits, per L/R within block)
#   - desirability_dist: |Phase-B-cohort-rated desirability - 5| on 1-9 scale
#       (5 = neutral; high values mean "humans say this is clearly desirable
#       or clearly undesirable")
#   - selfrating_dist: |no-persona Likert EV - 4| on 1-7 scale (4 = neutral;
#       high values mean "assistant has a clear default position on this")
#
# Question 1: Do desirability and self-rating constructs collapse? (cor)
# Question 2: Which predicts loading better? (univariate cors)
# Question 3: Each one's UNIQUE variance? (partial correlations)

suppressMessages({
  library(jsonlite)
  library(rstan)
  library(dplyr)
})

# --- 1. Per-item loadings: rebuild from rds files (averaged across all 21 P=60 fits) ---
rds_files <- list.files(
  "psychometrics/gfc_tirt",
  pattern = "_ipipneogfc60_hf_.*_indep_fit\\.rds$",
  full.names = TRUE
)
rds_files <- rds_files[!grepl("_n25_|_fake_good_", rds_files)]
message("P=60 honest fits: ", length(rds_files))

instr <- fromJSON("instruments/ipip_neo_gfc_P60.json", flatten = FALSE)
pairs <- instr$pairs
# Build a per-J (interleaved L,R) mapping to ipip_id + side
j_to_item <- data.frame()
for (i in seq_len(nrow(pairs))) {
  j_to_item <- rbind(j_to_item, data.frame(
    block = pairs$block[i], side = "L",
    ipip_id = pairs$left$ipip_id[i],
    trait = pairs$left$trait[i],
    keying = pairs$left$keying[i],
    text = pairs$left$text[i],
    sd = pairs$left$sd[i],
    stringsAsFactors = FALSE
  ))
  j_to_item <- rbind(j_to_item, data.frame(
    block = pairs$block[i], side = "R",
    ipip_id = pairs$right$ipip_id[i],
    trait = pairs$right$trait[i],
    keying = pairs$right$keying[i],
    text = pairs$right$text[i],
    sd = pairs$right$sd[i],
    stringsAsFactors = FALSE
  ))
}
j_to_item$j <- seq_len(nrow(j_to_item))

# Pull a_pos posteriors and aggregate
per_item_loadings <- matrix(NA, nrow = nrow(j_to_item), ncol = length(rds_files))
for (k in seq_along(rds_files)) {
  fit_obj <- readRDS(rds_files[k])
  a <- colMeans(rstan::extract(fit_obj$fit, pars = "a_pos")$a_pos)
  per_item_loadings[, k] <- a
}
j_to_item$a_pos_mean <- rowMeans(per_item_loadings)

# --- 2. Self-rating data ---
self <- fromJSON("results/persona/cohort_self_rating_P60.json", flatten = FALSE)
models <- names(self$ratings)
message("Self-rating models: ", length(models), " (", paste(models, collapse=", "), ")")

# Per-item: cohort-mean EV, cohort-mean |EV - 4|
sr <- data.frame(ipip_id = character(), cohort_ev = numeric(),
                 cohort_abs_dev = numeric(), per_model_ev = I(list()),
                 stringsAsFactors = FALSE)
for (ipip_id in unique(j_to_item$ipip_id)) {
  evs <- sapply(models, function(m) self$ratings[[m]][[ipip_id]]$ev)
  sr <- rbind(sr, data.frame(
    ipip_id = ipip_id,
    cohort_ev = mean(evs),
    cohort_abs_dev = mean(abs(evs - 4)),  # within-model |dev|, then averaged
    cohort_mean_dev_abs = abs(mean(evs) - 4),  # |cohort-mean - 4|
    stringsAsFactors = FALSE
  ))
}

# --- 3. Merge ---
df <- j_to_item %>%
  left_join(sr, by = "ipip_id") %>%
  mutate(
    desirability_dist = abs(sd - 5),  # 1-9 scale, neutral = 5
    # cohort_abs_dev = within-model mean of |EV - 4|, the "average distance"
    # cohort_mean_dev_abs = distance of the cohort-mean from neutral, the
    # "cohort-position" distance. We'll use both, but cohort_abs_dev is the
    # more natural "assistant-default-strength" measure.
    selfrating_dist = cohort_abs_dev
  ) %>%
  select(ipip_id, trait, keying, text, sd, desirability_dist,
         cohort_ev, selfrating_dist, cohort_mean_dev_abs, a_pos_mean)

message("\nMerged data: ", nrow(df), " items × ", ncol(df), " cols")
message("Per-item a_pos_mean: ",
        round(min(df$a_pos_mean), 3), " to ", round(max(df$a_pos_mean), 3),
        ", median ", round(median(df$a_pos_mean), 3))
message("desirability_dist (|sd - 5|): ",
        round(min(df$desirability_dist), 2), " to ",
        round(max(df$desirability_dist), 2),
        ", median ", round(median(df$desirability_dist), 2))
message("selfrating_dist (cohort within-model |EV - 4|): ",
        round(min(df$selfrating_dist), 2), " to ",
        round(max(df$selfrating_dist), 2),
        ", median ", round(median(df$selfrating_dist), 2))

# --- 4. Correlations ---
cat("\n=== Q1: Do SDR (desirability) and assistant-default (self-rating) collapse? ===\n")
r_sdr_sd <- cor(df$desirability_dist, df$selfrating_dist)
cat(sprintf("  cor(desirability_dist, selfrating_dist) = %.3f\n", r_sdr_sd))
cat(sprintf("  cor(desirability_dist, cohort_mean_dev_abs) = %.3f\n",
            cor(df$desirability_dist, df$cohort_mean_dev_abs)))

cat("\n=== Q2: Univariate correlations with loading ===\n")
cat(sprintf("  cor(a_pos, desirability_dist) = %.3f (predicted: NEGATIVE if SDR mechanism)\n",
            cor(df$a_pos_mean, df$desirability_dist)))
cat(sprintf("  cor(a_pos, selfrating_dist)  = %.3f (predicted: NEGATIVE if assistant-default mechanism)\n",
            cor(df$a_pos_mean, df$selfrating_dist)))
cat(sprintf("  cor(a_pos, cohort_mean_dev_abs) = %.3f\n",
            cor(df$a_pos_mean, df$cohort_mean_dev_abs)))

cat("\n=== Q3: Partial correlations (which has unique variance for loading?) ===\n")
# r(a, b | c) = (r_ab - r_ac * r_bc) / sqrt((1 - r_ac^2)(1 - r_bc^2))
partial_cor <- function(a, b, c) {
  r_ab <- cor(a, b); r_ac <- cor(a, c); r_bc <- cor(b, c)
  (r_ab - r_ac * r_bc) / sqrt((1 - r_ac^2) * (1 - r_bc^2))
}
p_load_sdr_given_self <- partial_cor(df$a_pos_mean, df$desirability_dist, df$selfrating_dist)
p_load_self_given_sdr <- partial_cor(df$a_pos_mean, df$selfrating_dist, df$desirability_dist)
cat(sprintf("  partial r(loading, desirability_dist | selfrating_dist) = %.3f\n",
            p_load_sdr_given_self))
cat(sprintf("  partial r(loading, selfrating_dist  | desirability_dist) = %.3f\n",
            p_load_self_given_sdr))

cat("\n=== Q4: Per-model — does each model's own self-rating predict loading? ===\n")
for (m in models) {
  evs <- sapply(unique(j_to_item$ipip_id), function(iid) self$ratings[[m]][[iid]]$ev)
  names(evs) <- unique(j_to_item$ipip_id)
  df$per_model_self_dist <- abs(evs[df$ipip_id] - 4)
  cat(sprintf("  %-8s cor(a_pos, |EV - 4|) = %+.3f\n",
              m, cor(df$a_pos_mean, df$per_model_self_dist)))
}

# --- 5. Save & per-item table ---
dir.create("results/persona", showWarnings = FALSE, recursive = TRUE)
out <- list(
  n_items = nrow(df),
  cor_sdr_assistant = r_sdr_sd,
  cor_loading_sdr = cor(df$a_pos_mean, df$desirability_dist),
  cor_loading_assistant = cor(df$a_pos_mean, df$selfrating_dist),
  partial_loading_sdr_given_assistant = p_load_sdr_given_self,
  partial_loading_assistant_given_sdr = p_load_self_given_sdr,
  per_item = df[, c("ipip_id", "trait", "keying", "sd",
                    "desirability_dist", "cohort_ev",
                    "selfrating_dist", "a_pos_mean")]
)
write(toJSON(out, pretty = TRUE, auto_unbox = TRUE),
      "results/persona/persona_sdr_vs_assistant.json")
message("\nWrote results/persona/persona_sdr_vs_assistant.json")
