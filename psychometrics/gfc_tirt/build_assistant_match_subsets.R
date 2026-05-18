#!/usr/bin/env Rscript
# W12 §6.6 follow-up: build pair subsets by within-pair assistant-default
# similarity (rather than informativeness as in the W12 ablation).
#
# Method: per pair p, compute |cohort_self_rating_L - cohort_self_rating_R|
# where the cohort self-rating is the within-model |EV - 4| averaged across
# 7 cohort models (i.e., the assistant-default-strength). Then rank pairs.
#
# Subsets:
#   asst_top30: pairs where L and R have most similar assistant-default
#               strength (effectively assistant-matched at the pair level)
#   asst_bot30: pairs where L and R differ most (assistant-mismatched)
#
# Output: psychometrics/gfc_tirt/ablation_assistant_subsets.json

suppressMessages({
  library(jsonlite)
  library(dplyr)
})

set.seed(20260517)

instr <- fromJSON("instruments/ipip_neo_gfc_P60.json", flatten = FALSE)
self <- fromJSON("results/persona/cohort_self_rating_P60.json", flatten = FALSE)
models <- names(self$ratings)

# For each item, cohort-mean self-rating EV
ipip_ids <- unique(c(instr$pairs$left$ipip_id, instr$pairs$right$ipip_id))
ev_by_id <- sapply(ipip_ids, function(id) {
  mean(sapply(models, function(m) self$ratings[[m]][[id]]$ev))
})
# assistant-default strength = |cohort_mean_ev - 4|
strength_by_id <- abs(ev_by_id - 4)

# Per-pair: |strength_L - strength_R|
pair_diff <- data.frame(
  block = instr$pairs$block,
  ipip_L = instr$pairs$left$ipip_id,
  ipip_R = instr$pairs$right$ipip_id,
  strength_L = strength_by_id[instr$pairs$left$ipip_id],
  strength_R = strength_by_id[instr$pairs$right$ipip_id],
  stringsAsFactors = FALSE
)
pair_diff$asst_diff <- abs(pair_diff$strength_L - pair_diff$strength_R)
pair_diff <- pair_diff[order(pair_diff$asst_diff), ]

cat("\n=== Pair-level assistant-default diff distribution ===\n")
cat(sprintf("  range: [%.3f, %.3f]; median %.3f\n",
            min(pair_diff$asst_diff), max(pair_diff$asst_diff),
            median(pair_diff$asst_diff)))

cat("\n=== Top 5 pairs by ASST-MATCH (smallest diff) ===\n")
print(head(pair_diff, 5), row.names = FALSE, digits = 3)
cat("\n=== Bottom 5 pairs by ASST-MATCH (largest diff = most mismatched) ===\n")
print(tail(pair_diff, 5), row.names = FALSE, digits = 3)

asst_top30 <- sort(pair_diff$block[1:30])
asst_bot30 <- sort(pair_diff$block[(nrow(pair_diff) - 29):nrow(pair_diff)])

cat("\nasst_top30 mean diff: ", round(mean(pair_diff$asst_diff[1:30]), 3),
    "\nasst_bot30 mean diff: ", round(mean(pair_diff$asst_diff[(nrow(pair_diff) - 29):nrow(pair_diff)]), 3),
    "\n", sep = "")

# Trait coverage (need both keying directions per trait for identification)
coverage <- function(blocks) {
  sel <- instr$pairs[instr$pairs$block %in% blocks, ]
  L_traits <- sel$left$trait;  L_keys <- sel$left$keying
  R_traits <- sel$right$trait; R_keys <- sel$right$keying
  table(c(L_traits, R_traits), c(L_keys, R_keys))
}
cat("\nasst_top30 trait coverage:\n"); print(coverage(asst_top30))
cat("\nasst_bot30 trait coverage:\n"); print(coverage(asst_bot30))

out <- list(
  asst_top30 = as.integer(asst_top30),
  asst_bot30 = as.integer(asst_bot30),
  pair_info  = pair_diff %>% mutate(block = as.integer(block)) %>% as.data.frame()
)
write(toJSON(out, pretty = TRUE, auto_unbox = TRUE),
      "psychometrics/gfc_tirt/ablation_assistant_subsets.json")
message("\nWrote psychometrics/gfc_tirt/ablation_assistant_subsets.json")
