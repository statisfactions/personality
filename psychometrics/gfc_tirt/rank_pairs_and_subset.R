#!/usr/bin/env Rscript
# Rank P=60 pairs by estimated informativeness, select three subsets
# (top30 / bot30 / rand30) for the ablation test.
#
# Pair info ~= a_L^2 + a_R^2 (Fisher-information proxy for TIRT).
# Loadings averaged across all 21 P=60 fits to reduce per-model noise.
#
# Output: psychometrics/gfc_tirt/ablation_subsets.json

suppressMessages({
  library(rstan)
  library(dplyr)
  library(jsonlite)
})

set.seed(20260516)

# --- collect a_pos posterior means per item, per P=60 fit ---
rds_files <- list.files(
  "psychometrics/gfc_tirt",
  pattern = "_ipipneogfc60_hf_.*_indep_fit\\.rds$",
  full.names = TRUE
)
message("P=60 fits: ", length(rds_files))

# Map j (interleaved L,R,L,R,...) -> (block, side).
# Block p has L at j=2p-1, R at j=2p.
j_to_block <- function(J) {
  data.frame(j = seq_len(J),
             block = (seq_len(J) + 1) %/% 2,
             side  = ifelse(seq_len(J) %% 2 == 1, "L", "R"))
}

per_item <- list()
for (f in rds_files) {
  fit_obj <- readRDS(f)
  a <- colMeans(rstan::extract(fit_obj$fit, pars = "a_pos")$a_pos)
  J <- length(a)
  m <- j_to_block(J)
  m$a <- a
  m$src <- basename(f)
  per_item[[length(per_item) + 1]] <- m
}
df <- do.call(rbind, per_item)

# Average a per (block, side) across all P=60 fits
avg <- df %>%
  group_by(block, side) %>%
  summarise(a_mean = mean(a), .groups = "drop")

# Per-block informativeness: a_L^2 + a_R^2
block_info <- avg %>%
  tidyr::pivot_wider(names_from = side, values_from = a_mean,
                     names_prefix = "a_") %>%
  mutate(info = a_L^2 + a_R^2,
         a_mean_pair = (a_L + a_R) / 2) %>%
  arrange(desc(info))

cat("\n=== Top 10 most informative pairs ===\n")
print(head(block_info, 10), digits = 3)
cat("\n=== Bottom 10 least informative pairs ===\n")
print(tail(block_info, 10), digits = 3)
cat("\nOverall info range: [", round(min(block_info$info), 3), ", ",
    round(max(block_info$info), 3), "]; median ", round(median(block_info$info), 3),
    "\n", sep = "")

# --- select subsets ---
top30  <- block_info$block[order(block_info$info, decreasing = TRUE)][1:30]
bot30  <- block_info$block[order(block_info$info, decreasing = FALSE)][1:30]
all60  <- block_info$block
rand30 <- sample(all60, 30)

cat("\nTop30 mean info:    ", round(mean(block_info$info[match(top30, block_info$block)]), 3), "\n")
cat("Bot30 mean info:    ", round(mean(block_info$info[match(bot30, block_info$block)]), 3), "\n")
cat("Rand30 mean info:   ", round(mean(block_info$info[match(rand30, block_info$block)]), 3), "\n")
cat("Full-60 mean info:  ", round(mean(block_info$info), 3), "\n")

# Check trait coverage (each subset should cover all 5 traits with both keying)
instrument <- fromJSON("instruments/ipip_neo_gfc_P60.json", flatten = FALSE)
pairs <- instrument$pairs

coverage <- function(blocks) {
  sel <- pairs[pairs$block %in% blocks, ]
  L_traits <- sel$left$trait;  L_keys <- sel$left$keying
  R_traits <- sel$right$trait; R_keys <- sel$right$keying
  traits <- c(L_traits, R_traits)
  keys   <- c(L_keys,   R_keys)
  tab <- table(traits, keys)
  tab
}

cat("\nTop30 trait coverage:\n");   print(coverage(top30))
cat("\nBot30 trait coverage:\n");   print(coverage(bot30))
cat("\nRand30 trait coverage:\n");  print(coverage(rand30))

# --- write subset definitions ---
out <- list(
  top30  = sort(as.integer(top30)),
  bot30  = sort(as.integer(bot30)),
  rand30 = sort(as.integer(rand30)),
  block_info = block_info %>%
    mutate(block = as.integer(block)) %>%
    as.data.frame()
)
write(toJSON(out, pretty = TRUE, auto_unbox = TRUE),
      "psychometrics/gfc_tirt/ablation_subsets.json")
message("\nWrote psychometrics/gfc_tirt/ablation_subsets.json")
