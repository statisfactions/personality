#!/usr/bin/env Rscript
# Compute cross-trait recovery matrices (theta_hat × ground_truth) from
# the FIXED per-model joint H+FG fits. Replicates §4b of the prior
# version of okada_pooled_replication.md.

suppressMessages({library(dplyr); library(tidyr)})

dir <- "psychometrics/gfc_tirt/per_model_pooled"
slugs <- c(
  "Haiku 4.5"   = "claude-haiku-4-5-20251001",
  "Gemma3-4B"   = "gemma3-4b",
  "Qwen2.5-3B"  = "qwen2.5-3b",
  "Phi4-mini"   = "phi4-mini",
  "Llama3.2-3B" = "llama3.2-3b"
)
traits <- c("A", "C", "E", "N", "O")

emit_matrix <- function(slug, display, condition) {
  rds <- file.path(dir, sprintf("%s_pooled_conditions_fit.rds", slug))
  if (!file.exists(rds)) return(invisible(NULL))
  fit <- readRDS(rds)
  scored <- fit$scored %>% filter(condition == !!condition)
  hat <- as.matrix(scored[, paste0(traits, "_hat")])
  tru <- as.matrix(scored[, traits])
  m <- cor(hat, tru)
  rownames(m) <- paste0("hat_", traits)
  colnames(m) <- paste0("true_", traits)
  cat(sprintf("\n=== %s | %s ===\n", display, condition))
  print(round(m, 2))
}

for (display in names(slugs)) {
  for (cond in c("honest", "fakegood")) {
    emit_matrix(slugs[[display]], display, cond)
  }
}
