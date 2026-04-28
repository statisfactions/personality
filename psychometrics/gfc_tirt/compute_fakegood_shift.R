#!/usr/bin/env Rscript
# Compute fake-good vs honest shift in θ̂ from the FIXED per-model joint
# H+FG fits. Replicates §4 of okada_pooled_replication.md but with the
# swap bug corrected.
#
# Sign convention: positive = shifted toward socially desirable direction
# (after sign-correcting per-trait recovery and applying g_N = -1).

suppressMessages({library(dplyr); library(tidyr); library(purrr)})

dir <- "psychometrics/gfc_tirt/per_model_pooled"
slugs <- c(
  "Haiku 4.5"   = "claude-haiku-4-5-20251001",
  "Gemma3-4B"   = "gemma3-4b",
  "Qwen2.5-3B"  = "qwen2.5-3b",
  "Phi4-mini"   = "phi4-mini",
  "Llama3.2-3B" = "llama3.2-3b"
)
traits <- c("A", "C", "E", "N", "O")
g_trait <- c(A = +1, C = +1, E = +1, N = -1, O = +1)  # neuroticism: low = desirable

shift_one <- function(slug, display) {
  rds <- file.path(dir, sprintf("%s_pooled_conditions_fit.rds", slug))
  if (!file.exists(rds)) return(NULL)
  fit <- readRDS(rds)
  scored <- fit$scored
  rec <- fit$recovery %>% filter(condition == "honest") %>%
    select(all_of(traits)) %>% slice(1) %>% as.numeric()
  names(rec) <- traits
  # sign-correct: if r<0 between θ̂ and ground truth, flip θ̂ sign
  sgn <- sign(rec); sgn[sgn == 0] <- 1
  raw_shift <- scored %>%
    group_by(condition) %>%
    summarise(across(all_of(paste0(traits, "_hat")), mean), .groups = "drop")
  delta <- as.numeric(raw_shift[raw_shift$condition == "fakegood", paste0(traits, "_hat")]) -
           as.numeric(raw_shift[raw_shift$condition == "honest",   paste0(traits, "_hat")])
  names(delta) <- traits
  # Sign-correct + apply g_N
  signed_shift <- delta * sgn * g_trait
  tibble(model = display,
         A = signed_shift["A"], C = signed_shift["C"],
         E = signed_shift["E"], N = signed_shift["N"],
         O = signed_shift["O"],
         mean_SDR = mean(signed_shift),
         A_rec = rec["A"], C_rec = rec["C"], E_rec = rec["E"],
         N_rec = rec["N"], O_rec = rec["O"])
}

shifts <- map2_dfr(slugs, names(slugs), ~shift_one(.x, .y))

cat("\n=== Fake-good − honest θ̂ shift (sign-corrected) ===\n")
cat("Positive = shift toward socially desirable. From FIXED per-model joint H+FG fits.\n\n")
print(shifts %>% select(model, A, C, E, N, O, mean_SDR), digits = 3)

cat("\n=== Per-trait honest-condition recovery used for sign correction ===\n")
print(shifts %>% select(model, A_rec, C_rec, E_rec, N_rec, O_rec), digits = 3)

# Markdown table
cat("\n=== Markdown ===\n\n")
cat("| Model | A | C | E | N | O | mean SDR |\n")
cat("|-------|--:|--:|--:|--:|--:|---------:|\n")
for (i in seq_len(nrow(shifts))) {
  row <- shifts[i, ]
  cat(sprintf("| %s | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f | %+.2f |\n",
              row$model, row$A, row$C, row$E, row$N, row$O, row$mean_SDR))
}
