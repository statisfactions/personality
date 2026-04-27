library(thurstonianIRT)
library(tidyverse)

# Load instrument metadata
inst <- jsonlite::fromJSON("instruments/okada_gfc30.json")
pairs <- inst$pairs

# Load ground-truth persona z-scores
personas_json <- jsonlite::fromJSON("instruments/synthetic_personas.json")
ground_truth <- tibble(
  persona_id = personas_json$personas$persona_id,
  A_true = personas_json$personas$z_scores$A,
  C_true = personas_json$personas$z_scores$C,
  E_true = personas_json$personas$z_scores$E,
  N_true = personas_json$personas$z_scores$N,
  O_true = personas_json$personas$z_scores$O
)

# Build blocks for thurstonianIRT
block_df <- tibble(
  block = rep(seq_len(nrow(pairs)), each = 2),
  item = character(nrow(pairs) * 2),
  trait = character(nrow(pairs) * 2),
  sign = numeric(nrow(pairs) * 2)
)
for (i in seq_len(nrow(pairs))) {
  left_name <- paste0(pairs$left$trait[i],
                       ifelse(pairs$left$keying[i] == "+", "p", "n"), i)
  right_name <- paste0(pairs$right$trait[i],
                        ifelse(pairs$right$keying[i] == "+", "p", "n"), i)
  idx <- (i - 1) * 2
  block_df$item[idx + 1] <- left_name
  block_df$trait[idx + 1] <- pairs$left$trait[i]
  block_df$sign[idx + 1] <- ifelse(pairs$left$keying[i] == "+", 1, -1)
  block_df$item[idx + 2] <- right_name
  block_df$trait[idx + 2] <- pairs$right$trait[i]
  block_df$sign[idx + 2] <- ifelse(pairs$right$keying[i] == "+", 1, -1)
}
blocks <- set_blocks_from_df(block_df)

# Load response data (wide format) — Gemma3 12B synthetic personas
wide <- read_csv("results/gemma3-12b_gfc30_synthetic_wide.csv", show_col_types = FALSE)
cat("Response matrix:", nrow(wide), "respondents x", ncol(wide)-1, "blocks\n")

# Build pairwise data frame with proper column names
pairwise_data <- data.frame(person = seq_len(nrow(wide)))
for (i in seq_len(nrow(pairs))) {
  left_name <- paste0(pairs$left$trait[i],
                       ifelse(pairs$left$keying[i] == "+", "p", "n"), i)
  right_name <- paste0(pairs$right$trait[i],
                        ifelse(pairs$right$keying[i] == "+", "p", "n"), i)
  col_name <- paste0(left_name, right_name)
  pairwise_data[[col_name]] <- wide[[paste0("b", i)]]
}

complete_mask <- complete.cases(pairwise_data[, -1])
pairwise_data <- pairwise_data[complete_mask, ]
pairwise_data$person <- seq_len(nrow(pairwise_data))
cat("Complete respondents:", nrow(pairwise_data), "\n\n")

# Store persona IDs for matching back to ground truth
persona_ids <- wide$persona_id[complete_mask]

# Create TIRT data
tirt_data <- make_TIRT_data(
  data = pairwise_data,
  blocks = blocks,
  direction = "larger",
  format = "pairwise",
  family = "cumulative",
  range = c(1, 7)
)
cat("TIRT data created:", nrow(tirt_data), "x", ncol(tirt_data), "\n\n")

# Fit Stan TIRT — 400 respondents, 2 chains x 1000 iter
cat("Fitting TIRT Stan model (cumulative family, Gemma3 12B, 400 respondents)...\n")
cat("Using 2 chains x 1000 iter (500 warmup)\n\n")

fit <- fit_TIRT_stan(
  tirt_data,
  chains = 2,
  iter = 1000,
  warmup = 500,
  cores = 2
)

cat("\n=== Stan TIRT Fit Complete ===\n\n")

# Save fit object
saveRDS(fit, "results/gemma3_gfc30_synthetic_tirt_fit.rds")
cat("Fit saved to results/gemma3_gfc30_synthetic_tirt_fit.rds\n\n")

# Extract trait scores
traits <- c("A", "C", "E", "N", "O")
scores <- predict(fit)
cat("Score dimensions:", nrow(scores), "x", ncol(scores), "\n")
cat("Score columns:", paste(colnames(scores), collapse = ", "), "\n\n")

# Match scores to ground truth
# predict() returns long format: N_persons * N_traits rows
score_df <- as_tibble(scores) %>%
  pivot_wider(names_from = trait, values_from = c(estimate, se, lower_ci, upper_ci),
              names_glue = "{trait}_{.value}") %>%
  mutate(persona_id = persona_ids) %>%
  left_join(ground_truth, by = "persona_id")

# Ground-truth recovery
cat("=== Ground-Truth Recovery (TIRT Stan scores) ===\n")
for (t in traits) {
  tirt_col <- paste0(t, "_estimate")
  true_col <- paste0(t, "_true")
  if (tirt_col %in% names(score_df)) {
    r <- cor(score_df[[tirt_col]], score_df[[true_col]], use = "complete.obs")
    cat(sprintf("  %s: r = %.3f\n", t, r))
  } else {
    cat(sprintf("  %s: column '%s' not found in scores\n", t, tirt_col))
  }
}

# Simple scoring for comparison
cat("\n=== Simple Scoring (for comparison) ===\n")
raw <- jsonlite::fromJSON("results/gemma3-12b_gfc30_synthetic.json", flatten = TRUE)
all_results <- as_tibble(raw$results)

trait_evidence <- all_results %>%
  filter(!is.na(response_argmax)) %>%
  mutate(
    resp_val = coalesce(as.numeric(response_ev), as.numeric(response_argmax)),
    pref = resp_val - 4,
    right_endorsement = pref * ifelse(right_keying == "+", 1, -1),
    left_endorsement = -pref * ifelse(left_keying == "+", 1, -1)
  )

simple_scores <- bind_rows(
  trait_evidence %>% select(persona_id, trait = right_trait, endorsement = right_endorsement),
  trait_evidence %>% select(persona_id, trait = left_trait, endorsement = left_endorsement)
) %>%
  group_by(persona_id, trait) %>%
  summarise(score = mean(endorsement, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = trait, values_from = score) %>%
  left_join(ground_truth, by = "persona_id")

cat("Ground-Truth Recovery (simple scoring):\n")
for (t in traits) {
  r <- cor(simple_scores[[t]], simple_scores[[paste0(t, "_true")]], use = "complete.obs")
  cat(sprintf("  %s: r = %.3f\n", t, r))
}

# Trait correlations from TIRT scores
cat("\n=== Estimated Trait Correlations (TIRT) ===\n")
tirt_cols <- paste0(traits, "_estimate")
tirt_cols_present <- tirt_cols[tirt_cols %in% names(score_df)]
if (length(tirt_cols_present) > 0) {
  print(round(cor(score_df[tirt_cols_present], use = "complete.obs"), 3))
}

cat("\nDone.\n")
