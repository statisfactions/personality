library(thurstonianIRT)
library(tidyverse)

# Load saved fit
fit <- readRDS("results/gemma2_gfc30_synthetic_tirt_fit.rds")

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

# Load wide CSV to get persona order
wide <- read_csv("results/gemma2-9b_gfc30_synthetic_wide.csv", show_col_types = FALSE)
persona_ids <- wide$persona_id

# Extract scores — predict returns long format (id, trait, estimate, ...)
scores <- predict(fit)
cat("Score dimensions:", nrow(scores), "x", ncol(scores), "\n")
cat("Score columns:", paste(colnames(scores), collapse=", "), "\n\n")
print(head(scores, 10))

# The scores are in long format: 400 persons × 5 traits = 2000 rows
# Pivot to wide
scores_wide <- scores %>%
  as_tibble() %>%
  select(id, trait, estimate) %>%
  pivot_wider(names_from = trait, values_from = estimate, names_prefix = "tirt_")

cat("\nScores wide:", nrow(scores_wide), "x", ncol(scores_wide), "\n")

# Map id back to persona_id
scores_wide <- scores_wide %>%
  mutate(persona_id = persona_ids[id]) %>%
  left_join(ground_truth, by = "persona_id")

# Ground-truth recovery
traits <- c("A", "C", "E", "N", "O")
cat("\n=== Ground-Truth Recovery (TIRT Stan scores) ===\n")
for (t in traits) {
  tirt_col <- paste0("tirt_", t)
  true_col <- paste0(t, "_true")
  if (tirt_col %in% names(scores_wide)) {
    r <- cor(scores_wide[[tirt_col]], scores_wide[[true_col]], use = "complete.obs")
    cat(sprintf("  %s: r = %.3f\n", t, r))
  } else {
    cat(sprintf("  %s: column '%s' not found\n", t, tirt_col))
  }
}

# Trait correlations from TIRT scores
cat("\n=== Estimated Trait Correlations (TIRT) ===\n")
tirt_cols <- paste0("tirt_", traits)
tirt_present <- tirt_cols[tirt_cols %in% names(scores_wide)]
if (length(tirt_present) > 0) {
  print(round(cor(scores_wide[tirt_present], use = "complete.obs"), 3))
}

# TIRT score summary
cat("\n=== TIRT Score Summary ===\n")
for (t in traits) {
  col <- paste0("tirt_", t)
  if (col %in% names(scores_wide)) {
    cat(sprintf("  %s: mean=%.3f  sd=%.3f  range=[%.3f, %.3f]\n",
                t, mean(scores_wide[[col]], na.rm=TRUE),
                sd(scores_wide[[col]], na.rm=TRUE),
                min(scores_wide[[col]], na.rm=TRUE),
                max(scores_wide[[col]], na.rm=TRUE)))
  }
}

# Simple scoring for comparison
cat("\n=== Simple Scoring (for comparison) ===\n")
raw <- jsonlite::fromJSON("results/gemma2-9b_gfc30_synthetic.json", flatten = TRUE)
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

# Side-by-side comparison
cat("\n=== TIRT vs Simple Scoring Recovery ===\n")
cat(sprintf("  %-6s %8s %8s\n", "Trait", "TIRT", "Simple"))
for (t in traits) {
  tirt_col <- paste0("tirt_", t)
  r_tirt <- if (tirt_col %in% names(scores_wide))
    cor(scores_wide[[tirt_col]], scores_wide[[paste0(t, "_true")]], use = "complete.obs")
  else NA
  r_simple <- cor(simple_scores[[t]], simple_scores[[paste0(t, "_true")]], use = "complete.obs")
  cat(sprintf("  %-6s %8.3f %8.3f\n", t, r_tirt, r_simple))
}

cat("\nDone.\n")
