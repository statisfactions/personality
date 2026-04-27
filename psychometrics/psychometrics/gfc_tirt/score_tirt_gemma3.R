library(thurstonianIRT)
library(tidyverse)

# Load saved TIRT fit
fit <- readRDS("results/gemma3_gfc30_synthetic_tirt_fit.rds")

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

# Load wide CSV to get persona_id order (after complete-case filtering)
wide <- read_csv("results/gemma3-12b_gfc30_synthetic_wide.csv", show_col_types = FALSE)
complete_mask <- complete.cases(wide[, -1])
persona_ids <- wide$persona_id[complete_mask]
cat("Complete respondents:", sum(complete_mask), "\n\n")

traits <- c("A", "C", "E", "N", "O")

# Extract trait scores — long format (N_persons * N_traits rows)
scores <- predict(fit)
cat("Score dimensions:", nrow(scores), "x", ncol(scores), "\n")
cat("Score columns:", paste(colnames(scores), collapse = ", "), "\n\n")

# Pivot to wide and attach persona IDs
score_df <- as_tibble(scores) %>%
  pivot_wider(names_from = trait, values_from = c(estimate, se, lower_ci, upper_ci),
              names_glue = "{trait}_{.value}") %>%
  mutate(persona_id = persona_ids) %>%
  left_join(ground_truth, by = "persona_id")

# Ground-truth recovery — TIRT
cat("=== Ground-Truth Recovery (TIRT Stan scores) ===\n")
for (t in traits) {
  tirt_col <- paste0(t, "_estimate")
  true_col <- paste0(t, "_true")
  r <- cor(score_df[[tirt_col]], score_df[[true_col]], use = "complete.obs")
  cat(sprintf("  %s: r = %.3f\n", t, r))
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
print(round(cor(score_df[tirt_cols], use = "complete.obs"), 3))

cat("\nDone.\n")
