#!/usr/bin/env Rscript
# Pooled Okada-style TIRT fit across multiple LLMs and instruction conditions.
#
# Each "respondent" row in the fit is one (model, persona, condition) combo.
# This mirrors Okada Appendix D: "we fit a single pooled IRT model across
# all LLMs and both instruction conditions". With ~5 models × (50 honest +
# 50 fake-good + 1 bare + 1 respondent) ≈ 510 rows, the pooled likelihood
# resolves the per-trait sign-rotation ambiguity that single-condition,
# single-model fits cannot.
#
# Usage:
#   Rscript psychometrics/gfc_tirt/fit_tirt_pooled.R psychometrics/gfc_tirt/pooled_tirt_fit.rds
#
# Stan model: psychometrics/gfc_tirt/tirt_okada_indep.stan (Okada-exact priors, indep θ).
#
# Inputs (loaded automatically if present):
#   psychometrics/gfc_tirt/<model_slug>_gfc30_synthetic.json           — persona + honest
#   psychometrics/gfc_tirt/<model_slug>_gfc30_synthetic-fakegood.json  — persona + fake-good
#   psychometrics/gfc_tirt/<model_slug>_gfc30_neutral-bare.json        — bare
#   psychometrics/gfc_tirt/<model_slug>_gfc30_neutral-respondent.json  — respondent
#
# Models scanned: claude-haiku-4-5-20251001, gemma3-4b, qwen2.5-3b,
# phi4-mini, llama3.2-3b. (phi4-mini-latest also accepted as alias.)

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(purrr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1) args[1] else "psychometrics/gfc_tirt/pooled_tirt_fit.rds"
n_personas  <- if (length(args) >= 2) as.integer(args[2]) else 50L
n_iter      <- if (length(args) >= 3) as.integer(args[3]) else 1500L
n_chains    <- if (length(args) >= 4) as.integer(args[4]) else 4L

stan_file <- "psychometrics/gfc_tirt/tirt_okada_indep.stan"
if (!file.exists(stan_file)) stop("Stan file missing: ", stan_file)

# Models to scan (Ollama / Anthropic slug). Order matters only for reporting.
MODELS <- c(
  "claude-haiku-4-5-20251001" = "Haiku 4.5",
  "gemma3-4b"                 = "Gemma3-4B",
  "qwen2.5-3b"                = "Qwen2.5-3B",
  "phi4-mini"                 = "Phi4-mini",
  "phi4-mini-latest"          = "Phi4-mini",  # alias for neutral files
  "llama3.2-3b"               = "Llama3.2-3B"
)

# --- Instrument metadata ---
inst <- fromJSON("instruments/okada_gfc30.json")
pairs <- inst$pairs
P <- nrow(pairs)
D <- 5L
K <- 7L
trait_names <- c("A", "C", "E", "N", "O")
trait_idx <- setNames(seq_along(trait_names), trait_names)

stmt_df <- tibble(
  block = rep(seq_len(P), each = 2),
  side  = rep(c("L", "R"), times = P),
  trait = c(rbind(pairs$left$trait,   pairs$right$trait)),
  key   = c(rbind(pairs$left$keying,  pairs$right$keying))
) %>%
  mutate(g = ifelse(key == "+", 1L, -1L),
         trait_id = trait_idx[trait])
stmt_df$stmt_index <- seq_len(nrow(stmt_df))
J <- nrow(stmt_df)
L_idx <- stmt_df$stmt_index[stmt_df$side == "L"]
R_idx <- stmt_df$stmt_index[stmt_df$side == "R"]

# --- Load all available response files ---
load_file <- function(path, model_slug, condition) {
  if (!file.exists(path)) return(NULL)
  raw <- fromJSON(path, flatten = TRUE)
  rs <- as_tibble(raw$results) %>%
    filter(!is.na(response_argmax)) %>%
    mutate(response_raw = as.integer(response_argmax),
           # Inference randomized L/R per prompt; un-swap so response is in
           # instrument-canonical L/R coords (matches stmt_df construction).
           response = ifelse(swapped, 8L - response_raw, response_raw),
           model    = model_slug,
           condition = condition)
  rs %>% select(model, condition, persona_id, block, response)
}

all_rows <- list()
for (slug in names(MODELS)) {
  for (cond in c("honest", "fakegood", "bare", "respondent")) {
    path <- switch(cond,
      honest     = sprintf("psychometrics/gfc_tirt/%s_gfc30_synthetic.json", slug),
      fakegood   = sprintf("psychometrics/gfc_tirt/%s_gfc30_synthetic-fakegood.json", slug),
      bare       = sprintf("psychometrics/gfc_tirt/%s_gfc30_neutral-bare.json", slug),
      respondent = sprintf("psychometrics/gfc_tirt/%s_gfc30_neutral-respondent.json", slug)
    )
    df <- load_file(path, MODELS[[slug]], cond)
    if (!is.null(df)) {
      n_rows <- length(unique(df$persona_id))
      message(sprintf("  loaded %s/%s: %d rows × %d blocks (%s)",
                      MODELS[[slug]], cond, n_rows, P, basename(path)))
      all_rows[[length(all_rows) + 1]] <- df
    }
  }
}

if (length(all_rows) == 0) stop("No response files found.")
long <- bind_rows(all_rows)

# Subset persona-condition rows to first n_personas (s1..s50)
target_persona_ids <- paste0("s", seq_len(n_personas))
long <- long %>%
  filter(condition %in% c("bare", "respondent") |
         persona_id %in% target_persona_ids)

# Build wide matrix; one row = (model, condition, persona_id)
wide <- long %>%
  pivot_wider(names_from = block, values_from = response,
              names_prefix = "b") %>%
  filter(complete.cases(across(starts_with("b"))))

N <- nrow(wide)
message(sprintf("\nPooled response matrix: N=%d rows × P=%d blocks", N, P))
message("\nRows by (model, condition):")
print(wide %>% count(model, condition) %>% arrange(model, condition))

y_mat <- as.matrix(wide[, paste0("b", seq_len(P))])
mode(y_mat) <- "integer"
stopifnot(all(y_mat >= 1 & y_mat <= K))

# --- Ground truth (only for persona conditions) ---
personas_json <- fromJSON("instruments/synthetic_personas.json")
gt <- tibble(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O
)

# --- Stan data ---
stan_data <- list(
  N = N, P = P, J = J, D = D, K = K,
  trait = stmt_df$trait_id,
  g     = stmt_df$g,
  L     = L_idx, R = R_idx,
  y     = y_mat
)

message("\nCompiling Stan model...")
model <- stan_model(stan_file)

message(sprintf("Sampling: %d chains × %d iter (%d warmup), N=%d",
                n_chains, n_iter, n_iter %/% 3, N))
fit <- sampling(
  model, data = stan_data,
  chains = n_chains, iter = n_iter, warmup = n_iter %/% 3,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  refresh = 100,
  pars = c("theta", "a_pos", "kappa")
)

theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
colnames(theta_mean) <- trait_names

# Attach θ̂ to the wide table
theta_df <- as_tibble(theta_mean) %>%
  rename_with(~ paste0(.x, "_hat"), all_of(trait_names)) %>%
  bind_cols(wide %>% select(model, condition, persona_id))

# --- Per-(model, condition) recovery vs ground truth ---
recovery <- theta_df %>%
  filter(condition %in% c("honest", "fakegood")) %>%
  inner_join(gt, by = "persona_id", suffix = c("_hat", "_true")) %>%
  group_by(model, condition) %>%
  summarise(
    A = cor(A_hat, A),
    C = cor(C_hat, C),
    E = cor(E_hat, E),
    N = cor(N_hat, N),
    O = cor(O_hat, O),
    n_personas = n(),
    .groups = "drop"
  )

message("\n=== Per-(model, condition) recovery vs ground truth (Pearson r) ===")
print(recovery, n = 100)

# Mean |r| per row
recovery_summary <- recovery %>%
  rowwise() %>%
  mutate(mean_abs_r = mean(abs(c(A, C, E, N, O))),
         mean_signed_r = mean(c(A, C, E, N, O))) %>%
  ungroup()
message("\n=== Summary by (model, condition) ===")
print(recovery_summary %>% select(model, condition, n_personas,
                                  mean_signed_r, mean_abs_r), n = 100)

# --- Neutral placement ---
neutral <- theta_df %>%
  filter(condition %in% c("bare", "respondent")) %>%
  select(model, condition, all_of(paste0(trait_names, "_hat")))
message("\n=== Neutral placement (model-default θ̂ on persona-fitted scale) ===")
print(neutral, n = 100)

# --- Save outputs ---
out <- list(
  fit = fit,
  theta_df = theta_df,
  recovery = recovery,
  recovery_summary = recovery_summary,
  neutral = neutral,
  wide_meta = wide %>% select(model, condition, persona_id),
  N = N
)
saveRDS(out, output_path)
message("\nSaved: ", output_path)

diagnostics <- rstan::summary(fit)$summary
high_rhat <- sum(diagnostics[, "Rhat"] > 1.05, na.rm = TRUE)
low_neff  <- sum(diagnostics[, "n_eff"] < 100, na.rm = TRUE)
message(sprintf("\nConvergence: %d Rhat>1.05, %d n_eff<100 (out of %d params)",
                high_rhat, low_neff, nrow(diagnostics)))
