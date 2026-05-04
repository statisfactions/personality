#!/usr/bin/env Rscript
# Single-model binary TIRT fit. Loads the four-condition input set for one
# model and fits tirt_okada_binary.stan jointly across honest + fakegood
# (+ bare + respondent if present, projected onto the same θ space).
#
# Usage:
#   Rscript psychometrics/gfc_tirt/fit_tirt_single_binary.R <model_slug> \
#       [output_path] [n_personas] [n_iter] [n_chains]
#
# Examples:
#   Rscript psychometrics/gfc_tirt/fit_tirt_single_binary.R \
#       claude-haiku-4-5-20251001
#   Rscript psychometrics/gfc_tirt/fit_tirt_single_binary.R gemma3-27b

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: fit_tirt_single_binary.R <model_slug> [out] [n_personas] [n_iter] [n_chains]")
}
model_slug  <- args[1]
output_path <- if (length(args) >= 2) args[2] else sprintf("psychometrics/gfc_tirt/single_binary_%s_fit.rds", model_slug)
n_personas  <- if (length(args) >= 3) as.integer(args[3]) else 50L
n_iter      <- if (length(args) >= 4) as.integer(args[4]) else 1500L
n_chains    <- if (length(args) >= 5) as.integer(args[5]) else 4L

stan_file <- "psychometrics/gfc_tirt/tirt_okada_binary.stan"

inst <- fromJSON("instruments/okada_gfc30.json")
pairs <- inst$pairs
P <- nrow(pairs); D <- 5L
trait_names <- c("A", "C", "E", "N", "O")
trait_idx <- setNames(seq_along(trait_names), trait_names)

stmt_df <- tibble(
  block = rep(seq_len(P), each = 2),
  side  = rep(c("L", "R"), times = P),
  trait = c(rbind(pairs$left$trait, pairs$right$trait)),
  key   = c(rbind(pairs$left$keying, pairs$right$keying))
) %>% mutate(g = ifelse(key == "+", 1L, -1L), trait_id = trait_idx[trait])
stmt_df$stmt_index <- seq_len(nrow(stmt_df))
J <- nrow(stmt_df)
L_idx <- stmt_df$stmt_index[stmt_df$side == "L"]
R_idx <- stmt_df$stmt_index[stmt_df$side == "R"]

load_file <- function(path, condition) {
  if (!file.exists(path)) return(NULL)
  raw <- fromJSON(path, flatten = TRUE)
  as_tibble(raw$results) %>%
    filter(!is.na(response_argmax)) %>%
    mutate(response_raw = as.integer(response_argmax),
           response = ifelse(swapped, 1L - response_raw, response_raw),
           condition = condition) %>%
    select(condition, persona_id, block, response)
}

paths <- list(
  honest     = sprintf("psychometrics/gfc_tirt/%s_gfc30_synthetic_binary.json", model_slug),
  fakegood   = sprintf("psychometrics/gfc_tirt/%s_gfc30_synthetic-fakegood_binary.json", model_slug),
  bare       = sprintf("psychometrics/gfc_tirt/%s_gfc30_neutral-bare_binary.json", model_slug),
  respondent = sprintf("psychometrics/gfc_tirt/%s_gfc30_neutral-respondent_binary.json", model_slug)
)

all_rows <- list()
for (cond in names(paths)) {
  df <- load_file(paths[[cond]], cond)
  if (!is.null(df)) {
    message(sprintf("  loaded %s: %d rows (%s)",
                    cond, nrow(df), basename(paths[[cond]])))
    all_rows[[length(all_rows) + 1]] <- df
  } else {
    message(sprintf("  missing %s: %s", cond, paths[[cond]]))
  }
}
if (length(all_rows) == 0) stop("No response files found for ", model_slug)
long <- bind_rows(all_rows)

target_persona_ids <- paste0("s", seq_len(n_personas))
long <- long %>%
  filter(condition %in% c("bare", "respondent") |
         persona_id %in% target_persona_ids)

wide <- long %>%
  pivot_wider(names_from = block, values_from = response, names_prefix = "b") %>%
  filter(complete.cases(across(starts_with("b"))))

N <- nrow(wide)
message(sprintf("\nResponse matrix: N=%d rows × P=%d blocks", N, P))
print(wide %>% count(condition))

y_mat <- as.matrix(wide[, paste0("b", seq_len(P))])
mode(y_mat) <- "integer"
stopifnot(all(y_mat %in% c(0L, 1L)))

personas_json <- fromJSON("instruments/synthetic_personas.json")
gt <- tibble(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O
)

stan_data <- list(N=N, P=P, J=J, D=D,
                  trait=stmt_df$trait_id, g=stmt_df$g,
                  L=L_idx, R=R_idx, y=y_mat)

message("\nCompiling Stan model...")
model <- stan_model(stan_file)

message(sprintf("Sampling: %d chains × %d iter (%d warmup), N=%d",
                n_chains, n_iter, n_iter %/% 3, N))
fit <- sampling(model, data = stan_data,
                chains = n_chains, iter = n_iter, warmup = n_iter %/% 3,
                control = list(adapt_delta = 0.95, max_treedepth = 12),
                refresh = 100,
                pars = c("theta", "a_pos"))

theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
colnames(theta_mean) <- trait_names

theta_df <- as_tibble(theta_mean) %>%
  rename_with(~ paste0(.x, "_hat"), all_of(trait_names)) %>%
  bind_cols(wide %>% select(condition, persona_id))

recovery <- theta_df %>%
  filter(condition %in% c("honest", "fakegood")) %>%
  inner_join(gt, by = "persona_id", suffix = c("_hat", "_true")) %>%
  group_by(condition) %>%
  summarise(
    A = cor(A_hat, A), C = cor(C_hat, C),
    E = cor(E_hat, E), N = cor(N_hat, N), O = cor(O_hat, O),
    n_personas = n(), .groups = "drop"
  )

message("\n=== Recovery vs ground truth (", model_slug, ") ===")
print(recovery, n = 100)

recovery_summary <- recovery %>%
  rowwise() %>%
  mutate(mean_abs_r = mean(abs(c(A, C, E, N, O))),
         mean_signed_r = mean(c(A, C, E, N, O))) %>%
  ungroup()
message("\n=== Summary ===")
print(recovery_summary %>% select(condition, n_personas,
                                  mean_signed_r, mean_abs_r), n = 100)

neutral <- theta_df %>%
  filter(condition %in% c("bare", "respondent")) %>%
  select(condition, all_of(paste0(trait_names, "_hat")))
if (nrow(neutral)) {
  message("\n=== Neutral placement ===")
  print(neutral, n = 100)
}

out <- list(model_slug = model_slug, fit = fit, theta_df = theta_df,
            recovery = recovery, recovery_summary = recovery_summary,
            neutral = neutral, N = N)
saveRDS(out, output_path)
message("\nSaved: ", output_path)

diagnostics <- rstan::summary(fit)$summary
high_rhat <- sum(diagnostics[, "Rhat"] > 1.05, na.rm = TRUE)
low_neff  <- sum(diagnostics[, "n_eff"] < 100, na.rm = TRUE)
message(sprintf("\nConvergence: %d Rhat>1.05, %d n_eff<100 (out of %d params)",
                high_rhat, low_neff, nrow(diagnostics)))
