#!/usr/bin/env Rscript
# Per-model TIRT fit, pooling HONEST + FAKE-GOOD conditions (N=100 per model).
# Tests the "joint-condition fitting" hypothesis in isolation from
# cross-model pooling (which hurt because of model heterogeneity).
#
# Usage:
#   Rscript psychometrics/gfc_tirt/fit_tirt_per_model_pooled_conditions.R [output_dir]

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[1] else "psychometrics/gfc_tirt/per_model_pooled"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

stan_file <- "psychometrics/gfc_tirt/tirt_okada_indep.stan"
model <- stan_model(stan_file)

trait_names <- c("A", "C", "E", "N", "O")
trait_idx <- setNames(seq_along(trait_names), trait_names)

inst <- fromJSON("instruments/okada_gfc30.json")
pairs <- inst$pairs
P <- nrow(pairs); D <- 5L; K <- 7L

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

personas_json <- fromJSON("instruments/synthetic_personas.json")
gt <- tibble(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O
)

MODELS <- list(
  "Haiku 4.5"    = "claude-haiku-4-5-20251001",
  "Gemma3-4B"    = "gemma3-4b",
  "Gemma3-27B"   = "gemma3-27b",
  "Qwen2.5-3B"   = "qwen2.5-3b",
  "Phi4-mini"    = "phi4-mini",
  "Llama3.2-3B"  = "llama3.2-3b",
  "Llama3.3-70B" = "llama3.3-70b"
)

load_one <- function(slug, condition) {
  path <- switch(condition,
    honest   = sprintf("psychometrics/gfc_tirt/%s_gfc30_synthetic.json", slug),
    fakegood = sprintf("psychometrics/gfc_tirt/%s_gfc30_synthetic-fakegood.json", slug)
  )
  if (!file.exists(path)) return(NULL)
  raw <- fromJSON(path, flatten = TRUE)
  as_tibble(raw$results) %>%
    filter(!is.na(response_argmax)) %>%
    mutate(response_raw = as.integer(response_argmax),
           # Inference randomized L/R per prompt; un-swap so that response
           # is always in instrument-canonical L/R coordinates (Stan stmt_df
           # is built from canonical pairs$left / pairs$right).
           response = ifelse(swapped, 8L - response_raw, response_raw),
           condition = condition)
}

target_persona_ids <- paste0("s", seq_len(50))

all_recovery <- list()

for (display_name in names(MODELS)) {
  slug <- MODELS[[display_name]]
  cat(sprintf("\n=== %s (%s) ===\n", display_name, slug))

  cache_file <- file.path(output_dir, sprintf("%s_pooled_conditions_fit.rds", slug))
  if (file.exists(cache_file)) {
    cat("  Cached fit found, loading recovery only\n")
    cached <- readRDS(cache_file)
    all_recovery[[display_name]] <- cached$recovery
    next
  }

  honest <- load_one(slug, "honest")
  fakegood <- load_one(slug, "fakegood")
  if (is.null(honest) || is.null(fakegood)) {
    cat("  Missing data, skipping\n")
    next
  }

  long <- bind_rows(honest, fakegood) %>%
    filter(persona_id %in% target_persona_ids)
  wide <- long %>%
    select(condition, persona_id, block, response) %>%
    pivot_wider(names_from = block, values_from = response, names_prefix = "b") %>%
    filter(complete.cases(across(starts_with("b"))))

  N <- nrow(wide)
  y_mat <- as.matrix(wide[, paste0("b", seq_len(P))]); mode(y_mat) <- "integer"
  cat(sprintf("  N=%d (%d honest + %d fakegood)\n", N,
              sum(wide$condition == "honest"), sum(wide$condition == "fakegood")))

  stan_data <- list(N=N, P=P, J=J, D=D, K=K,
                    trait=stmt_df$trait_id, g=stmt_df$g,
                    L=L_idx, R=R_idx, y=y_mat)

  fit <- sampling(model, data = stan_data,
                  chains = 4, iter = 1000, warmup = 333,
                  control = list(adapt_delta = 0.95, max_treedepth = 12),
                  refresh = 0,
                  pars = c("theta", "a_pos", "kappa"))

  theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
  colnames(theta_mean) <- trait_names

  scored <- as_tibble(theta_mean) %>%
    rename_with(~ paste0(.x, "_hat"), all_of(trait_names)) %>%
    bind_cols(wide %>% select(condition, persona_id)) %>%
    inner_join(gt, by = "persona_id", suffix = c("_hat", "_true"))

  rec <- scored %>%
    group_by(condition) %>%
    summarise(
      A = cor(A_hat, A), C = cor(C_hat, C),
      E = cor(E_hat, E), N = cor(N_hat, N), O = cor(O_hat, O),
      n = n(), .groups = "drop"
    ) %>%
    mutate(model = display_name) %>%
    select(model, condition, n, A, C, E, N, O)

  print(rec)
  cat(sprintf("  mean |r| (honest):    %.3f\n",
              rec %>% filter(condition == "honest") %>%
                summarise(m = mean(abs(c(A, C, E, N, O)))) %>% pull(m)))
  cat(sprintf("  mean |r| (fakegood):  %.3f\n",
              rec %>% filter(condition == "fakegood") %>%
                summarise(m = mean(abs(c(A, C, E, N, O)))) %>% pull(m)))

  diagnostics <- rstan::summary(fit)$summary
  high_rhat <- sum(diagnostics[, "Rhat"] > 1.05, na.rm = TRUE)
  cat(sprintf("  Convergence: %d Rhat>1.05 (of %d)\n",
              high_rhat, nrow(diagnostics)))

  saveRDS(list(fit = fit, scored = scored, recovery = rec),
          file.path(output_dir, sprintf("%s_pooled_conditions_fit.rds", slug)))
  all_recovery[[display_name]] <- rec
}

combined <- bind_rows(all_recovery)
saveRDS(combined, file.path(output_dir, "all_recovery.rds"))

cat("\n\n=== Combined recovery (per model × condition, joint H+FG fit) ===\n")
print(combined, n = 100)

cat("\n=== Mean |r| summary ===\n")
print(combined %>%
        rowwise() %>%
        mutate(mean_abs_r = mean(abs(c(A, C, E, N, O))),
               mean_signed_r = mean(c(A, C, E, N, O))) %>%
        ungroup() %>%
        select(model, condition, n, mean_signed_r, mean_abs_r),
      n = 100)
