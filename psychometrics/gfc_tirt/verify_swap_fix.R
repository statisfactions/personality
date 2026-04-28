#!/usr/bin/env Rscript
# Verify the swap-aware fix on Haiku honest single-model fit.
# Runs both BUGGY (no swap correction) and FIXED versions of the Stan fit
# on the same data, prints recovery side-by-side.

suppressMessages({
  library(rstan)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})
rstan_options(auto_write = TRUE)
options(mc.cores = 4)

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

raw <- fromJSON("psychometrics/gfc_tirt/claude-haiku-4-5-20251001_gfc30_synthetic.json",
                flatten = TRUE)
df <- as_tibble(raw$results) %>%
  filter(!is.na(response_argmax), persona_id %in% paste0("s", 1:50)) %>%
  mutate(response_raw = as.integer(response_argmax),
         response_fixed = ifelse(swapped, 8L - response_raw, response_raw))

cat(sprintf("Rows: %d  swapped: %d (%.1f%%)\n",
            nrow(df), sum(df$swapped), 100*mean(df$swapped)))

run_fit <- function(df, response_col, label) {
  cat(sprintf("\n=== %s ===\n", label))
  wide <- df %>% select(persona_id, block, all_of(response_col)) %>%
    rename(response = !!response_col) %>%
    pivot_wider(names_from = block, values_from = response, names_prefix = "b") %>%
    arrange(match(persona_id, paste0("s", 1:50)))
  keep <- complete.cases(wide[, -1])
  wide <- wide[keep, ]
  N <- nrow(wide)
  y_mat <- as.matrix(wide[, paste0("b", seq_len(P))]); mode(y_mat) <- "integer"

  stan_data <- list(N=N, P=P, J=J, D=D, K=K,
                    trait=stmt_df$trait_id, g=stmt_df$g,
                    L=L_idx, R=R_idx, y=y_mat)
  fit <- sampling(model, data=stan_data, chains=4, iter=1000, warmup=333,
                  control=list(adapt_delta=0.95, max_treedepth=12),
                  refresh=0, pars=c("theta"))
  theta_mean <- apply(rstan::extract(fit, "theta")$theta, c(2, 3), mean)
  colnames(theta_mean) <- trait_names
  scored <- as_tibble(theta_mean) %>%
    rename_with(~paste0(.x, "_hat"), all_of(trait_names)) %>%
    bind_cols(persona_id = wide$persona_id) %>%
    inner_join(gt, by="persona_id")
  cat("Recovery (Pearson r):\n")
  for (t in trait_names) {
    r <- cor(scored[[paste0(t, "_hat")]], scored[[t]])
    cat(sprintf("  %s: %+.3f\n", t, r))
  }
  cat(sprintf("  mean |r|: %.3f\n",
              mean(abs(sapply(trait_names, function(t)
                cor(scored[[paste0(t, "_hat")]], scored[[t]]))))))
  invisible(scored)
}

run_fit(df, "response_raw",   "BUGGY: ignore swapped (current behavior)")
run_fit(df, "response_fixed", "FIXED: 8 - response when swapped")
