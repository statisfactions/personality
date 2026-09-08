#!/usr/bin/env Rscript
# Gain model via per-frame lm() (2026-09-07):
#   for each (model, framing):  lm(ev ~ t)
#     intercept -> lambda_mf (t is centered, so intercept = frame mean)
#     slope     -> beta_mf, with PER-FRAME standard errors.
# Why not one joint lm(ev ~ 0 + framing + framing:t) per model: the design
# is balanced (same n and same t in every frame), so the joint fit's pooled
# sigma gives every frame IDENTICAL SEs — and the homoscedasticity it
# assumes is mildly false in a known direction (the assistant frame fits t
# worst; its residual SD runs ~1.10x direct's). Per-frame fits give each
# coefficient its own honest error. Estimates are unchanged either way.
# t (the trait template) is stage one — framing-average of per-(m,f)-
# centered profiles — because lambda + beta*t is bilinear.
# Input:  results/adjectives/self_ev_long.csv
# Output: results/adjectives/gain_model_R.csv
suppressMessages({library(dplyr); library(tidyr); library(tibble)})

d <- read.csv("results/adjectives/self_ev_long.csv", stringsAsFactors = FALSE)

d <- d |>
  group_by(model, framing) |>
  mutate(c = ev - mean(ev)) |>
  group_by(model, adjective) |>
  mutate(t = mean(c)) |>
  ungroup()

res <- d |>
  group_by(model, framing) |>
  group_modify(function(g, key) {
    m <- lm(ev ~ t, data = g)
    co <- summary(m)$coefficients
    tibble(lam = co[1, 1], lam_se = co[1, 2],
           beta = co[2, 1], beta_se = co[2, 2],
           r = cor(g$c, g$t),
           amp = sqrt(sum(g$c^2) / sum(g$t^2)),
           resid_se = summary(m)$sigma,
           r2 = summary(m)$r.squared)
  }) |>
  ungroup()

write.csv(res, "results/adjectives/gain_model_R.csv", row.names = FALSE)

cat("cohort means by framing (per-frame lm):\n")
res |> group_by(framing) |>
  summarise(beta = mean(beta), beta_se = mean(beta_se), r = mean(r),
            amp = mean(amp)) |>
  mutate(across(where(is.numeric), \(x) round(x, 3))) |> print(n = 6)

wide <- res |> select(model, framing, beta, beta_se) |>
  pivot_wider(names_from = framing, values_from = c(beta, beta_se))
z <- with(wide, (beta_direct - beta_assistant) /
                sqrt(beta_se_direct^2 + beta_se_assistant^2))
cat(sprintf("\nbeta_direct > beta_assistant at |z|>2: %d/%d models (z<-2: %d)\n",
            sum(z > 2), length(z), sum(z < -2)))
cat(sprintf("wrote gain_model_R.csv (%d rows)\n", nrow(res)))

# --verbose: per-model coefficient blocks for eyeballing
if ("--verbose" %in% commandArgs(trailingOnly = TRUE)) {
  out <- "results/adjectives/gain_model_fits.txt"
  sink(out)
  for (mm in sort(unique(res$model))) {
    rm_ <- res |> filter(model == mm)
    cat("\n", strrep("=", 72), "\n ", mm,
        sprintf("  (per-frame fits, n = 525 each; R^2 %.3f-%.3f)\n",
                min(rm_$r2), max(rm_$r2)))
    rm_ |> select(framing, lam, lam_se, beta, beta_se, r, amp, resid_se) |>
      mutate(across(where(is.numeric), \(x) round(x, 3))) |>
      as.data.frame() |> print(row.names = FALSE)
  }
  sink()
  cat(sprintf("verbose: wrote %s (%d models)\n", out,
              length(unique(res$model))))
}
