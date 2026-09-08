#!/usr/bin/env Rscript
# Gain model via lm() (2026-09-07): per model m,
#     lm(ev ~ 0 + framing + framing:t)
# framing dummies -> lambda_mf; framing:t slopes -> beta_mf (with SEs).
# t (the trait template) is stage-one: the framing-average of per-
# (model,framing)-centered profiles. It must be estimated first because
# lambda + beta*t is bilinear — lm() can fit the second stage only.
# r and amp (beta = r * amp) are computed alongside.
# Input:  results/adjectives/self_ev_long.csv
# Output: results/adjectives/gain_model_R.csv
suppressMessages(library(dplyr))

d <- read.csv("results/adjectives/self_ev_long.csv", stringsAsFactors = FALSE)

# stage one: the template
d <- d |>
  group_by(model, framing) |>
  mutate(c = ev - mean(ev)) |>
  group_by(model, adjective) |>
  mutate(t = mean(c)) |>
  ungroup()

fit_one <- function(dm) {
  m <- lm(ev ~ 0 + framing + framing:t, data = dm)
  co <- summary(m)$coefficients
  fr <- sub("^framing", "", grep("^framing[^:]+$", rownames(co), value = TRUE))
  lam <- co[paste0("framing", fr), "Estimate"]
  beta <- co[paste0("framing", fr, ":t"), "Estimate"]
  se <- co[paste0("framing", fr, ":t"), "Std. Error"]
  aux <- dm |> group_by(framing) |>
    summarise(r = cor(c, t), amp = sqrt(sum(c^2) / sum(t^2)), .groups = "drop")
  tibble(framing = fr, lam = unname(lam), beta = unname(beta),
         beta_se = unname(se)) |> left_join(aux, by = "framing")
}

res <- d |> group_by(model) |> group_modify(~ fit_one(.x)) |> ungroup()
write.csv(res, "results/adjectives/gain_model_R.csv", row.names = FALSE)

cat("cohort means by framing (lm):\n")
res |> group_by(framing) |>
  summarise(beta = mean(beta), se = mean(beta_se), r = mean(r),
            amp = mean(amp)) |>
  mutate(across(where(is.numeric), \(x) round(x, 3))) |> print(n = 6)

# is beta_assistant distinguishable from beta_direct, per model?
wide <- res |> select(model, framing, beta, beta_se) |>
  tidyr::pivot_wider(names_from = framing, values_from = c(beta, beta_se))
z <- with(wide, (beta_direct - beta_assistant) /
                sqrt(beta_se_direct^2 + beta_se_assistant^2))
cat(sprintf("\nbeta_direct > beta_assistant at |z|>2: %d/%d models (z<-2: %d)\n",
            sum(z > 2), length(z), sum(z < -2)))
cat(sprintf("wrote gain_model_R.csv (%d rows)\n", nrow(res)))

# --verbose: dump every per-model lm summary to a text file for eyeballing
if ("--verbose" %in% commandArgs(trailingOnly = TRUE)) {
  out <- "results/adjectives/gain_model_fits.txt"
  sink(out)
  for (mm in sort(unique(d$model))) {
    dm <- d[d$model == mm, ]
    fit <- lm(ev ~ 0 + framing + framing:t, data = dm)
    cat("\n", strrep("=", 72), "\n", mm,
        sprintf("   (n = %d, R^2 = %.3f, resid SE = %.3f)\n",
                nrow(dm), summary(fit)$r.squared, summary(fit)$sigma))
    print(summary(fit)$coefficients |> round(4))
  }
  sink()
  cat(sprintf("verbose: wrote %s (%d fits)\n", out, length(unique(d$model))))
}
