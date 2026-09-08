#!/usr/bin/env Rscript
# Gain model, R replication (2026-09-07): likert_mfa = lambda_mf + beta_mf * t_ma
#   t_m  = framing-average of per-(model,framing)-centered profiles
#   beta = slope of frame shape on template; r = cor; amp = norm ratio
# Input:  results/adjectives/self_ev_long.csv  (model, framing, adjective, ev)
# Output: results/adjectives/gain_model_R.csv  (compare vs gain_model.csv)
suppressMessages(library(dplyr))

d <- read.csv("results/adjectives/self_ev_long.csv", stringsAsFactors = FALSE)

res <- d |>
  group_by(model, framing) |>
  mutate(lam = mean(ev), c = ev - lam) |>
  group_by(model, adjective) |>
  mutate(t = mean(c)) |>
  group_by(model, framing) |>
  summarise(lam = first(lam),
            beta = sum(c * t) / sum(t * t),
            r = cor(c, t),
            amp = sqrt(sum(c^2) / sum(t^2)),
            .groups = "drop")

write.csv(res, "results/adjectives/gain_model_R.csv", row.names = FALSE)

cat("cohort mean beta by framing:\n")
res |> group_by(framing) |> summarise(beta = mean(beta), r = mean(r),
                                      amp = mean(amp)) |>
  mutate(across(where(is.numeric), \(x) round(x, 3))) |> print(n = 6)
cat(sprintf("wrote gain_model_R.csv (%d rows)\n", nrow(res)))
