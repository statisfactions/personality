#!/usr/bin/env Rscript
# Pull a_pos posterior means out of every saved TIRT fit and inspect.
#
# Questions:
#   - Are loadings hugging the prior (small, shrunken to ~0 under HN(0, 0.5))?
#   - Are they uniform across traits, or trait-asymmetric?
#   - Do P=60 fits have different loading magnitudes than P=30?
#   - Do variant-prior fits (tightkappa, lognorm) free the loadings up?

suppressMessages({
  library(rstan)
  library(dplyr)
  library(jsonlite)
})

trait_names <- c("A", "C", "E", "N", "O")

rds_files <- list.files(
  "psychometrics/gfc_tirt",
  pattern = "_(gfc30_hf|ipipneogfc60_hf)_.*_indep_fit\\.rds$",
  full.names = TRUE
)
rds_files <- rds_files[!grepl("_n25_", rds_files)]

parse_meta <- function(f) {
  bn <- sub("_indep_fit$", "", tools::file_path_sans_ext(basename(f)))
  if (grepl("_ipipneogfc60_hf_", bn, fixed = TRUE)) {
    parts <- strsplit(bn, "_ipipneogfc60_hf_", fixed = TRUE)[[1]]
    list(model = parts[1], form = parts[2], inst = "P60")
  } else if (grepl("_gfc30_hf_", bn, fixed = TRUE)) {
    parts <- strsplit(bn, "_gfc30_hf_", fixed = TRUE)[[1]]
    list(model = parts[1], form = parts[2], inst = "P30")
  } else NULL
}

# For each fit, derive per-statement trait labels from the response JSON
# (same approach as the fitter — read records, filter swapped=FALSE).
load_trait_map <- function(responses_path) {
  raw <- fromJSON(responses_path, flatten = FALSE)
  res <- raw$results %>%
    filter(!swapped) %>%
    arrange(block) %>%
    group_by(block) %>%
    slice(1) %>%
    ungroup()
  # statements are interleaved L,R,L,R,... so trait[j] = c(rbind(left, right))
  c(rbind(res$left_trait, res$right_trait))
}

rows <- list()
per_item <- list()

for (f in rds_files) {
  meta <- parse_meta(f)
  if (is.null(meta)) next
  fit_obj <- readRDS(f)
  # extract a_pos posterior mean
  a_post <- rstan::extract(fit_obj$fit, pars = "a_pos")$a_pos  # iter x J
  a_mean <- colMeans(a_post)
  a_sd   <- apply(a_post, 2, sd)

  trait_map <- tryCatch(load_trait_map(fit_obj$responses_path), error = function(e) NULL)
  if (is.null(trait_map) || length(trait_map) != length(a_mean)) {
    message("Skip (trait map mismatch): ", basename(f)); next
  }

  rows[[length(rows) + 1]] <- data.frame(
    inst = meta$inst, model = meta$model, form = meta$form,
    J = length(a_mean),
    a_mean_mean   = mean(a_mean),
    a_mean_median = median(a_mean),
    a_mean_min    = min(a_mean),
    a_mean_max    = max(a_mean),
    pct_below_0.2 = mean(a_mean < 0.2) * 100,
    pct_above_1   = mean(a_mean > 1.0) * 100,
    stringsAsFactors = FALSE
  )
  per_item[[length(per_item) + 1]] <- data.frame(
    inst = meta$inst, model = meta$model, form = meta$form,
    item = seq_along(a_mean), trait = trait_map,
    a_mean = a_mean, a_sd = a_sd, stringsAsFactors = FALSE
  )
}

df <- do.call(rbind, rows)
df <- df[order(df$inst, df$form, df$model), ]
cat("\n=== Per-fit a_pos summary ===\n")
print(df, row.names = FALSE, digits = 3)

pooled <- do.call(rbind, per_item)

cat("\n=== Pooled a_pos by instrument ===\n")
# Rename to avoid dplyr self-reference inside summarise()
pooled2 <- pooled %>% rename(a = a_mean)
agg_inst <- pooled2 %>%
  group_by(inst) %>%
  summarise(
    n_items = n(),
    mean = mean(a),
    median = median(a),
    p10 = quantile(a, 0.10),
    p90 = quantile(a, 0.90),
    sd = sd(a),
    pct_below_0.2 = mean(a < 0.2) * 100,
    pct_above_1 = mean(a > 1.0) * 100,
    .groups = "drop"
  )
print(agg_inst, digits = 3)

cat("\n=== Pooled a_pos by instrument x trait ===\n")
agg_trait <- pooled2 %>%
  group_by(inst, trait) %>%
  summarise(
    mean = mean(a),
    median = median(a),
    sd = sd(a),
    .groups = "drop"
  )
print(agg_trait, digits = 3)

cat("\n=== Pooled a_pos by instrument x model ===\n")
agg_model <- pooled2 %>%
  group_by(inst, model) %>%
  summarise(
    mean = mean(a),
    p90 = quantile(a, 0.90),
    max = max(a),
    .groups = "drop"
  )
print(agg_model, digits = 3)

# Prior context
cat("\nPrior reference:\n")
cat("  HalfNormal(0, 0.5): mean = sigma*sqrt(2/pi) =", round(0.5 * sqrt(2/pi), 3), "\n")
cat("                     median = sigma*sqrt(2)*qnorm(0.75)/sqrt(pi) ~ ", round(0.5 * qnorm(0.75), 3), "\n")
cat("                     P(a < 0.2) under prior =", round(2*(pnorm(0.2/0.5) - 0.5), 3), "\n")

# Variant fits weren't persisted as their own .rds — those files are
# auto-cached stanmodel objects, not stanfits.

write(toJSON(list(per_fit = df, by_instrument = agg_inst, by_instrument_trait = agg_trait),
             pretty = TRUE, auto_unbox = TRUE),
      "results/persona/persona_loadings_summary.json")
message("\nWrote results/persona/persona_loadings_summary.json")
