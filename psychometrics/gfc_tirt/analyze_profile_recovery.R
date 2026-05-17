#!/usr/bin/env Rscript
# Profile recovery diagnostic: complementary to the diagonal-r metric.
#
# Diagonal r (already reported) is ACROSS-PERSONA, PER-TRAIT:
#   for each trait t, cor(theta[:, t], gt[:, t]).
# It answers "do high-A personas get higher A estimates than low-A ones?"
#
# Profile r (this script) is WITHIN-PERSON, ACROSS-TRAITS:
#   for each persona i, cor(theta[i, ], gt[i, ]) over the 5 traits.
# It answers "within a single persona, is the trait ordering recovered?"
#
# Per-trait sign flips (known A/C flip on some models under Okada-indep)
# tank profile r mechanically, so we sign-correct per (model, form, inst)
# using the cohort diagonal sign before computing profile correlations.
#
# Usage: Rscript psychometrics/gfc_tirt/analyze_profile_recovery.R

suppressMessages({
  library(jsonlite)
  library(dplyr)
})

trait_names <- c("A", "C", "E", "N", "O")

personas_json <- fromJSON("instruments/synthetic_personas.json")
gt <- data.frame(
  persona_id = personas_json$personas$persona_id,
  A = personas_json$personas$z_scores$A,
  C = personas_json$personas$z_scores$C,
  E = personas_json$personas$z_scores$E,
  N = personas_json$personas$z_scores$N,
  O = personas_json$personas$z_scores$O,
  stringsAsFactors = FALSE
)

rds_files <- list.files(
  "psychometrics/gfc_tirt",
  pattern = "_(gfc30_hf|ipipneogfc60_hf)_.*_indep_fit\\.rds$",
  full.names = TRUE
)
rds_files <- rds_files[!grepl("_n25_", rds_files)]
message("Found ", length(rds_files), " fits")

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

rows <- list()
all_profile <- list()  # for distribution plotting / pooled stats

for (f in rds_files) {
  meta <- parse_meta(f)
  if (is.null(meta)) next
  fit_obj <- readRDS(f)
  theta <- fit_obj$theta_mean
  pids  <- fit_obj$persona_ids
  g <- gt[match(pids, gt$persona_id), ]
  if (any(is.na(g$persona_id))) {
    message("Skip (persona mismatch): ", basename(f)); next
  }
  gt_mat <- as.matrix(g[, trait_names])

  diag_r <- suppressWarnings(sapply(trait_names, function(t) cor(theta[, t], gt_mat[, t])))
  signs  <- ifelse(diag_r < 0, -1, 1)
  theta_signed <- sweep(theta, 2, signs, `*`)

  profile_sp <- suppressWarnings(sapply(seq_len(nrow(theta_signed)), function(i) {
    cor(theta_signed[i, ], gt_mat[i, ], method = "spearman")
  }))
  profile_pe <- suppressWarnings(sapply(seq_len(nrow(theta_signed)), function(i) {
    cor(theta_signed[i, ], gt_mat[i, ], method = "pearson")
  }))

  rows[[length(rows) + 1]] <- data.frame(
    inst = meta$inst, model = meta$model, form = meta$form,
    n = nrow(theta),
    diag_abs_mean = mean(abs(diag_r)),
    n_flipped = sum(signs == -1),
    profile_sp_mean   = mean(profile_sp,  na.rm = TRUE),
    profile_sp_median = median(profile_sp, na.rm = TRUE),
    profile_pe_mean   = mean(profile_pe,  na.rm = TRUE),
    pct_sp_gt_0   = mean(profile_sp > 0,   na.rm = TRUE) * 100,
    pct_sp_gt_0.5 = mean(profile_sp > 0.5, na.rm = TRUE) * 100,
    stringsAsFactors = FALSE
  )
  all_profile[[length(all_profile) + 1]] <- data.frame(
    inst = meta$inst, model = meta$model, form = meta$form,
    persona_id = pids, profile_sp = profile_sp, profile_pe = profile_pe,
    stringsAsFactors = FALSE
  )
}

df <- do.call(rbind, rows)
df <- df[order(df$inst, df$form, df$model), ]
cat("\n=== Per-fit profile recovery (sign-corrected) ===\n")
print(df, row.names = FALSE, digits = 3)

pooled <- do.call(rbind, all_profile)

cat("\n=== Pooled by instrument ===\n")
agg_inst <- pooled %>%
  group_by(inst) %>%
  summarise(
    n_fits = length(unique(paste(model, form))),
    n_obs = n(),
    profile_sp_mean = mean(profile_sp, na.rm = TRUE),
    profile_sp_median = median(profile_sp, na.rm = TRUE),
    profile_pe_mean = mean(profile_pe, na.rm = TRUE),
    pct_gt_0 = mean(profile_sp > 0, na.rm = TRUE) * 100,
    pct_gt_0.5 = mean(profile_sp > 0.5, na.rm = TRUE) * 100,
    .groups = "drop"
  )
print(agg_inst, digits = 3)

cat("\n=== Pooled by instrument x form ===\n")
agg_form <- pooled %>%
  group_by(inst, form) %>%
  summarise(
    profile_sp_mean = mean(profile_sp, na.rm = TRUE),
    profile_pe_mean = mean(profile_pe, na.rm = TRUE),
    pct_gt_0 = mean(profile_sp > 0, na.rm = TRUE) * 100,
    .groups = "drop"
  )
print(agg_form, digits = 3)

cat("\n=== Reference: average diagonal |r| by instrument ===\n")
ref <- df %>%
  group_by(inst) %>%
  summarise(
    diag_abs_mean = mean(diag_abs_mean),
    n_sign_flips_total = sum(n_flipped),
    .groups = "drop"
  )
print(ref, digits = 3)

out_path <- "results/persona/persona_profile_recovery.json"
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write(toJSON(list(
  per_fit = df,
  by_instrument = agg_inst,
  by_instrument_form = agg_form
), pretty = TRUE, auto_unbox = TRUE), out_path)
message("\nWrote ", out_path)
