##############################################################################
# fit_irt_bfi_full.R
#
# Fit GRM and NRM to the full BFI run on Llama-2 7B Chat using the
# modified soft-evidence mirt package.
#
# The top-1 data only has categories {1, 3, 5} (category 2 suppression +
# ceiling effect). mirt remaps these to {1, 2, 3}. For soft evidence, we
# collapse the 5-category logprob distributions to match: p(low) = p1+p2,
# p(mid) = p3, p(high) = p4+p5.
#
# Input:  results_bfi-full_llama2-7b-chat_0.csv (from export_bfi_full_csv.py)
# Output: fit_irt_bfi_full_results.rds (fitted model objects + diagnostics)
#
# Usage: Rscript fit_irt_bfi_full.R   (or source interactively)
##############################################################################

library(mirt, lib.loc = "/home/ethanbrown/R/library")
library(tidyverse)

cat("====================================================================\n")
cat("  IRT analysis: GRM & NRM on full BFI run (Llama-2 7B Chat)\n")
cat("====================================================================\n\n")

# ── 1. Load and prepare data ────────────────────────────────────────────────

csv_path <- "results_bfi-full_llama2-7b-chat_0.csv"
cat("Loading", csv_path, "...\n")
df <- read_csv(csv_path, show_col_types = FALSE)
cat(sprintf("  %d rows, %d columns\n", nrow(df), ncol(df)))

# ── BFI scale definitions with reverse-keyed items ──
bfi_scales <- list(
  `BFI-EXT` = list(
    items = c("bf1","bf6","bf11","bf16","bf21","bf26","bf31","bf36"),
    reverse = c("bf6","bf21","bf31")
  ),
  `BFI-AGR` = list(
    items = c("bf2","bf7","bf12","bf17","bf22","bf27","bf32","bf37","bf42"),
    reverse = c("bf2","bf12","bf27","bf37")
  ),
  `BFI-CON` = list(
    items = c("bf3","bf8","bf13","bf18","bf23","bf28","bf33","bf38","bf43"),
    reverse = c("bf8","bf18","bf23","bf43")
  ),
  `BFI-NEU` = list(
    items = c("bf4","bf9","bf14","bf19","bf24","bf29","bf34","bf39"),
    reverse = c("bf9","bf24","bf34")
  ),
  `BFI-OPE` = list(
    items = c("bf5","bf10","bf15","bf20","bf25","bf30","bf35","bf40","bf41","bf44"),
    reverse = c("bf35","bf41")
  )
)

# ── Use logprob_argmax as top-1 response (avoids contaminated model_output) ──
df <- df %>%
  mutate(response = as.integer(logprob_argmax))

# Verify SPID structure
cat(sprintf("  SPIDs: %d | Items: %d\n",
            n_distinct(df$spid), n_distinct(df$item_id)))

# ── 2. Build person-by-item matrices ────────────────────────────────────────

# Top-1 responses: wide format (SPID × item)
resp_wide <- df %>%
  select(spid, item_id, response) %>%
  pivot_wider(names_from = item_id, values_from = response) %>%
  arrange(spid)

spid_ids <- resp_wide$spid
N <- nrow(resp_wide)
cat(sprintf("  Person-by-item matrix: %d persons x %d items\n",
            N, ncol(resp_wide) - 1))

# Logprob probability matrices: build N × J × K_raw (K_raw = 5) array
all_items <- sort(unique(df$item_id))
J <- length(all_items)
K_raw <- 5L

# Create long-form logprob data, sorted by spid and item
lp_data <- df %>%
  select(spid, item_id, logprob_1, logprob_2, logprob_3, logprob_4, logprob_5) %>%
  arrange(spid, item_id)

# Verify completeness (each SPID should have all 44 items)
completeness <- lp_data %>% count(spid)
stopifnot(all(completeness$n == J))

# Convert logprobs to probabilities and renormalize
lp_mat <- lp_data %>% select(starts_with("logprob_")) %>% as.matrix()

# Handle NAs: impute with floor (very low probability)
lp_floor <- min(lp_mat, na.rm = TRUE)
n_na <- sum(is.na(lp_mat))
if (n_na > 0) {
  cat(sprintf("  Imputing %d NA logprobs with floor value %.2f\n", n_na, lp_floor))
  lp_mat[is.na(lp_mat)] <- lp_floor
}

# Exponentiate and renormalize each row to sum to 1
prob_mat <- exp(lp_mat)
row_sums <- rowSums(prob_mat)
prob_mat <- prob_mat / row_sums

# Reshape into N × J × K_raw array (rows are ordered by spid, item_id)
pi_array_5 <- array(prob_mat, dim = c(J, N, K_raw))  # item-major from sorted data
pi_array_5 <- aperm(pi_array_5, c(2, 1, 3))          # → N × J × K_raw

cat(sprintf("  5-category soft-evidence array: %d × %d × %d\n",
            dim(pi_array_5)[1], dim(pi_array_5)[2], dim(pi_array_5)[3]))

# ── Collapse to 3 observed categories ──
# The top-1 data only has {1, 3, 5}. mirt remaps to {1, 2, 3}.
# Collapse soft evidence: p(low) = p1+p2, p(mid) = p3, p(high) = p4+p5.
K <- 3L
pi_array <- array(0, dim = c(N, J, K))
pi_array[, , 1] <- pi_array_5[, , 1] + pi_array_5[, , 2]  # low (1+2)
pi_array[, , 2] <- pi_array_5[, , 3]                        # mid (3)
pi_array[, , 3] <- pi_array_5[, , 4] + pi_array_5[, , 5]  # high (4+5)

# Verify sums
slice_sums <- apply(pi_array, c(1, 2), sum)
cat(sprintf("  3-category prob sum range: [%.6f, %.6f]\n",
            min(slice_sums), max(slice_sums)))

# Report mean probabilities across all items
cat(sprintf("  Mean 3-cat probs: low=%.4f, mid=%.4f, high=%.4f\n",
            mean(pi_array[, , 1]), mean(pi_array[, , 2]), mean(pi_array[, , 3])))


# ── 3. Helper: build soft-evidence matrix for mirt ──────────────────────────

build_softmat <- function(pi_sub) {
  # pi_sub: N × J_sub × K array of probabilities
  # Returns: N × sum(K) matrix in mirt's fulldata format (K cols per item)
  n <- dim(pi_sub)[1]
  j <- dim(pi_sub)[2]
  k <- dim(pi_sub)[3]
  softmat <- matrix(0, nrow = n, ncol = j * k)
  for (jj in 1:j) {
    cols <- ((jj - 1) * k + 1):(jj * k)
    softmat[, cols] <- pi_sub[, jj, ]
  }
  softmat
}


# ── 4. Fit models per Big Five trait ────────────────────────────────────────

results <- list()

for (scale_id in names(bfi_scales)) {

  cat("\n====================================================================\n")
  cat(sprintf("  Fitting: %s (%d items, %d reverse-keyed)\n",
              scale_id,
              length(bfi_scales[[scale_id]]$items),
              length(bfi_scales[[scale_id]]$reverse)))
  cat("====================================================================\n")

  items <- bfi_scales[[scale_id]]$items
  rev_items <- bfi_scales[[scale_id]]$reverse
  J_sub <- length(items)

  # Find column indices in all_items
  item_idx <- match(items, all_items)

  # ── Top-1 response matrix for this scale ──
  dat_top1 <- resp_wide %>% select(all_of(items)) %>% as.data.frame()

  # Reverse-key: for IRT, we need all items keyed in the same direction
  # Reverse-keyed items: 6 - response (for 1-5 scale)
  for (ri in rev_items) {
    dat_top1[[ri]] <- 6L - dat_top1[[ri]]
  }

  # After reverse-keying, data has {1, 3, 5}. mirt will remap to {1, 2, 3}.
  cat("  Response distribution (after reverse-keying):\n")
  print(table(unlist(dat_top1), useNA = "ifany"))

  # ── Soft-evidence sub-array for this scale (3-category) ──
  pi_sub <- pi_array[, item_idx, , drop = FALSE]

  # Reverse-key the soft evidence: flip category order
  # For 3-cat: low↔high, mid stays
  for (ri in rev_items) {
    ri_idx <- which(items == ri)
    pi_sub[, ri_idx, ] <- pi_sub[, ri_idx, K:1]
  }

  # ── 4a. Standard GRM (top-1) — fit first to learn actual K per item ──
  cat("\n  --- Standard GRM (top-1 responses) ---\n")
  t0 <- Sys.time()
  mod_grm_top1 <- tryCatch(
    mirt(dat_top1, 1, itemtype = "graded", verbose = FALSE, TOL = 1e-4),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  cat(sprintf("  Time: %.1f sec\n", as.numeric(Sys.time() - t0, units = "secs")))

  if (!is.null(mod_grm_top1)) {
    cat("  Coefficients:\n")
    print(round(coef(mod_grm_top1, simplify = TRUE)$items, 3))
    cat(sprintf("  logLik: %.1f | AIC: %.1f | BIC: %.1f\n",
                mod_grm_top1@Fit$logLik, mod_grm_top1@Fit$AIC, mod_grm_top1@Fit$BIC))
  }

  # ── Build softmat using mirt's actual K per item ──
  # Some items may have fewer than 3 observed categories (e.g., bf15 has K=2).
  # We need to collapse the 3-category probs to match mirt's internal structure.
  softmat <- NULL
  if (!is.null(mod_grm_top1)) {
    K_vec <- extract.mirt(mod_grm_top1, "K")
    itemloc <- extract.mirt(mod_grm_top1, "itemloc")
    ncols <- sum(K_vec)
    softmat <- matrix(0, nrow = N, ncol = ncols)
    for (jj in 1:J_sub) {
      cols <- itemloc[jj]:(itemloc[jj + 1] - 1)
      if (K_vec[jj] == 3) {
        softmat[, cols] <- pi_sub[, jj, ]
      } else if (K_vec[jj] == 2) {
        # Item has only 2 observed categories — collapse mid into whichever
        # category it's closer to based on the observed values
        obs_cats <- sort(unique(dat_top1[[items[jj]]]))
        if (length(obs_cats) == 2 && all(obs_cats == c(3, 5))) {
          # Only mid(3) and high(5): collapse low into mid
          softmat[, cols[1]] <- pi_sub[, jj, 1] + pi_sub[, jj, 2]
          softmat[, cols[2]] <- pi_sub[, jj, 3]
        } else if (length(obs_cats) == 2 && all(obs_cats == c(1, 5))) {
          # Only low(1) and high(5): collapse mid into low
          softmat[, cols[1]] <- pi_sub[, jj, 1] + pi_sub[, jj, 2]
          softmat[, cols[2]] <- pi_sub[, jj, 3]
        } else {
          # Default: first half low, second half high
          softmat[, cols[1]] <- pi_sub[, jj, 1] + pi_sub[, jj, 2]
          softmat[, cols[2]] <- pi_sub[, jj, 3]
        }
        cat(sprintf("  Note: %s has K=%d, collapsed soft evidence accordingly\n",
                    items[jj], K_vec[jj]))
      }
    }
    # Verify row sums
    row_sums_sm <- rowSums(softmat)
    cat(sprintf("  Softmat: %d × %d (sum(K)=%d), row sum range: [%.4f, %.4f]\n",
                nrow(softmat), ncol(softmat), ncols,
                min(row_sums_sm), max(row_sums_sm)))
  }

  # ── 4b. Soft-evidence GRM ──
  cat("\n  --- Soft-evidence GRM ---\n")
  t0 <- Sys.time()
  mod_grm_soft <- if (!is.null(softmat)) {
    tryCatch(
      mirt(dat_top1, 1, itemtype = "graded", verbose = FALSE, TOL = 1e-4,
           technical = list(softevidence = softmat)),
      error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
    )
  } else {
    cat("  SKIPPED (no softmat available)\n"); NULL
  }
  cat(sprintf("  Time: %.1f sec\n", as.numeric(Sys.time() - t0, units = "secs")))

  if (!is.null(mod_grm_soft)) {
    cat("  Coefficients:\n")
    print(round(coef(mod_grm_soft, simplify = TRUE)$items, 3))
    cat(sprintf("  logLik: %.1f | AIC: %.1f | BIC: %.1f\n",
                mod_grm_soft@Fit$logLik, mod_grm_soft@Fit$AIC, mod_grm_soft@Fit$BIC))
  }

  # ── 4c. Standard NRM (top-1) ──
  cat("\n  --- Standard NRM (top-1 responses) ---\n")
  t0 <- Sys.time()
  mod_nrm_top1 <- tryCatch(
    mirt(dat_top1, 1, itemtype = "nominal", verbose = FALSE, TOL = 1e-4),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )
  cat(sprintf("  Time: %.1f sec\n", as.numeric(Sys.time() - t0, units = "secs")))

  if (!is.null(mod_nrm_top1)) {
    cat("  Coefficients:\n")
    nrm_coefs <- coef(mod_nrm_top1, simplify = TRUE)$items
    print(round(nrm_coefs, 3))
    cat(sprintf("  logLik: %.1f | AIC: %.1f | BIC: %.1f\n",
                mod_nrm_top1@Fit$logLik, mod_nrm_top1@Fit$AIC, mod_nrm_top1@Fit$BIC))
  }

  # ── 4d. Soft-evidence NRM ──
  # NRM may have different K per item than GRM (unlikely but possible).
  # Rebuild softmat from NRM's itemloc structure if needed.
  softmat_nrm <- NULL
  if (!is.null(mod_nrm_top1)) {
    K_vec_nrm <- extract.mirt(mod_nrm_top1, "K")
    itemloc_nrm <- extract.mirt(mod_nrm_top1, "itemloc")
    ncols_nrm <- sum(K_vec_nrm)
    if (ncols_nrm == ncol(softmat) && !is.null(softmat)) {
      softmat_nrm <- softmat  # Same K structure
    } else {
      # Rebuild for NRM's K structure
      softmat_nrm <- matrix(0, nrow = N, ncol = ncols_nrm)
      for (jj in 1:J_sub) {
        cols <- itemloc_nrm[jj]:(itemloc_nrm[jj + 1] - 1)
        if (K_vec_nrm[jj] == 3) {
          softmat_nrm[, cols] <- pi_sub[, jj, ]
        } else if (K_vec_nrm[jj] == 2) {
          softmat_nrm[, cols[1]] <- pi_sub[, jj, 1] + pi_sub[, jj, 2]
          softmat_nrm[, cols[2]] <- pi_sub[, jj, 3]
        }
      }
    }
  }

  cat("\n  --- Soft-evidence NRM ---\n")
  t0 <- Sys.time()
  mod_nrm_soft <- if (!is.null(softmat_nrm)) {
    tryCatch(
      mirt(dat_top1, 1, itemtype = "nominal", verbose = FALSE, TOL = 1e-4,
           technical = list(softevidence = softmat_nrm)),
      error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
    )
  } else {
    cat("  SKIPPED (no softmat available)\n"); NULL
  }
  cat(sprintf("  Time: %.1f sec\n", as.numeric(Sys.time() - t0, units = "secs")))

  if (!is.null(mod_nrm_soft)) {
    cat("  Coefficients:\n")
    nrm_soft_coefs <- coef(mod_nrm_soft, simplify = TRUE)$items
    print(round(nrm_soft_coefs, 3))
    cat(sprintf("  logLik: %.1f | AIC: %.1f | BIC: %.1f\n",
                mod_nrm_soft@Fit$logLik, mod_nrm_soft@Fit$AIC, mod_nrm_soft@Fit$BIC))
  }

  # ── 4e. Model comparison ──
  cat("\n  --- Model Comparison ---\n")

  if (!is.null(mod_grm_top1) && !is.null(mod_nrm_top1)) {
    cat("  Top-1 data:\n")
    cat(sprintf("    GRM: logLik=%.1f  AIC=%.1f  BIC=%.1f\n",
                mod_grm_top1@Fit$logLik, mod_grm_top1@Fit$AIC, mod_grm_top1@Fit$BIC))
    cat(sprintf("    NRM: logLik=%.1f  AIC=%.1f  BIC=%.1f\n",
                mod_nrm_top1@Fit$logLik, mod_nrm_top1@Fit$AIC, mod_nrm_top1@Fit$BIC))
    delta_aic <- mod_grm_top1@Fit$AIC - mod_nrm_top1@Fit$AIC
    cat(sprintf("    Delta AIC (GRM - NRM): %.1f (%s)\n",
                delta_aic,
                ifelse(delta_aic < 0, "GRM preferred", "NRM preferred")))
  }

  if (!is.null(mod_grm_soft) && !is.null(mod_nrm_soft)) {
    cat("  Soft-evidence data:\n")
    cat(sprintf("    GRM: logLik=%.1f  AIC=%.1f  BIC=%.1f\n",
                mod_grm_soft@Fit$logLik, mod_grm_soft@Fit$AIC, mod_grm_soft@Fit$BIC))
    cat(sprintf("    NRM: logLik=%.1f  AIC=%.1f  BIC=%.1f\n",
                mod_nrm_soft@Fit$logLik, mod_nrm_soft@Fit$AIC, mod_nrm_soft@Fit$BIC))
    delta_aic <- mod_grm_soft@Fit$AIC - mod_nrm_soft@Fit$AIC
    cat(sprintf("    Delta AIC (GRM - NRM): %.1f (%s)\n",
                delta_aic,
                ifelse(delta_aic < 0, "GRM preferred", "NRM preferred")))
  }

  # ── Compare soft vs top-1 GRM parameters ──
  if (!is.null(mod_grm_top1) && !is.null(mod_grm_soft)) {
    cat("\n  --- Soft vs Top-1 GRM parameter comparison ---\n")
    coefs_t <- coef(mod_grm_top1, simplify = TRUE)$items
    coefs_s <- coef(mod_grm_soft, simplify = TRUE)$items
    shared <- intersect(colnames(coefs_t), colnames(coefs_s))
    diff_mat <- coefs_s[, shared] - coefs_t[, shared]
    cat(sprintf("    Mean abs difference: %.4f\n", mean(abs(diff_mat))))
    cat(sprintf("    Max abs difference:  %.4f\n", max(abs(diff_mat))))
    # Discrimination correlation
    a_cor <- cor(coefs_t[, "a1"], coefs_s[, "a1"])
    cat(sprintf("    Discrimination correlation (a1): %.4f\n", a_cor))
  }

  # ── Factor scores ──
  if (!is.null(mod_grm_soft)) {
    cat("\n  --- Factor scores (soft-evidence GRM) ---\n")
    fs <- tryCatch(
      fscores(mod_grm_soft, method = "EAP", verbose = FALSE),
      error = function(e) { cat("  fscores ERROR:", e$message, "\n"); NULL }
    )
    if (!is.null(fs)) {
      cat(sprintf("    EAP scores: mean=%.3f, sd=%.3f, range=[%.3f, %.3f]\n",
                  mean(fs[,1]), sd(fs[,1]), min(fs[,1]), max(fs[,1])))
    }
  }

  # Store results
  results[[scale_id]] <- list(
    grm_top1  = mod_grm_top1,
    grm_soft  = mod_grm_soft,
    nrm_top1  = mod_nrm_top1,
    nrm_soft  = mod_nrm_soft,
    items     = items,
    reverse   = rev_items,
    dat_top1  = dat_top1,
    softmat   = softmat,
    pi_sub    = pi_sub
  )
}


# ── 5. Summary table ────────────────────────────────────────────────────────

cat("\n\n====================================================================\n")
cat("  SUMMARY: AIC/BIC across all scales\n")
cat("====================================================================\n\n")

summary_rows <- names(results) %>%
  map(function(scale_id) {
    r <- results[[scale_id]]
    tibble(
      Scale = scale_id,
      J = length(r$items),
      GRM_top1_AIC = if (!is.null(r$grm_top1)) r$grm_top1@Fit$AIC else NA_real_,
      GRM_top1_BIC = if (!is.null(r$grm_top1)) r$grm_top1@Fit$BIC else NA_real_,
      GRM_soft_AIC = if (!is.null(r$grm_soft)) r$grm_soft@Fit$AIC else NA_real_,
      GRM_soft_BIC = if (!is.null(r$grm_soft)) r$grm_soft@Fit$BIC else NA_real_,
      NRM_top1_AIC = if (!is.null(r$nrm_top1)) r$nrm_top1@Fit$AIC else NA_real_,
      NRM_top1_BIC = if (!is.null(r$nrm_top1)) r$nrm_top1@Fit$BIC else NA_real_,
      NRM_soft_AIC = if (!is.null(r$nrm_soft)) r$nrm_soft@Fit$AIC else NA_real_,
      NRM_soft_BIC = if (!is.null(r$nrm_soft)) r$nrm_soft@Fit$BIC else NA_real_
    )
  }) %>%
  bind_rows()

print(summary_rows, n = Inf, width = 120)

# ── 6. Save ──
rds_path <- "fit_irt_bfi_full_results.rds"
saveRDS(results, rds_path)
cat(sprintf("\nResults saved to %s\n", rds_path))

cat("\nDone.\n")
