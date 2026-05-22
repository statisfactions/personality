# R stub: item-level reclustering / EFA on IPIP-NEO-300 (288 items after deny-list)
#
# Inputs (in results/item_level/, exported by scripts/export_item_matrices.py):
#   item_meta.csv        — 288 rows: item_id, trait, facet, pole, item_text
#   human_item_corr.csv  — 288x288 Pearson from IPIP300.por (N=145,388)
#   <model>_item_cos.csv — 288x288 cosine, one per cohort model
#
# Outputs (whatever you write — left as exercises):
#   - Hierarchical clusters per model and for the human matrix
#   - 5-factor EFA loading patterns per model and human
#   - Side-by-side cluster/loading assignment tables per item
#
# IMPORTANT — anisotropy and what NOT to do:
#
# Pre-norm transformer residual streams have global anisotropy: most pairs of
# item activations sit at cosine 0.95+. This makes raw cosine matrices appear
# nearly rank-1 and is visible in the model CSVs (Phi4 has range +0.87+1.0,
# others +0.95-0.99 to +1.0).
#
# It is tempting to "fix" this by projecting out leading PCs of the item pool,
# subtracting the IPIP centroid, or subtracting a neutral-text baseline. DON'T
# DO THIS without thinking — those transformations risk removing exactly the
# trait/affect structure we want to discover. Per W9 §3-5: PC projection is
# only safe when the leading PCs are known noise; the IPIP centroid IS the
# signal we want to factor; neutral baseline shifts the cloud but doesn't
# isotropize it on high-anisotropy models.
#
# Recommended accommodation: extract ONE MORE FACTOR than you'd naively pick.
# The leading factor (MR1) in high-anisotropy models will be a "general
# activity / norm" factor that nearly every item loads positively on. The
# remaining factors capture the substantive structure orthogonal to it. This
# is standard bifactor-model practice in psychometrics. See nfactors=6 below.

library(psych)        # fa(), fa.diagram(), fa.sort()
library(GPArotation)  # rotation methods
# library(corrplot)   # optional, for visualization

DATA_DIR <- "results/item_level"

read_matrix <- function(name) {
  mat <- read.csv(file.path(DATA_DIR, name), row.names = 1, check.names = FALSE)
  as.matrix(mat)
}

# ---- Load -------------------------------------------------------------------
meta  <- read.csv(file.path(DATA_DIR, "item_meta.csv"), stringsAsFactors = FALSE)
human <- read_matrix("human_item_corr.csv")
model_name <- "Gemma4"   # pick any cohort model: Gemma, Gemma12, Llama, Llama8,
                          # Phi4, Qwen, Qwen7, Gemma27, Qwen32, Gemma4
model <- read_matrix(sprintf("%s_item_cos.csv", model_name))

stopifnot(nrow(human) == 288L, nrow(model) == 288L)
stopifnot(all(rownames(human) == rownames(model)))   # same item order
stopifnot(all(rownames(human) == meta$item_id))

# ---- Hierarchical clustering (Ward, cosine-distance) ------------------------
# Note: hclust expects a distance object. We use 1 - similarity as the proxy.
# WARNING for model matrices: anisotropy means raw 1-cosine distances are all
# very small (<0.05), and ward.D2 will be sensitive to noise in the lowest
# few digits. Treat the cluster output as exploratory only; the FA path below
# is more principled. Consider clustering on FA-derived factor scores (ex-MR1)
# as a follow-up if this section is uninformative.
hclust_assignments <- function(sim_matrix, k = 5) {
  diag(sim_matrix) <- 1.0
  d <- as.dist(1 - sim_matrix)
  hc <- hclust(d, method = "ward.D2")
  cutree(hc, k = k)
}

human_clusters  <- hclust_assignments(human, k = 5)
model_clusters  <- hclust_assignments(model, k = 5)

# Cross-tab clusters vs the Big-Five trait labels (sanity check):
cat("\nHuman item clusters x trait:\n")
print(table(meta$trait, human_clusters))
cat("\n", model_name, "item clusters x trait:\n", sep = "")
print(table(meta$trait, model_clusters))

# Items whose cluster differs between human and model:
diff_idx <- which(human_clusters != model_clusters)
cat(sprintf("\nItems in different clusters between human and %s: %d / %d\n",
            model_name, length(diff_idx), nrow(meta)))

# ---- EFA: 5-factor varimax-rotated -----------------------------------------
# fa() with cor=TRUE uses the matrix directly as the correlation/cov input.
# Use minres extraction (fm="minres") — the default and most stable. For the
# human matrix this is the standard EFA. For the model cosine matrix it's a
# defensible analogue since cosines behave like correlations on unit-normed
# vectors.

n_human_obs <- 145388   # for SE calibration
# Human IPIP-NEO-300 data has well-behaved item correlations (no anisotropy);
# 5 factors is the standard Big-Five solution.
fa_human <- fa(r = human, nfactors = 5, fm = "minres",
               rotate = "varimax", n.obs = n_human_obs)

# Model item-cosine matrices have anisotropy. Extract 6 factors and expect MR1
# to be a "general norm/activity" factor that everything loads positively on
# (the residual-stream anisotropy axis). MR2-MR6 carry the substantive
# structure orthogonal to it.
fa_model <- fa(r = model, nfactors = 6, fm = "minres",
               rotate = "varimax", n.obs = 50)  # 50 personas/item is loose

cat("\nHuman 5-factor solution: variance explained =\n")
print(fa_human$Vaccounted)
cat("\n", model_name, " 5-factor solution: variance explained =\n", sep = "")
print(fa_model$Vaccounted)

# Top-loading items per factor for inspection. Use fa.sort to reorder.
print_top_loaders <- function(fa_obj, label, n_top = 5) {
  L <- fa_obj$loadings
  cat(sprintf("\n=== Top %d-loading items per factor: %s ===\n", n_top, label))
  for (k in seq_len(ncol(L))) {
    top <- order(abs(L[, k]), decreasing = TRUE)[seq_len(n_top)]
    cat(sprintf("\nFactor MR%d:\n", k))
    for (i in top) {
      cat(sprintf("  %.2f  %s:%s  %s  %s\n",
                  L[i, k], meta$trait[i], meta$facet[i],
                  meta$pole[i], substr(meta$item_text[i], 1, 60)))
    }
  }
}
print_top_loaders(fa_human, "human IPIP-NEO-300", n_top = 5)
print_top_loaders(fa_model, sprintf("%s (cosine)", model_name), n_top = 5)

# ---- Item-level migration table --------------------------------------------
# Per item, dominant factor (by |loading|). For the model, exclude MR1 from
# "dominant" because it's likely the anisotropy general factor; the
# substantive structure is in MR2-MR6. Toggle drop_first=FALSE if you want
# MR1 included (e.g., when comparing across models to see whether MR1 itself
# carries trait content in low-anisotropy models like Phi4).
dominant_factor <- function(fa_obj, drop_first = FALSE) {
  L <- fa_obj$loadings
  if (drop_first) L <- L[, -1, drop = FALSE]
  apply(abs(L), 1, which.max) + as.integer(drop_first)
}
df <- data.frame(
  item_id = meta$item_id,
  trait   = meta$trait,
  facet   = meta$facet,
  pole    = meta$pole,
  human_factor = dominant_factor(fa_human, drop_first = FALSE),
  model_factor = dominant_factor(fa_model, drop_first = TRUE),
  human_cluster = human_clusters,
  model_cluster = model_clusters,
  item_text = meta$item_text,
  stringsAsFactors = FALSE
)
write.csv(df, file.path(DATA_DIR,
          sprintf("item_assignments_%s.csv", model_name)),
          row.names = FALSE)
cat(sprintf("\nWrote %s/item_assignments_%s.csv\n",
            DATA_DIR, model_name))

# Items where human and model disagree on the dominant factor:
migrants <- subset(df, human_factor != model_factor)
cat(sprintf("\nItems with different dominant factor between human and %s: %d\n",
            model_name, nrow(migrants)))
cat("By trait:\n")
print(table(migrants$trait))
