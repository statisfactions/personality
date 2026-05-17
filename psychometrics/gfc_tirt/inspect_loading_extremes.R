#!/usr/bin/env Rscript
# Show the items at the loading extremes — both globally (averaged across
# fits, after z-scoring within fit to remove per-model scale) and within
# the strongest individual fit. Does an item's loading magnitude track
# anything intuitive about its content?

suppressMessages({
  library(rstan)
  library(dplyr)
  library(jsonlite)
})

# Map block,side -> item text from the instrument JSON.
load_instrument_items <- function(path) {
  d <- fromJSON(path, flatten = FALSE)
  # pairs[[i]] has block, left, right; each side has text, trait, keying, sd
  pairs <- d$pairs
  rows <- lapply(seq_len(nrow(pairs)), function(i) {
    p <- pairs[i, ]
    rbind(
      data.frame(block = p$block, side = "L",
                 text = p$left$text, trait = p$left$trait, keying = p$left$keying,
                 sd = p$left$sd, stringsAsFactors = FALSE),
      data.frame(block = p$block, side = "R",
                 text = p$right$text, trait = p$right$trait, keying = p$right$keying,
                 sd = p$right$sd, stringsAsFactors = FALSE)
    )
  })
  do.call(rbind, rows)
}

p30_items <- load_instrument_items("instruments/okada_gfc30.json")
p60_items <- load_instrument_items("instruments/ipip_neo_gfc_P60.json")
# Order: block 1 L, block 1 R, block 2 L, block 2 R, ... -> matches stmt_index
p30_items <- p30_items[order(p30_items$block, p30_items$side == "R"), ]
p30_items$j <- seq_len(nrow(p30_items))
p60_items <- p60_items[order(p60_items$block, p60_items$side == "R"), ]
p60_items$j <- seq_len(nrow(p60_items))

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

# Collect a_pos posteriors with item identity per fit
all_rows <- list()
for (f in rds_files) {
  meta <- parse_meta(f)
  if (is.null(meta)) next
  fit_obj <- readRDS(f)
  a <- colMeans(rstan::extract(fit_obj$fit, pars = "a_pos")$a_pos)
  J <- length(a)
  items <- if (meta$inst == "P30") p30_items else p60_items
  if (nrow(items) != J) { message("J mismatch ", basename(f)); next }
  all_rows[[length(all_rows) + 1]] <- data.frame(
    inst = meta$inst, model = meta$model, form = meta$form,
    j = items$j, block = items$block, side = items$side,
    trait = items$trait, keying = items$keying, text = items$text,
    a = a,
    stringsAsFactors = FALSE
  )
}
df <- do.call(rbind, all_rows)

# Per-fit z-score to remove per-model scale differences
df <- df %>%
  group_by(inst, model, form) %>%
  mutate(a_z = (a - mean(a)) / sd(a)) %>%
  ungroup()

# Aggregate across (model, form) within an instrument: which items are
# CONSISTENTLY high or low? Average z across all fits using that item.
agg <- df %>%
  group_by(inst, text, trait, keying) %>%
  summarise(
    a_z_mean = mean(a_z),
    a_mean = mean(a),
    n_fits = n(),
    .groups = "drop"
  )

cat("\n=== P=30 (Okada): items with HIGHEST mean z-loading across all fits ===\n")
p30_top <- agg %>% filter(inst == "P30") %>% arrange(desc(a_z_mean)) %>% head(8)
print(p30_top %>% select(text, trait, keying, a_z_mean, a_mean), row.names = FALSE)

cat("\n=== P=30 (Okada): items with LOWEST mean z-loading across all fits ===\n")
p30_bot <- agg %>% filter(inst == "P30") %>% arrange(a_z_mean) %>% head(8)
print(p30_bot %>% select(text, trait, keying, a_z_mean, a_mean), row.names = FALSE)

cat("\n=== P=60 (IPIP-NEO-GFC): items with HIGHEST mean z-loading across all fits ===\n")
p60_top <- agg %>% filter(inst == "P60") %>% arrange(desc(a_z_mean)) %>% head(10)
print(p60_top %>% select(text, trait, keying, a_z_mean, a_mean), row.names = FALSE)

cat("\n=== P=60 (IPIP-NEO-GFC): items with LOWEST mean z-loading across all fits ===\n")
p60_bot <- agg %>% filter(inst == "P60") %>% arrange(a_z_mean) %>% head(10)
print(p60_bot %>% select(text, trait, keying, a_z_mean, a_mean), row.names = FALSE)

cat("\n=== Single strongest fit: Gemma12 P60 description ===\n")
gx <- df %>% filter(inst == "P60", model == "Gemma12", form == "description")
cat("\n  TOP 8 raw a_pos:\n")
print(gx %>% arrange(desc(a)) %>% head(8) %>% select(text, trait, keying, a),
      row.names = FALSE)
cat("\n  BOTTOM 8 raw a_pos:\n")
print(gx %>% arrange(a) %>% head(8) %>% select(text, trait, keying, a),
      row.names = FALSE)
