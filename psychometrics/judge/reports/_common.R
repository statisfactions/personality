# Shared helpers for the JUDGE psychometric reports (ecb track).
# Data produced by ../convert_judge.py into ../data/. Models are treated as raters.
suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(tidyr); library(purrr)
  library(ggplot2); library(tibble); library(stringr)
})

JUDGE_DATA <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "."), "..", "data"),
                            mustWork = FALSE)
`%||%` <- function(a, b) if (is.null(a)) b else a
# Robust data dir resolution regardless of knit working dir.
.find_data <- function() {
  for (p in c("../data", "data", "psychometrics/judge/data",
              file.path(getwd(), "../data"))) if (dir.exists(p)) return(normalizePath(p))
  stop("cannot locate judge data dir")
}
DATA <- .find_data()

fread_gz <- function(name) data.table::fread(cmd = paste("gzip -dc", shQuote(file.path(DATA, name))))
read_csv_ <- function(name) data.table::fread(file.path(DATA, name))

# ---- metadata ----
model_meta <- read_csv_("model_meta.csv") %>% as_tibble()
MODELS  <- model_meta$model                       # canonical order (family, size)
DISPLAY <- setNames(model_meta$display, model_meta$model)
FAMILY  <- setNames(model_meta$family,  model_meta$model)
PARAMS  <- setNames(model_meta$params_b, model_meta$model)
adjectives <- read_csv_("adjectives.csv") %>% as_tibble()
ADJ <- adjectives$adjective                       # 0-indexed by position (idx col)

# family palette (colour-blind-safe-ish, consistent across all reports)
FAM_PAL <- c(Qwen="#4C72B0", Gemma="#DD8452", Llama="#55A868", Phi="#C44E52",
             Aya="#8172B3", Falcon="#937860")
disp_ord <- model_meta %>% arrange(family, params_b) %>% pull(display)

# ---- matrix reconstruction from a wide i,j,<model...> table ----
# `kind`: "sym" (fill both triangles), "dir" (directed, fill [i,j] only),
#         "asym" (fill [i,j]=v, [j,i]=-v). Returns 523x523 with NA diagonal.
build_matrix <- function(dt, model, kind = "sym") {
  n <- nrow(adjectives)
  M <- matrix(NA_real_, n, n)
  i <- dt$i + 1L; j <- dt$j + 1L; v <- dt[[model]]
  M[cbind(i, j)] <- v
  if (kind == "sym")  M[cbind(j, i)] <- v
  if (kind == "asym") M[cbind(j, i)] <- -v
  M
}

# Convert a symmetric similarity matrix (1..7 Likert EV, NA diag) into a
# correlation-like matrix for factoring, via profile correlation (correlate the
# off-diagonal co-endorsement profiles). This mirrors rgb's W16 §9 note that the
# human matrix is itself a profile correlation over people; used consistently.
profile_cor <- function(S) {
  diag(S) <- NA
  stats::cor(S, use = "pairwise.complete.obs")
}

# double-center off-diagonal (strip additive row/col marginals) — the
# prevalence-free comparison basis (rgb W16 §8 correction).
double_center <- function(M) {
  D <- M; diag(D) <- NA
  r <- rowMeans(D, na.rm = TRUE); c <- colMeans(D, na.rm = TRUE)
  g <- mean(D, na.rm = TRUE)
  out <- sweep(sweep(D, 1, r), 2, c) + g
  out
}

theme_judge <- function() theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"),
        legend.position = "bottom")

upper_vec <- function(M) M[upper.tri(M)]
