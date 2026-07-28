#!/usr/bin/env Rscript
# Sample the adjective pairs for the frame experiment (Report VI §8.2).
#
# Design: the contrast model and the conditional-probability ("base rate") account of
# JUDGE's asymmetry make different predictions about how judgments behave when the FRAME
# changes. We hold the pair fixed and vary the question:
#   cond  - rgb's original conditional  ("a very-i person: how likely also j?")
#   sim   - directed similarity         ("how similar is a very-i person TO a very-j person?")
#   diff  - directed difference         ("how different is a very-i person FROM a very-j?")
# each in BOTH orders, so 6 prompts per pair.
#
# Sampling is stratified on
#   (a) prominence gap |f_i - f_j|  -- high vs low decile; the asymmetry lives here
#   (b) symmetric similarity tercile -- needed to vary common vs distinctive feature mass
#       for the non-complementarity test (does sim + diff stay constant?)
#
# Output: data/frame_pairs.csv  (i, j, adj_i, adj_j, block, sim_tercile, f_i, f_j, gap, sym)
suppressPackageStartupMessages({library(data.table); library(dplyr); library(tibble)})
setwd(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))), "reports"))
source("_common.R")

N_PER_CELL <- 42L          # 2 gap blocks x 3 similarity terciles x 42 = 252 pairs/block-set
set.seed(20260728)

sym  <- fread_gz("sym_ev.csv.gz")
asy  <- fread_gz("asym_ev.csv.gz")
disc <- setdiff(MODELS, c("Llama", "FalconMamba"))
n    <- nrow(adjectives)

Dbar <- Reduce(`+`, lapply(disc, function(m) {
  D <- build_matrix(asy, m, "asym"); D[is.na(D)] <- 0; D })) / length(disc)
Sbar <- Reduce(`+`, lapply(disc, function(m) build_matrix(sym, m, "sym"))) / length(disc)
fbar <- -rowMeans(Dbar)                      # consensus prominence, as in Report VI §2

ij  <- as.matrix(sym[, .(i, j)]) + 1L
pool <- tibble(i = ij[,1], j = ij[,2],
               f_i = fbar[ij[,1]], f_j = fbar[ij[,2]],
               gap = abs(fbar[ij[,1]] - fbar[ij[,2]]),
               sym = Sbar[ij])

qg <- quantile(pool$gap, c(0.10, 0.90))
pool <- pool %>%
  mutate(block = case_when(gap >= qg[2] ~ "high_gap", gap <= qg[1] ~ "low_gap",
                           TRUE ~ NA_character_)) %>%
  filter(!is.na(block)) %>%
  group_by(block) %>% mutate(sim_tercile = ntile(sym, 3)) %>% ungroup()

samp <- pool %>% group_by(block, sim_tercile) %>%
  slice_sample(n = N_PER_CELL) %>% ungroup() %>%
  mutate(adj_i = ADJ[i], adj_j = ADJ[j],
         i = i - 1L, j = j - 1L) %>%          # back to 0-based npz index space
  select(i, j, adj_i, adj_j, block, sim_tercile, f_i, f_j, gap, sym) %>%
  arrange(block, sim_tercile)

fwrite(samp, file.path(DATA, "frame_pairs.csv"))
cat(sprintf("wrote %s: %d pairs (%d prompts/model at 3 frames x 2 orders)\n",
            file.path(DATA, "frame_pairs.csv"), nrow(samp), 6*nrow(samp)))
print(samp %>% count(block, sim_tercile) %>% as.data.frame())
cat("\ngap / similarity by block:\n")
print(samp %>% group_by(block) %>%
        summarise(gap = mean(gap), sym_mean = mean(sym), sym_lo = min(sym),
                  sym_hi = max(sym), .groups = "drop") %>%
        as.data.frame())
cat("\nexample high-gap pairs:\n")
print(samp %>% filter(block == "high_gap") %>% slice_sample(n = 6) %>%
        select(adj_i, adj_j, gap, sym) %>% as.data.frame())
