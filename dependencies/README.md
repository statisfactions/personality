# dependencies/ — third-party repos and downloads (gitignored)

Contents are never committed; `scripts/fetch_dependencies.sh` restores the
automatable ones. Manual-retrieval datasets are documented below and live in
`data/` (also gitignored).

## Automated (fetch_dependencies.sh)

| dep | source | used by |
|---|---|---|
| `ValuePortrait/` | github.com/holi-lab/ValuePortrait | `vp_rescore.py`, `vp_eta2.py`, `vp_fig1_transparency.py` (W18 §7) |
| `jacobian-lens/` | github.com/anthropics/jacobian-lens | reference implementation for the J-lens probes (the lens *weights* come from HF, below) |
| `persona-cartography/` | github.com/persona-cartography/persona-cartography | queued: LoRA ΔW vs ENACT-span bridge experiment |

## Auto-downloaded at runtime (HuggingFace hub cache, no action needed)

- `neuronpedia/jacobian-lens` — pre-fitted J_l matrices per model
  (`jlens_enact_probe.py` pulls via `hf_hub_download`; ~700 MB per model).
- All model weights (`hf_logprobs.MODELS`), gated ones need `HF_TOKEN`
  with Gemma/Llama access approved.
- sentence-transformers encoders (`all-mpnet-base-v2`, GloVe/komninos
  averages, `BAAI/bge-large-en-v1.5`, `dwulff/mpnet-personality`) — pulled
  on first use by the W16 regress and VP-transparency scripts.

## Manual retrieval → `data/` (gitignored; ask rgb for a copy)

| dataset | where | notes |
|---|---|---|
| `data/escs_525pda/` | Harvard Dataverse doi:10.7910/DVN/GHYMEV | Saucier 525-PDA raw self-ratings (700×525); drop 2 corrupted cols → the 523 set. Basis of HUMAN everywhere. |
| `data/escs_360pda/`, `data/escs_sdv/`, `data/escs_selfpeer/` | same Dataverse deposit | companion ESCS files |
| `data/kajonius_johnson_2019/` | PsychArchives (K&J 2019 deposit) | IPIP300.por / IPIP120.por — **stored pre-reversed, do NOT re-apply 6−x** (see memory/W13 notes) |
| `data/cutler_condon_2022/` | SAPA/Dataverse per paper | ipsatized adjective data (C&C, S&G-1996 lineage) |
| `data/neuroquest_ipip/` | per source paper | — |

Registration walls prevent scripting the Dataverse/PsychArchives pulls;
grab them in a browser and drop the files in the paths above.
