"""Build the paper's small/large cohort tables from cohort100_manifest.

Merges the wide-n manifest with the standing deep-10 (hf.MODELS), computes
per-model channel coverage from disk, applies the 2026-08-22/23 breakage
taxonomy, and emits into ../personality-paper/:
  _data/cohort_models.csv       one row per model, machine-readable
  _data/cohort_table_small.md   deep-10 table (main text)
  _data/cohort_table_large.md   wide cohort table (appendix)
  refs.bib                      appends model tech-report entries (deduped)

Status values: ok | excluded:<reason> | flagged:<reason>. The R1 rows are
"flagged" not "excluded" — decision postponed to statisfactions (to_try
2026-08-23). Glimmer is ok-with-note: prefill SELF row excluded
(mode-broken), think-arm row replaces it.

Usage: PYTHONPATH=scripts python scripts/build_paper_cohort.py
"""
import csv
import glob
import json
import os

import hf_logprobs as hf

PAPER = os.path.expanduser("~/src/personality-paper")
MANIFEST = "instruments/cohort100_manifest.json"

DEEP = {  # standing deep cohort: short name -> repo (all five channels)
    n: hf.MODELS[n] for n in
    ["llama3.2", "qwen2.5", "gemma3", "phi4", "Llama8", "Qwen7", "Qwen32",
     "Gemma12", "Gemma27", "Aya"]
}
EXTRA_WIDE = {  # standing models in the wide-SELF collection, not in manifest
    "FalconMamba": ("tiiuae/falcon-mamba-7b-instruct", "falcon", 7, 2024, []),
}

STATUS = {
    "internlm/internlm2_5-7b-chat":
        "excluded:instrument-broken (adjective-invariant digit distributions)",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
        "flagged:respondent-absent (framing-incoherent at every grain; "
        "pending statisfactions)",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":
        "flagged:respondent-absent (framing-incoherent at every grain; "
        "pending statisfactions)",
    "meta-models/Muse-Glimmer-30B":
        "ok:think-arm row (prefill SELF row mode-broken, excluded)",
    "tiiuae/falcon-7b-instruct":
        "flagged:flat-row leverage (spread 0.16; single-handedly rotates "
        "ipsatized shape axes, LOO 1-|r|=0.90 on iPC4 — benched from "
        "shape analyses)",
}

# repo (or prefix) -> bibkey; entries below
CITES = [
    ("meta-llama/Llama-2-", "touvronLlama2Open2023"),
    ("meta-llama/Meta-Llama-3-", "grattafioriLlama3Herd2024"),
    ("meta-llama/Llama-3.", "grattafioriLlama3Herd2024"),
    ("lmsys/vicuna", "chiangVicunaOpenSource2023"),
    ("allenai/Llama-3.1-Tulu-3", "lambertTulu3Pushing2024"),
    ("deepseek-ai/DeepSeek-R1-", "deepseekaiDeepSeekR1Incentivizing2025"),
    ("Qwen/Qwen1.5-", "qwenteamIntroducingQwen152024"),
    ("Qwen/Qwen2-", "yangQwen2Technical2024"),
    ("Qwen/Qwen2.5-", "yangQwen25Technical2024"),
    ("Qwen/Qwen3-", "yangQwen3Technical2025"),
    ("Qwen/Qwen3.8-", "qwenteamQwen38Model2026"),
    ("google/gemma-2b", "gemmateamGemmaOpen2024"),
    ("google/gemma-7b", "gemmateamGemmaOpen2024"),
    ("google/gemma-2-", "gemmateamGemma2Improving2024"),
    ("google/gemma-3-", "gemmateamGemma3Technical2025"),
    ("google/gemma-4-", "gemmateamGemma4Model2026"),
    ("microsoft/Phi-3", "abdinPhi3Technical2024"),
    ("microsoft/phi-4", "abdinPhi4Technical2024"),
    ("microsoft/Phi-4-mini", "aboueleninPhi4MiniTechnical2025"),
    ("mistralai/Mistral-7B-", "jiangMistral7B2023"),
    ("mistralai/Ministral-", "mistralaiUnMinistralDes2024"),
    ("mistralai/Mistral-Nemo-", "mistralaiMistralNeMo2024"),
    ("mistralai/Mistral-Small-24B", "mistralaiMistralSmall32025"),
    ("HuggingFaceH4/zephyr", "tunstallZephyrDirect2023"),
    ("CohereLabs/aya-expanse", "dangAyaExpanseCombining2024"),
    ("CohereLabs/c4ai-command-r7b", "cohereIntroducingCommandR7B2024"),
    ("allenai/OLMo-2-", "olmoteam2OLMo2Furious2025"),
    ("tiiuae/falcon-7b", "almazroueiFalconSeries2023"),
    ("tiiuae/Falcon3-", "tiiteamFalcon3Family2024"),
    ("tiiuae/falcon-mamba", "zuoFalconMambaFirst2024"),
    ("THUDM/glm-4-", "glmteamChatGLMFamily2024"),
    ("01-ai/Yi-1.5-", "youngYiOpenFoundation2024"),
    ("internlm/internlm2_5", "caiInternLM2Technical2024"),
    ("internlm/internlm3", "internlmteamInternLM3Model2025"),
    ("ibm-granite/granite-3", "graniteteamGranite3Language2024"),
    ("HuggingFaceTB/SmolLM2", "allalSmolLM2Smol2025"),
    ("stabilityai/stablelm-2", "bellagenteStableLM216B2024"),
    ("openbmb/MiniCPM3", "huMiniCPMUnveiling2024"),
    ("LGAI-EXAONE/EXAONE-3.5", "lgairesearchEXAONE35Series2024"),
    ("nvidia/Llama-3.1-Nemotron", "nvidiaLlamaNemotronEfficient2025"),
    ("meta-models/Muse-Glimmer", "metamodelsMuseGlimmer2026"),
]

# Display-name convention: {name}{version}-{size} (rgb). Overrides for rows
# whose manifest/short name doesn't already conform; default = row name.
DISPLAY = {
    "llama3.2": "Llama3.2-3B", "qwen2.5": "Qwen2.5-3B", "gemma3": "Gemma3-4B",
    "phi4": "Phi4-3.8B", "Qwen7": "Qwen2.5-7B", "Qwen32": "Qwen2.5-32B",
    "Gemma12": "Gemma3-12B", "Gemma27": "Gemma3-27B", "Aya": "Aya-8B",
    "Phi3-mini": "Phi3-3.8B", "Phi3-medium": "Phi3-14B",
    "Phi3.5-mini": "Phi3.5-3.8B", "Phi4": "Phi4-14B",
    "Vicuna-7B": "Vicuna1.5-7B", "R1-Llama8": "R1-Llama-8B",
    "R1-Qwen7": "R1-Qwen-7B", "Mistral7B-v0.1": "Mistral0.1-7B",
    "Mistral7B-v0.3": "Mistral0.3-7B", "Mistral-Nemo": "MistralNemo-12B",
    "Mistral-Small24": "MistralSmall-24B", "CommandR7B": "CommandR-7B",
    "Falcon-7B": "Falcon1-7B", "FalconMamba": "FalconMamba-7B",
    "Nemotron-Nano8B": "NemotronNano-8B",
}

BIB = r"""
% --- model tech reports (generated by ../personality/scripts/build_paper_cohort.py) ---
@online{touvronLlama2Open2023,
  author = {Touvron, Hugo and Martin, Louis and Stone, Kevin and others},
  title = {Llama 2: Open Foundation and Fine-Tuned Chat Models},
  year = {2023}, eprint = {2307.09288}, eprinttype = {arxiv}}
@online{grattafioriLlama3Herd2024,
  author = {Grattafiori, Aaron and Dubey, Abhimanyu and Jauhri, Abhinav and others},
  title = {The Llama 3 Herd of Models},
  year = {2024}, eprint = {2407.21783}, eprinttype = {arxiv}}
@online{chiangVicunaOpenSource2023,
  author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and others},
  title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
  year = {2023}, url = {https://lmsys.org/blog/2023-03-30-vicuna/}}
@online{lambertTulu3Pushing2024,
  author = {Lambert, Nathan and Morrison, Jacob and Pyatkin, Valentina and others},
  title = {T\"ulu 3: Pushing Frontiers in Open Language Model Post-Training},
  year = {2024}, eprint = {2411.15124}, eprinttype = {arxiv}}
@online{deepseekaiDeepSeekR1Incentivizing2025,
  author = {{DeepSeek-AI}},
  title = {DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  year = {2025}, eprint = {2501.12948}, eprinttype = {arxiv}}
@online{qwenteamIntroducingQwen152024,
  author = {{Qwen Team}},
  title = {Introducing Qwen1.5},
  year = {2024}, url = {https://qwenlm.github.io/blog/qwen1.5/}}
@online{yangQwen2Technical2024,
  author = {Yang, An and Yang, Baosong and Hui, Binyuan and others},
  title = {Qwen2 Technical Report},
  year = {2024}, eprint = {2407.10671}, eprinttype = {arxiv}}
@online{yangQwen25Technical2024,
  author = {Yang, An and Yang, Baosong and Zhang, Beichen and others},
  title = {Qwen2.5 Technical Report},
  year = {2024}, eprint = {2412.15115}, eprinttype = {arxiv}}
@online{yangQwen3Technical2025,
  author = {Yang, An and Li, Anfeng and Yang, Baosong and others},
  title = {Qwen3 Technical Report},
  year = {2025}, eprint = {2505.09388}, eprinttype = {arxiv}}
@online{qwenteamQwen38Model2026,
  author = {{Qwen Team}},
  title = {Qwen3.8-27B Model Card},
  year = {2026}, url = {https://huggingface.co/Qwen/Qwen3.8-27B}}
@online{gemmateamGemmaOpen2024,
  author = {{Gemma Team}},
  title = {Gemma: Open Models Based on Gemini Research and Technology},
  year = {2024}, eprint = {2403.08295}, eprinttype = {arxiv}}
@online{gemmateamGemma2Improving2024,
  author = {{Gemma Team}},
  title = {Gemma 2: Improving Open Language Models at a Practical Size},
  year = {2024}, eprint = {2408.00118}, eprinttype = {arxiv}}
@online{gemmateamGemma3Technical2025,
  author = {{Gemma Team}},
  title = {Gemma 3 Technical Report},
  year = {2025}, eprint = {2503.19786}, eprinttype = {arxiv}}
@online{gemmateamGemma4Model2026,
  author = {{Gemma Team}},
  title = {Gemma 4 Model Card},
  year = {2026}, url = {https://huggingface.co/google/gemma-4-31B-it}}
@online{abdinPhi3Technical2024,
  author = {Abdin, Marah and Aneja, Jyoti and Awadalla, Hany and others},
  title = {Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone},
  year = {2024}, eprint = {2404.14219}, eprinttype = {arxiv}}
@online{abdinPhi4Technical2024,
  author = {Abdin, Marah and Aneja, Jyoti and Behl, Harkirat and others},
  title = {Phi-4 Technical Report},
  year = {2024}, eprint = {2412.08905}, eprinttype = {arxiv}}
@online{aboueleninPhi4MiniTechnical2025,
  author = {Abouelenin, Abdelrahman and Ashfaq, Atabak and Atkinson, Adam and others},
  title = {Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs},
  year = {2025}, eprint = {2503.01743}, eprinttype = {arxiv}}
@online{jiangMistral7B2023,
  author = {Jiang, Albert Q. and Sablayrolles, Alexandre and Mensch, Arthur and others},
  title = {Mistral 7B},
  year = {2023}, eprint = {2310.06825}, eprinttype = {arxiv}}
@online{mistralaiUnMinistralDes2024,
  author = {{Mistral AI}},
  title = {Un Ministral, des Ministraux},
  year = {2024}, url = {https://mistral.ai/news/ministraux}}
@online{mistralaiMistralNeMo2024,
  author = {{Mistral AI}},
  title = {Mistral NeMo},
  year = {2024}, url = {https://mistral.ai/news/mistral-nemo}}
@online{mistralaiMistralSmall32025,
  author = {{Mistral AI}},
  title = {Mistral Small 3},
  year = {2025}, url = {https://mistral.ai/news/mistral-small-3}}
@online{tunstallZephyrDirect2023,
  author = {Tunstall, Lewis and Beeching, Edward and Lambert, Nathan and others},
  title = {Zephyr: Direct Distillation of LM Alignment},
  year = {2023}, eprint = {2310.16944}, eprinttype = {arxiv}}
@online{dangAyaExpanseCombining2024,
  author = {Dang, John and Singh, Shivalika and D'souza, Daniel and others},
  title = {Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier},
  year = {2024}, eprint = {2412.04261}, eprinttype = {arxiv}}
@online{cohereIntroducingCommandR7B2024,
  author = {{Cohere}},
  title = {Introducing Command R7B: Fast and Efficient Generative AI},
  year = {2024}, url = {https://cohere.com/blog/command-r7b}}
@online{olmoteam2OLMo2Furious2025,
  author = {{OLMo Team}},
  title = {2 OLMo 2 Furious},
  year = {2025}, eprint = {2501.00656}, eprinttype = {arxiv}}
@online{almazroueiFalconSeries2023,
  author = {Almazrouei, Ebtesam and Alobeidli, Hamza and Alshamsi, Abdulaziz and others},
  title = {The Falcon Series of Open Language Models},
  year = {2023}, eprint = {2311.16867}, eprinttype = {arxiv}}
@online{tiiteamFalcon3Family2024,
  author = {{Falcon-LLM Team}},
  title = {The Falcon 3 Family of Open Models},
  year = {2024}, url = {https://huggingface.co/blog/falcon3}}
@online{zuoFalconMambaFirst2024,
  author = {Zuo, Jingwei and Velikanov, Maksim and Rhaiem, Dhia Eddine and others},
  title = {Falcon Mamba: The First Competitive Attention-free 7B Language Model},
  year = {2024}, eprint = {2410.05355}, eprinttype = {arxiv}}
@online{glmteamChatGLMFamily2024,
  author = {{GLM Team}},
  title = {ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools},
  year = {2024}, eprint = {2406.12793}, eprinttype = {arxiv}}
@online{youngYiOpenFoundation2024,
  author = {Young, Alex and Chen, Bei and Li, Chao and others},
  title = {Yi: Open Foundation Models by 01.AI},
  year = {2024}, eprint = {2403.04652}, eprinttype = {arxiv}}
@online{caiInternLM2Technical2024,
  author = {Cai, Zheng and Cao, Maosong and Chen, Haojiong and others},
  title = {InternLM2 Technical Report},
  year = {2024}, eprint = {2403.17297}, eprinttype = {arxiv}}
@online{internlmteamInternLM3Model2025,
  author = {{InternLM Team}},
  title = {InternLM3-8B-Instruct Model Card},
  year = {2025}, url = {https://huggingface.co/internlm/internlm3-8b-instruct}}
@online{graniteteamGranite3Language2024,
  author = {{Granite Team, IBM}},
  title = {Granite 3.0 Language Models},
  year = {2024}, url = {https://github.com/ibm-granite/granite-3.0-language-models}}
@online{allalSmolLM2Smol2025,
  author = {Allal, Loubna Ben and Lozhkov, Anton and Bakouch, Elie and others},
  title = {SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model},
  year = {2025}, eprint = {2502.02737}, eprinttype = {arxiv}}
@online{bellagenteStableLM216B2024,
  author = {Bellagente, Marco and Tow, Jonathan and Mahan, Dakota and others},
  title = {Stable LM 2 1.6B Technical Report},
  year = {2024}, eprint = {2402.17834}, eprinttype = {arxiv},
  note = {The 12B variant is documented in the model card only}}
@online{huMiniCPMUnveiling2024,
  author = {Hu, Shengding and Tu, Yuge and Han, Xu and others},
  title = {MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies},
  year = {2024}, eprint = {2404.06395}, eprinttype = {arxiv}}
@online{lgairesearchEXAONE35Series2024,
  author = {{LG AI Research}},
  title = {EXAONE 3.5: Series of Large Language Models for Real-world Use Cases},
  year = {2024}, eprint = {2412.04862}, eprinttype = {arxiv}}
@online{nvidiaLlamaNemotronEfficient2025,
  author = {{NVIDIA}},
  title = {Llama-Nemotron: Efficient Reasoning Models},
  year = {2025}, eprint = {2505.00949}, eprinttype = {arxiv}}
@online{metamodelsMuseGlimmer2026,
  author = {{Meta Models}},
  title = {Muse Glimmer 30B Model Card},
  year = {2026}, url = {https://huggingface.co/meta-models/Muse-Glimmer-30B}}
"""


def cite_for(repo):
    for prefix, key in CITES:
        if repo.startswith(prefix):
            return key
    return ""


# Channel letters: S SELF, R REPRESENT, J JUDGE, E persona ENACT (the
# cohort-10 write-side extraction), D default-persona rollouts (the wide-n
# __default__ capture — a different, weaker object). D is hidden from the
# tables for now (rgb 2026-08-26); flip to True to show it.
SHOW_DEFAULT_ENACT = False


def coverage(repo, alt=None, deep=False):
    names = [repo.replace("/", "_")] + ([alt] if alt else [])
    pats = [("S", "results/adjectives/selfreport/{n}_self_full.json"),
            ("R", "results/adjectives/acts/{n}__pers.pt"),
            ("T", "results/adjectives/selfreport/{n}_self_full_think*.json")]
    if SHOW_DEFAULT_ENACT:
        pats.insert(2, ("D", "results/cohort100/default_enact/{n}_default.pt"))
    ch = [tag for tag, pat in pats
          if any(glob.glob(pat.format(n=n)) for n in names)]
    if deep:            # cohort-10 program: JUDGE + persona ENACT captured
        ch = [c for c in ch if c not in "JE"]
        ch.insert(min(2, len(ch)), "J")
        ch.insert(min(3, len(ch)), "E")
    return "".join(ch)


def think_mode(flags):
    if "always_think" in flags:
        return "always"
    if "reasoning_default_off" in flags:
        return "toggle (off by default)"
    if "thinking" in flags:
        return "hybrid"
    return "none"


def main():
    man = json.load(open(MANIFEST))["models"]
    deep_repos = set(DEEP.values())
    rows = []
    seen = set()
    for e in man:
        repo = e["repo"]
        seen.add(repo)
        status = STATUS.get(repo, "ok")
        rows.append({
            "name": e["name"], "repo": repo, "family": e["family"],
            "params_b": e["params_b"], "generation": e.get("generation", ""),
            "cohort": "deep+wide" if repo in deep_repos else "wide",
            "channels": coverage(repo, deep=repo in deep_repos),
            "thinking": think_mode(e.get("flags", [])),
            "status": status, "cite": cite_for(repo)})
    DEEP_META = {  # family, params_b, generation for standing-cohort rows
        "llama3.2": ("llama", 3, 2024), "qwen2.5": ("qwen", 3, 2024),
        "gemma3": ("gemma", 4, 2025), "phi4": ("phi", 3.8, 2025),
        "Llama8": ("llama", 8, 2024), "Qwen7": ("qwen", 7, 2024),
        "Qwen32": ("qwen", 32, 2024), "Gemma12": ("gemma", 12, 2025),
        "Gemma27": ("gemma", 27, 2025), "Aya": ("cohere", 8, 2024),
    }
    for short, repo in DEEP.items():
        if repo in seen:
            continue
        fam, pb, gen = DEEP_META[short]
        # deep models keep short-name result files; all five channels were
        # captured in the cohort-10 program (JUDGE/ENACT included)
        rows.append({
            "name": short, "repo": repo, "family": fam,
            "params_b": pb, "generation": gen,
            "cohort": "deep+wide", "channels": "SRJE", "thinking": "none",
            "status": "ok", "cite": cite_for(repo)})
    for short, (repo, fam, pb, gen, flags) in EXTRA_WIDE.items():
        rows.append({
            "name": short, "repo": repo, "family": fam, "params_b": pb,
            "generation": gen, "cohort": "wide",
            "channels": coverage(repo, alt=short),
            "thinking": think_mode(flags), "status": "ok",
            "cite": cite_for(repo)})
    for r in rows:
        r["display"] = DISPLAY.get(r["name"], r["name"])
    # family -> year -> citekey -> size, so rows sharing a citekey sit
    # adjacent and the blank-repeat convention below reads as a ditto
    rows.sort(key=lambda r: (r["family"], str(r["generation"]), r["cite"],
                             float(r["params_b"] or 0)))

    os.makedirs(f"{PAPER}/_data", exist_ok=True)
    with open(f"{PAPER}/_data/cohort_models.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> _data/cohort_models.csv")
    missing = [r["name"] for r in rows if not r["cite"]]
    if missing:
        print("MISSING CITES:", missing)

    deep_rows = [r for r in rows if r["cohort"] == "deep+wide"]
    wide_rows = [r for r in rows if "wide" in r["cohort"]]

    def cite_cell(r, seen):
        if not r["cite"] or r["cite"] in seen:
            return ""
        seen.add(r["cite"])
        return f"[@{r['cite']}]"

    def render_pipe(header, body_rows, cap=None):
        """Pipe table with content-proportional delimiter widths: pandoc
        takes relative column widths from the DASH COUNTS in the delimiter
        row when lines run long, so equal dashes would mean equal-width
        p{} columns. cap[i] (optional) limits column i's dash count — used
        where the rendered content is shorter than the source (citekeys
        render as "(Author, year)")."""
        cols = list(zip(*([header] + body_rows)))
        w = [max(3, max(len(c) for c in col)) for col in cols]
        d = [min(w[i], cap[i]) if cap and cap[i] else w[i]
             for i in range(len(w))]
        def line(cells):
            return "| " + " | ".join(c.ljust(w[i])
                                     for i, c in enumerate(cells)) + " |"
        out = [line(header), "|" + "|".join("-" * (d[i] + 2)
                                            for i in range(len(w))) + "|"]
        out += [line(r) for r in body_rows]
        return out

    seen = set()
    body = [[r["display"], r["family"], str(r["params_b"]), r["thinking"],
             cite_cell(r, seen)] for r in deep_rows]
    lines = render_pipe(["Model", "Family", "Params (B)", "Thinking",
                         "Citation"], body, cap=[0, 0, 0, 0, 26])
    lines.append("\n: The deep cohort: all five channels (SELF, REPRESENT, "
                 "JUDGE, ENACT, and rollout text). Citations are shown at "
                 "first occurrence. {#tbl-cohort-small}\n")
    with open(f"{PAPER}/_data/cohort_table_small.md", "w") as fp:
        fp.write("\n".join(lines))

    THINK_ABBR = {"none": "—", "hybrid": "hybrid", "always": "always",
                  "toggle (off by default)": "toggle"}
    seen = set()
    body = [[r["display"], str(r["generation"]),
             r["channels"], THINK_ABBR[r["thinking"]],
             r["status"].split(":")[0], cite_cell(r, seen)]
            for r in wide_rows]
    lines = render_pipe(["Model", "Gen", "Channels", "Think", "Status",
                         "Citation"], body, cap=[0, 0, 0, 0, 0, 26])
    chan_legend = ("Channels: S = SELF, R = REPRESENT, J = JUDGE, E = persona "
                   "ENACT, T = thinking-mode SELF arm"
                   + (", D = default-persona rollouts" if SHOW_DEFAULT_ENACT
                      else "") + ". ")
    lines.append("\n: The wide cohort. " + chan_legend + "Status: excluded/flagged "
                 "rows per the measurement-failure taxonomy (see text). "
                 "Citations are shown at first occurrence. "
                 "{#tbl-cohort-large}\n")
    with open(f"{PAPER}/_data/cohort_table_large.md", "w") as fp:
        fp.write("\n".join(lines))
    print("wrote _data/cohort_table_small.md, _data/cohort_table_large.md")

    refs = open(f"{PAPER}/refs.bib").read()
    new_keys = [l.split("{")[1].rstrip(",\n")
                for l in BIB.splitlines() if l.startswith("@")]
    if not any(k in refs for k in new_keys):
        with open(f"{PAPER}/refs.bib", "a") as fp:
            fp.write(BIB)
        print(f"appended {len(new_keys)} entries to refs.bib")
    else:
        present = [k for k in new_keys if k in refs]
        print(f"refs.bib already has {len(present)} of these keys — not appending")


if __name__ == "__main__":
    main()
