"""Calibration registers of the comparative pair (rgb 2026-09-14): per model,
P(4) vs entropy over the six comparative prompts (44 medoids each). Three
corners: decisive rejection (high P(4), low H), don't-know (low P(4), high H),
confident comparison (low P(4), low H). Point size = comparative spread (d sd);
color = family. Also the same for the direct framing, small, for contrast.
Usage: .venv/bin/python scripts/fig_compare_registers.py
"""
import glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pkit

files = sorted(glob.glob("results/adjectives/refgroup/*_pair.json")); runs, seen = [], set()
for p in files:
    d = json.load(open(p)); repo = pkit.roster.MODELS.get(d["model"], d["model"]).split("/")[-1]
    if repo in seen: continue
    seen.add(repo); d["repo"] = repo; runs.append(d)
A = runs[0]["adjectives"]; comp = [f"{s}_{k}" for k in ["assistant", "ai", "lm"] for s in ["more", "less"]]
FAM = [("gemma", "Gemma"), ("Qwen", "Qwen"), ("Llama|Tulu", "Llama"), ("Mistral|Ministral|Nemo", "Mistral"), ("Phi", "Phi"),
       ("aya|command", "Cohere"), ("Yi", "Yi"), ("OLMo", "OLMo"), ("Falcon|falcon", "Falcon"), ("granite", "Granite"), (".", "other")]
fam = lambda n: next(f for pat, f in FAM if re.search(pat, n))
COL = {f: c for (_, f), c in zip(FAM, plt.cm.tab10(np.linspace(0, 1, 11)))}
pts = []
for d in runs:
    R = d["results"]
    P4 = np.mean([float(R[v][a]["dist"].get("4", 0)) for v in comp for a in A]); H = np.mean([R[v][a]["entropy"] for v in comp for a in A])
    P4d = np.mean([float(R["direct"][a]["dist"].get("4", 0)) for a in A]); Hd = np.mean([R["direct"][a]["entropy"] for a in A])
    dd = np.mean([(np.array([R[f"more_{k}"][a]["ev"] for a in A]) - np.array([R[f"less_{k}"][a]["ev"] for a in A])) / 2 for k in ["assistant", "ai", "lm"]], 0)
    sz = re.findall(r"(\d+(?:\.\d+)?)[bB]", d["repo"]); sz = float(sz[-1]) if sz else 4.0
    pts.append((d["repo"], fam(d["repo"]), P4, H, P4d, Hd, dd.std(), sz))
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for n, f, P4, H, _, _, sd, sz in pts:
    ax.scatter(H, P4, s=30 + 220 * sd, c=[COL[f]], alpha=.75, edgecolor="k", lw=.4)
    ax.annotate(n.replace("-Instruct", "").replace("-instruct", "").replace("-Chat", "").replace("-it", ""), (H, P4), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
ax.axhline(.75, color="k", lw=.5, ls=":"); ax.axvline(1.2, color="k", lw=.5, ls=":")
ax.text(0.30, .93, "decisive rejection (peaked 4)", fontsize=8, va="top", transform=ax.transAxes, style="italic")
ax.text(0.98, .12, "don't know (near-uniform)", fontsize=8, ha="right", transform=ax.transAxes, style="italic")
ax.text(0.30, .06, "confident comparison", fontsize=8, transform=ax.transAxes, style="italic")
ax.set_xlabel("mean entropy of the comparative digit distribution (nats; max 1.95)"); ax.set_ylabel("mean P(4 = neither) over the six comparative prompts")
ax.set_xlim(-.05, 2.0); ax.set_ylim(-.03, 1.03); ax.set_title("Comparative pair: three registers (size = spread of more-or-less self)", fontsize=9)
for f in sorted({p[1] for p in pts}): ax.scatter([], [], c=[COL[f]], label=f, s=30, edgecolor="k", lw=.4)
ax.legend(fontsize=6.5, loc="center right", frameon=False)
ax = axes[1]
for n, f, P4, H, P4d, Hd, sd, sz in pts:
    ax.annotate("", xy=(H, P4), xytext=(Hd, P4d), arrowprops=dict(arrowstyle="->", color=COL[f], lw=.7, alpha=.7))
    ax.scatter(Hd, P4d, s=12, c=[COL[f]], alpha=.6)
ax.axhline(.75, color="k", lw=.5, ls=":"); ax.axvline(1.2, color="k", lw=.5, ls=":")
ax.set_xlim(-.05, 2.0); ax.set_ylim(-.03, 1.03); ax.set_xlabel("entropy"); ax.set_title("direct framing -> comparative (arrow per model)", fontsize=9)
fig.tight_layout(); out = "results/persona_vectors/figs/fig_compare_registers"
fig.savefig(out + ".pdf", dpi=300); fig.savefig(out + ".png", dpi=150); print("wrote", out)
# counts
dec = sum(1 for p in pts if p[2] >= .75); dk = sum(1 for p in pts if p[2] < .75 and p[3] >= 1.2); conf = len(pts) - dec - dk
print(f"decisive rejection {dec}, don't-know {dk}, confident comparison {conf} (of {len(pts)})")
print("direct-framing registers for contrast:", sum(1 for p in pts if p[4] >= .75), sum(1 for p in pts if p[4] < .75 and p[5] >= 1.2))
