"""Compare induction FRAMES for the assistant-drift instrument (W17):
human-anchor (subtleA) vs no-anchor (subtleB) vs HHH-assistant, on traits
chosen to expose the split — anti-assistant (cruel/dishonest/lazy/rude),
pro-HHH (kind/honest/careful), normalcy (normal/abnormal/strange), eval
anchors (wonderful/awful).

Key questions:
  - does the HHH frame CHOKE on anti-HHH traits (weak ‖dir‖ / refusals in
    text) while the human anchor expresses them?
  - cos(dir, assistant axis): assistant > no-anchor > human?
  - which frame keeps the cleanest trait discrimination (cos(W,A))?

Usage:
  PYTHONPATH=scripts .venv/bin/python scripts/compare_frames.py
"""

import json
import re
from pathlib import Path

import numpy as np
import torch

PV = Path("results/persona_vectors")
STAGE = re.compile(r"(?<!\*)\*[^*\n]{1,60}\*(?!\*)")
FRAMES = [("human", "framecmp_subtleA"), ("no-anchor", "framecmp_subtleB"),
          ("HHH-asst", "framecmp_assistant")]
CATS = {"anti-assistant": ["cruel", "dishonest", "lazy", "rude"],
        "pro-HHH": ["kind", "honest", "careful"],
        "normalcy": ["normal", "abnormal", "strange"],
        "eval": ["wonderful", "awful"]}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    data, texts = {}, {}
    for name, tag in FRAMES:
        p = PV / f"llama3.2_{tag}.pt"
        if not p.exists():
            print(f"missing {name}: {p.name}")
            continue
        data[name] = torch.load(p, weights_only=False)
        texts[name] = json.load(open(PV / f"llama3.2_{tag}_texts.json"))
    frames = list(data)
    mid = data[frames[0]]["report"]["mid_layer"]

    # per-category magnitude / drift / on-axis, per frame
    print(f"mid layer {mid}\n")
    hdr = "category".ljust(16) + "".join(f"{f:>22s}" for f in frames)
    print(hdr)
    print("(each cell: mean‖dir‖ | drift | cos→asst)")
    for cat, adjs in CATS.items():
        row = cat.ljust(16)
        for f in frames:
            d = data[f]
            aa = unit(d["assistant_axis"][mid].astype(np.float64))
            present = [a for a in adjs if a in d["directions"]]
            mags, drifts, cass = [], [], []
            for a in present:
                v = d["directions"][a][mid].astype(np.float64)
                mags.append(np.linalg.norm(v))
                drifts.append(np.linalg.norm(v - d["assistant_axis"][mid].astype(np.float64)))
                cass.append(abs(unit(v) @ aa))
            row += f"  {np.mean(mags):4.2f}|{np.mean(drifts):4.2f}|{np.mean(cass):+.2f}"
        print(row)

    # cos(W,A) and theater per frame
    print()
    for f in frames:
        d = data[f]
        wa = (unit(d["directions"]["wonderful"][mid].astype(np.float64))
              @ unit(d["directions"]["awful"][mid].astype(np.float64)))
        rows = [r for c, rs in texts[f].items() if c != "__default__" for r in rs]
        th = np.mean([bool(STAGE.search(r["text"])) for r in rows])
        leak = np.mean([r.get("leak", False) for r in rows])
        print(f"  {f:>10s}: cos(W,A)={wa:+.3f}  theater={th:.2f}  leak={leak:.2f}")

    # per-adjective ‖dir‖ for anti-assistant traits (the choke test)
    print("\nanti-assistant trait magnitudes (choke test):")
    print("adj".ljust(12) + "".join(f"{f:>12s}" for f in frames))
    for a in CATS["anti-assistant"]:
        row = a.ljust(12)
        for f in frames:
            v = data[f]["directions"].get(a)
            row += f"{np.linalg.norm(v[mid]):12.2f}" if v is not None else f"{'--':>12s}"
        print(row)

    # sample dishonest texts per frame (does HHH refuse?)
    print("\n--- 'dishonest' sample responses ---")
    for f in frames:
        print(f"[{f}]")
        for r in texts[f].get("dishonest", [])[:2]:
            print("  " + r["text"][:150].replace("\n", " "))


if __name__ == "__main__":
    main()
