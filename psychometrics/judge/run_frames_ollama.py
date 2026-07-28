#!/usr/bin/env python3
"""Frame experiment for JUDGE Report VI §8.2 — Orin/Ollama inference.

Holds the adjective pair fixed and varies the QUESTION FRAME, in both orders:

  cond  Consider a person who is very {a}.
        How likely is this person to also be {b}?            <- rgb's original tom_likely
  sim   How similar is a person who is very {a} to a person who is very {b}?
  diff  How different is a person who is very {a} from a person who is very {b}?

All three use the same 1-7 response instruction and the same system message, so any
frame difference cannot come from the answer format. Grammatical subject = Tversky's
"subject" role; running both orders is the focusing manipulation.

Why: everything in Report VI is a statement about (beta - alpha)*f. The frames separate
the weights, and they adjudicate the deflationary rival that B[i,j] is just P(j|i):
  - conditional-probability account: `diff` is `sim` reverse-scored, nothing more, so
    sim + reverse(diff) is constant and the asymmetries match after reflection;
  - contrast model: theta/alpha/beta are re-weighted between similarity and difference,
    so sim + diff varies systematically with the pair's feature richness (Tversky's
    non-complementarity), and `cond` need not behave like `sim` at all.

Output: JSONL, one record per prompt, appended incrementally so the run resumes after
an interruption (re-run the same command; already-completed keys are skipped).

Usage:
  export OLLAMA_API_KEY=$(grep OLLAMA_API_KEY .env | cut -d= -f2- | tr -d "'\\"")
  python3 run_frames_ollama.py --model gemma3:4b
  python3 run_frames_ollama.py --model gemma3:4b --pairs 10 --frames sim,diff   # pilot
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# Raw collected inference output: NOT under data/ (which is gitignored and regenerable) --
# this costs an Orin run to reproduce, so it is tracked.
OUT = os.path.join(HERE, "frames_raw")
REMOTE_URL = "https://apollo.quocanmeomeo.io.vn"

SYS_MSG = "Respond with ONLY a single integer from 1 to 7."
SCALE = {
    "cond": "Answer with one number from 1 to 7, where 1 = very unlikely and 7 = very likely.",
    "sim":  "Answer with one number from 1 to 7, where 1 = very dissimilar and 7 = very similar.",
    "diff": "Answer with one number from 1 to 7, where 1 = not at all different and 7 = extremely different.",
}
BODY = {
    "cond": "Consider a person who is very {a}.\nHow likely is this person to also be {b}?",
    "sim":  "How similar is a person who is very {a} to a person who is very {b}?",
    "diff": "How different is a person who is very {a} from a person who is very {b}?",
}
FRAMES = list(BODY)
OPTS = [str(k) for k in range(1, 8)]


def build_prompt(frame, a, b):
    return f"{BODY[frame].format(a=a, b=b)}\n{SCALE[frame]}\nNumber:"


def call_remote(model, prompt, api_key, timeout=120, top_logprobs=20):
    """One chat completion via curl (compact JSON payload; see CLAUDE.md gotchas)."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYS_MSG},
                     {"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
        "max_tokens": 4,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    cmd = ["curl", "-s", "-m", str(timeout),
           "-H", f"Authorization: Bearer {api_key}",
           "-H", "Content-Type: application/json",
           "-d", json.dumps(payload, separators=(",", ":")),
           f"{REMOTE_URL}/v1/chat/completions"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None, f"curl rc={res.returncode}"
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None, f"bad json: {res.stdout[:200]}"
    if "choices" not in data:
        return None, f"no choices: {res.stdout[:200]}"
    choice = data["choices"][0]
    return {
        "text": (choice["message"]["content"] or "").strip(),
        "logprobs": choice.get("logprobs", {}).get("content", []) or [],
    }, None


def distribution(resp):
    """Renormalized P over '1'..'7' from the first scored token, plus its argmax.

    Falls back to parsing the generated text when no logprobs come back."""
    dist, argmax = None, None
    for entry in resp["logprobs"]:
        cands = entry.get("top_logprobs", [])
        if not cands:
            continue
        p = {}
        for c in cands:
            tok = (c.get("token") or "").strip()
            if tok in OPTS:
                p[tok] = max(p.get(tok, 0.0), __import__("math").exp(c["logprob"]))
        if p:
            tot = sum(p.values())
            dist = {k: p.get(k, 0.0) / tot for k in OPTS}
            argmax = max(dist, key=dist.get)
            return dist, argmax, tot
        break                       # only the first generated token is the answer
    txt = "".join(ch for ch in resp["text"] if ch.isdigit())
    if txt and txt[0] in OPTS:
        argmax = txt[0]
    return dist, argmax, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", type=int, default=0, help="limit pairs (pilot)")
    ap.add_argument("--frames", default=",".join(FRAMES))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        sys.exit("OLLAMA_API_KEY not set")
    frames = [f.strip() for f in args.frames.split(",")]
    assert all(f in BODY for f in frames), frames

    import csv
    with open(os.path.join(DATA, "frame_pairs.csv")) as fh:
        pairs = list(csv.DictReader(fh))
    if args.pairs:
        pairs = pairs[:args.pairs]

    os.makedirs(OUT, exist_ok=True)
    stem = args.model.replace(":", "-").replace("/", "-")
    path = args.out or os.path.join(OUT, f"{stem}_frames.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["i"], r["j"], r["frame"], r["order"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"resuming: {len(done)} records already in {path}")

    jobs = [(p, f, o) for p in pairs for f in frames for o in ("ij", "ji")
            if (int(p["i"]), int(p["j"]), f, o) not in done]
    # frame_pairs.csv is sorted by block x similarity tercile, so run in a fixed shuffled
    # order: a partial/interrupted run is then still a balanced sample of the design.
    random.Random(20260728).shuffle(jobs)
    print(f"{args.model}: {len(jobs)} prompts to run "
          f"({len(pairs)} pairs x {len(frames)} frames x 2 orders)")

    t0 = time.time()
    nfail = 0
    with open(path, "a") as fh:
        for k, (p, frame, order) in enumerate(jobs, 1):
            a, b = (p["adj_i"], p["adj_j"]) if order == "ij" else (p["adj_j"], p["adj_i"])
            prompt = build_prompt(frame, a, b)
            resp, err = call_remote(args.model, prompt, key, timeout=args.timeout)
            if resp is None:
                nfail += 1
                rec = {"i": int(p["i"]), "j": int(p["j"]), "frame": frame, "order": order,
                       "subject": a, "referent": b, "error": err}
            else:
                dist, argmax, mass = distribution(resp)
                rec = {"i": int(p["i"]), "j": int(p["j"]), "frame": frame, "order": order,
                       "subject": a, "referent": b, "text": resp["text"],
                       "response": int(argmax) if argmax else None,
                       "ev": (sum(int(c) * dist[c] for c in OPTS) if dist else None),
                       "valid_mass": mass, "dist": [dist[c] for c in OPTS] if dist else None}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 50 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)}  {el:.0f}s  ({el/k:.2f}s/prompt, "
                      f"eta {(len(jobs)-k)*el/k/60:.1f} min, {nfail} failed)", flush=True)
    print(f"done: {path}  ({nfail} failures)")


if __name__ == "__main__":
    main()
