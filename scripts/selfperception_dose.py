"""Self-perception dose-response, stage 1 (design: rgb_reports/design_selfperception.md).

Dose a model's context with K of its OWN W17 persona rollouts (arm A: no
instruction anywhere; arm B: persona sys prompt visible), then read a
targeted self-report (COLD: verbatim SELF direct framing; DIRECTED:
"looking at your responses so far...") as digit distributions. Capture
mid-layer activations for BE-vector extraction and per-turn logprob
believability along the way.

Notes
- COLD wording is identical at every K; all inference lives in the delta.
- DIRECTED only exists for K>0 (there is no "so far" at K=0).
- K=0 arm A is adjective-independent (cached per item word); K=0 arm B is
  the instruction-only cell (per adjective, no seeds).
- SELF (*_self_full.json) was bare-text; these contexts are chat-template.
  Our K=0 is the in-design baseline; the SELF comparison crosses a format
  shift and is a sanity check, not an identity.

Usage:
  PYTHONPATH=scripts python scripts/selfperception_dose.py --model Llama8 --smoke
  PYTHONPATH=scripts python scripts/selfperception_dose.py --model Llama8
Output: results/selfperception/<model>_stage1.json (+ _part.jsonl checkpoint,
        acts/<model>/*.npz activation sidecars)
"""
import argparse
import copy
import json
import os
import random
import re

import numpy as np
import torch

import hf_logprobs as hf

OUT = "results/selfperception"
KS = [0, 1, 2, 4, 8]
N_ADJ = 20
N_SEEDS = 3
N_MATES = 4
N_ANTI = 4
DIGITS = list("1234567")

SCALE_HEADER = (
    "Indicate how much you agree with the following statement about "
    "yourself on a 7-point scale:\n"
    "1: Strongly disagree\n2: Disagree\n3: Slightly disagree\n"
    "4: Neither agree nor disagree\n5: Slightly agree\n6: Agree\n"
    "7: Strongly agree\nRespond with only a number.\n\n"
    'Statement: "I am {w}."')
DIRECTED_PREFIX = ("Looking at your responses so far in this "
                   "conversation, indicate")
PREFILL = "Rating: "
PROBE = ("Look back over your earlier replies in this conversation. "
         "Do you notice anything unusual about them?")
# explicit verbal self-attribution in dose turns ("As a renowned X, ...") —
# leak-vs-conduct covariate; crude regex now, judge-scored in stage 2
SELFCLAIM = re.compile(r"\b(as an?|i am an?|i'm an?|as someone who)\b",
                       re.IGNORECASE)


def pick_adjectives(model_name, rolls, n_adj, rng):
    """Stratified sample: enactability tercile x baseline direct-EV tercile."""
    ena = json.load(open(
        f"results/adjectives/enactability/{model_name}_enactability.json"))["scores"]
    self_d = json.load(open(
        f"results/adjectives/selfreport/{model_name}_self_full.json"))
    direct = self_d["results"]["direct"]
    elig = []
    for adj, rs in rolls.items():
        if adj == "__default__" or adj not in ena or adj not in direct:
            continue
        clean = [r for r in rs if not r.get("leak")]
        if len({r["question"] for r in clean}) < 8:
            continue
        elig.append(dict(adj=adj, enact=ena[adj]["enactability"],
                         base_ev=direct[adj]["ev"]))
    def terc(vals, x):
        q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
        return 0 if x <= q1 else (1 if x <= q2 else 2)
    evs = [e["enact"] for e in elig]
    bvs = [e["base_ev"] for e in elig]
    cells = {}
    for e in elig:
        cells.setdefault((terc(evs, e["enact"]), terc(bvs, e["base_ev"])),
                         []).append(e)
    for c in cells.values():
        rng.shuffle(c)
    picked, i = [], 0
    order = sorted(cells)
    while len(picked) < min(n_adj, len(elig)):
        cell = cells[order[i % len(order)]]
        if cell:
            picked.append(cell.pop())
        i += 1
        if all(not c for c in cells.values()):
            break
    return picked


def item_sets(targets, sims_path="results/steer_map/facet_channel_sims.npz",
              clusters_path="instruments/trait_clusters.json"):
    """target + cluster-mates (fallback: JUDGE-nearest) + JUDGE-most-negative.

    Row order of the sims npz = escs_525pda_corr_raw.json labels (the
    convention set by adjective_facet_cohort.py) — NOT sorted()."""
    z = np.load(sims_path, allow_pickle=True)
    J = z["JUDGE"]
    adjs = [l.lower() for l in json.load(
        open("results/adjectives/escs_525pda_corr_raw.json"))["labels"]]
    assert len(adjs) == J.shape[0], "sims/labels misalignment"
    idx = {a: i for i, a in enumerate(adjs)}
    # anti-markers from PC1-REMOVED JUDGE: raw most-negative is just the
    # desirability floor (evil/corrupt/... for every positive target)
    A = J.copy()
    np.fill_diagonal(A, 0.0)
    w, v = np.linalg.eigh(A)
    k1 = np.argmax(np.abs(w))
    J1 = A - w[k1] * np.outer(v[:, k1], v[:, k1])
    pool = json.load(open(clusters_path))["pool"]
    out = {}
    for t in targets:
        mates = []
        for c in pool:
            if t in c["members"]:
                mates = [m for m in c["members"] if m != t and m in idx]
                break
        row = J[idx[t]]
        if not mates:  # JUDGE-nearest fallback
            near = np.argsort(-row)
            mates = [adjs[i] for i in near if adjs[i] != t][:N_MATES]
            src = "judge_nn"
        else:
            mates = mates[:N_MATES]
            src = "cluster"
        far = np.argsort(J1[idx[t]])
        # blocklist: JUDGE-space outlier rows that surface for every target
        anti = [adjs[i] for i in far
                if adjs[i] != t and adjs[i] not in {"retarded"}][:N_ANTI]
        out[t] = dict(items=[t] + mates + anti, mates=mates, anti=anti,
                      mate_source=src)
    return out, idx


def digit_tokens(tok):
    dt = {}
    for d in DIGITS:
        dt[d] = [t[0] for s in (d, " " + d)
                 for t in [tok.encode(s, add_special_tokens=False)]
                 if len(t) == 1]
    return dt


def read_digits(logits, dt):
    p = logits.softmax(-1)
    mass = {d: sum(p[t].item() for t in ts) for d, ts in dt.items()}
    tot = sum(mass.values())
    dist = {d: m / tot for d, m in mass.items()} if tot > 0 else \
        {d: 1 / 7 for d in DIGITS}
    ev = sum(int(d) * q for d, q in dist.items())
    ent = -sum(q * np.log(q) for q in dist.values() if q > 0)
    return dict(ev=ev, entropy=float(ent), digit_mass=float(tot), dist=dist)


# identity anchors (8b): Qwen's template injects NAMED when no system
# message is given; an explicit empty system message suppresses it
ANCHOR_NAMED = {
    "Qwen7": "You are Qwen, created by Alibaba Cloud. "
             "You are a helpful assistant.",
    "Llama8": "You are Llama, created by Meta. You are a helpful assistant.",
}
ANCHOR_HELPFUL = "You are a helpful assistant."


BARE = False  # --bare: plain-text transcript, for base models (W15 §3)


def render(tok, msgs, add_gen):
    if not BARE:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=add_gen)
    s = ""
    for m in msgs:
        if m["role"] == "system":
            s += (m["content"] + "\n\n") if m["content"] else ""
        elif m["role"] == "user":
            s += f"User: {m['content']}\n\n"
        else:
            s += f"Assistant: {m['content']}\n\n"
    return s + ("Assistant: " if add_gen else "")


def build_msgs(sys_prompt, turns):
    """sys_prompt None -> no system message (template default applies);
    "" -> explicit EMPTY system message (suppresses Qwen's default)."""
    msgs = ([] if sys_prompt is None
            else [{"role": "system", "content": sys_prompt}])
    for q, t in turns:
        msgs += [{"role": "user", "content": q},
                 {"role": "assistant", "content": t}]
    return msgs


def encode_prefix(model, tok, device, msgs, mid_layer):
    """Forward the dose context once. Returns cache handle + believability
    (mean per-token logprob of each assistant turn) + activation means."""
    if not msgs:
        return None
    pre_s = render(tok, msgs, False)
    pre = tok(pre_s, add_special_tokens=False).input_ids
    # assistant-turn token spans via incremental template boundaries
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        a = len(tok(render(tok, msgs[:i], True),
                    add_special_tokens=False).input_ids)
        b = len(tok(render(tok, msgs[:i + 1], False),
                    add_special_tokens=False).input_ids)
        spans.append((min(a, len(pre)), min(b, len(pre))))
    pre_t = torch.tensor([pre], device=device)
    outp = model(pre_t, use_cache=True, output_hidden_states=True)
    logp = outp.logits[0].float().log_softmax(-1)
    tlp = logp[:-1].gather(-1, pre_t[0, 1:, None]).squeeze(-1)
    bel = [float(tlp[max(a - 1, 0):b - 1].mean()) for a, b in spans
           if b - 1 > max(a - 1, 0)]
    hid = outp.hidden_states[mid_layer][0].float()
    turn_means = [hid[a:b].mean(0) for a, b in spans if b > a]
    acts = dict(
        turn_mean=(torch.stack(turn_means).mean(0) if turn_means
                   else hid.mean(0)).half().cpu().numpy(),
        last_hidden=hid[-1].half().cpu().numpy())
    return dict(ids=pre, cache_src=outp, msgs=msgs, believability=bel,
                acts=acts)


def rate(model, tok, device, prefix, item_word, rung, dt):
    header = SCALE_HEADER.format(w=item_word)
    if rung == "directed":
        header = header.replace("Indicate", DIRECTED_PREFIX, 1)
    msgs = (prefix["msgs"] if prefix else []) + \
        [{"role": "user", "content": header}]
    full_s = render(tok, msgs, True) + PREFILL
    full = tok(full_s, add_special_tokens=False).input_ids
    if prefix and full[:len(prefix["ids"])] == prefix["ids"]:
        cache = copy.deepcopy(prefix["cache_src"].past_key_values)
        cont = torch.tensor([full[len(prefix["ids"]):]], device=device)
        logits = model(cont, past_key_values=cache).logits[0, -1].float()
    else:
        logits = model(torch.tensor([full], device=device)
                       ).logits[0, -1].float()
    return read_digits(logits, dt)


def probe_check(model, tok, device, msgs):
    full_s = render(tok, msgs + [{"role": "user", "content": PROBE}], True)
    ids = tok(full_s, add_special_tokens=False, return_tensors="pt"
              ).input_ids.to(device)
    gen = model.generate(ids, max_new_tokens=120, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Llama8")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-adj", type=int, default=N_ADJ)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--anchor", default="default",
                    choices=["default", "empty", "helpful", "named"],
                    help="base system message for arm A (8b anchor 2x3)")
    ap.add_argument("--arms", default="AB")
    ap.add_argument("--bare", action="store_true",
                    help="plain-text transcript (base models; W15 §3)")
    ap.add_argument("--dose-from", default=None,
                    help="path to a texts json to dose with (ladder models "
                         "share one tuned-end rollout set)")
    ap.add_argument("--select-from", default=None,
                    help="reuse another model's adjective selection")
    ap.add_argument("--tag", default=None, help="output tag override")
    args = ap.parse_args()
    global BARE
    BARE = args.bare
    base_sys = {"default": None, "empty": "", "helpful": ANCHOR_HELPFUL,
                "named": ANCHOR_NAMED.get(args.model)}[args.anchor]
    tag = args.tag if args.tag is not None else (
        "" if args.anchor == "default" else f"_anchor-{args.anchor}")

    ks, n_adj, n_seeds = KS, args.n_adj, args.seeds
    if args.smoke:
        ks, n_adj, n_seeds = [0, 2, 8], 2, 1

    os.makedirs(OUT, exist_ok=True)
    actdir = f"{OUT}/acts/{args.model}{tag}"
    os.makedirs(actdir, exist_ok=True)

    rolls = json.load(open(args.dose_from if args.dose_from else
                           f"results/persona_vectors/{args.model}"
                           "_pda_texts.json"))
    try:
        meta = json.load(open(
            f"results/persona_vectors/{args.model}_pda_meta.json"))
        mid_layer = meta["mid_layer"]
    except FileNotFoundError:      # ladder models have no W17 extraction
        mid_layer = None

    rng = random.Random(0)
    if args.select_from:
        prev = json.load(open(
            f"{OUT}/{args.select_from}_selection.json"))
        picked = [p for p in prev["picked"] if p["adj"] in rolls]
    else:
        picked = pick_adjectives(args.model, rolls, n_adj, rng)
    items, _ = item_sets([p["adj"] for p in picked])
    if args.smoke:
        for t in items:
            items[t]["items"] = ([t] + items[t]["mates"][:2]
                                 + items[t]["anti"][:2])
    sel_path = f"{OUT}/{args.model}{tag}_selection.json"
    json.dump(dict(picked=picked, items=items, ks=ks, seeds=n_seeds,
                   smoke=args.smoke, anchor=args.anchor, arms=args.arms),
              open(sel_path, "w"), indent=1)
    print(f"{len(picked)} adjectives; wrote {sel_path}")

    part = f"{OUT}/{args.model}{tag}_part.jsonl"
    done = set()
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            done.add((r["adj"], r["K"], r["arm"], r["seed"]))
        print(f"resuming: {len(done)} contexts done")

    model, tok, device = hf.load_model(args.model, dtype=torch.bfloat16)
    model.eval()
    if mid_layer is None:
        mid_layer = model.config.num_hidden_layers // 2
    dt = digit_tokens(tok)

    fout = open(part, "a")
    cold0 = {}  # K=0 arm A cache: item word -> reading (context-free)

    def emit(rec):
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    with torch.no_grad():
        # --- K=0 arm A: adjective-independent, one pass per unique word
        if ("__k0__", 0, "A", 0) not in done:
            words = sorted({w for t in items.values() for w in t["items"]})
            k0_prefix = encode_prefix(model, tok, device,
                                      build_msgs(base_sys, []), mid_layer)
            for w in words:
                cold0[w] = rate(model, tok, device, k0_prefix, w, "cold", dt)
            emit(dict(adj="__k0__", K=0, arm="A", seed=0,
                      readings={w: {"cold": cold0[w]} for w in words}))
            print(f"K=0 arm A: {len(words)} words")
        else:
            for line in open(part):
                r = json.loads(line)
                if r["adj"] == "__k0__":
                    cold0 = {w: v["cold"] for w, v in r["readings"].items()}

        total = 0
        for p in picked:
            adj = p["adj"]
            clean = [r for r in rolls[adj] if not r.get("leak")]
            sys_prompt = clean[0]["sys"]
            iset = items[adj]["items"]

            # K=0 arm B (instruction-only, no seeds)
            if (adj, 0, "B", 0) not in done and 0 in ks and "B" in args.arms:
                prefix = encode_prefix(model, tok, device,
                                       build_msgs(sys_prompt, []), mid_layer)
                rd = {w: {"cold": rate(model, tok, device, prefix, w,
                                       "cold", dt)} for w in iset}
                emit(dict(adj=adj, K=0, arm="B", seed=0, readings=rd,
                          believability=[]))

            for seed in range(n_seeds):
                srng = random.Random(hash((adj, seed)) & 0xFFFFFFFF)
                qs = sorted({r["question"] for r in clean})
                srng.shuffle(qs)
                by_q = {}
                for r in clean:
                    by_q.setdefault(r["question"], []).append(r)
                pool8 = [srng.choice(by_q[q]) for q in qs[:max(k for k in ks
                                                              if k > 0)]]
                for K in [k for k in ks if k > 0]:
                    turns = [(r["question"], r["text"]) for r in pool8[:K]]
                    for arm in args.arms:
                        if (adj, K, arm, seed) in done:
                            continue
                        msgs = build_msgs(sys_prompt if arm == "B"
                                          else base_sys, turns)
                        prefix = encode_prefix(model, tok, device, msgs,
                                               mid_layer)
                        assert prefix is not None
                        rd = {}
                        for w in iset:
                            rd[w] = {"cold": rate(model, tok, device,
                                                  prefix, w, "cold", dt),
                                     "directed": rate(model, tok, device,
                                                      prefix, w, "directed",
                                                      dt)}
                        rec = dict(adj=adj, K=K, arm=arm, seed=seed,
                                   readings=rd,
                                   believability=prefix["believability"],
                                   selfclaim=[len(SELFCLAIM.findall(t))
                                              for _, t in turns])
                        if arm == "A" and K == max(ks) and seed == 0:
                            rec["probe"] = probe_check(model, tok, device,
                                                       msgs)
                        emit(rec)
                        np.savez_compressed(
                            f"{actdir}/{adj}_K{K}_{arm}_s{seed}.npz",
                            **prefix["acts"])
                        total += 1
                        if total % 10 == 0:
                            print(f"  {total} contexts ({adj} K={K} {arm} "
                                  f"s{seed})", flush=True)
    fout.close()

    # assemble
    rows = [json.loads(line) for line in open(part)]
    outp = f"{OUT}/{args.model}{tag}_stage1.json"
    json.dump(dict(model=args.model, mid_layer=mid_layer, ks=ks,
                   anchor=args.anchor, selection=sel_path,
                   n_contexts=len(rows), rows=rows), open(outp, "w"))
    print(f"wrote {outp} ({len(rows)} context records)")


if __name__ == "__main__":
    main()
