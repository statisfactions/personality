#!/usr/bin/env python3
"""Mixture model analysis using denoised (mean-across-variants) item EVs.

Reads variant results files and computes:
1. Denoised EVs per item (mean across prompt variants)
2. Mixture model decomposition using the most centrist model as assistant proxy
3. Variance decomposition: total → shared + unique → reliable unique + noise
4. Inter-model correlations on denoised values

Usage:
    python scripts/analyze_denoised.py
"""

import json
import math
import sys

ADMIN_SESSION = "admin_sessions/prod_run_01_external_rating.json"

VARIANT_FILES = {
    "gemma3":    "results/gemma3_4b_variants.json",
    "llama3.2":  "results/llama3.2_3b_variants.json",
    "phi4-mini": "results/phi4-mini_variants.json",
    "qwen3":     "results/qwen3_8b_variants.json",
}


def load_denoised(filepath):
    """Load variant results and return denoised (mean-across-variants) EVs per item."""
    with open(filepath) as f:
        data = json.load(f)

    variant_evs = data.get("variant_evs", {})
    if not variant_evs:
        # Fall back to single-variant EVs
        denoised = {}
        for item_id, result in data["item_results"].items():
            ev = result.get("expected_value")
            if ev is not None:
                denoised[item_id] = ev
        return denoised, data

    denoised = {}
    for item_id, evs in variant_evs.items():
        valid = [x for x in evs if x is not None]
        if valid:
            denoised[item_id] = sum(valid) / len(valid)
    return denoised, data


def main():
    # Load all models
    models = {}
    icc_data = {}
    for label, filepath in VARIANT_FILES.items():
        try:
            denoised, raw = load_denoised(filepath)
            models[label] = denoised
            icc_data[label] = raw
        except FileNotFoundError:
            print(f"  Skipping {label}: {filepath} not found")

    if len(models) < 2:
        print("Need at least 2 models to compare.")
        sys.exit(1)

    # Common items
    all_items = sorted(set.intersection(*[set(m.keys()) for m in models.values()]))
    model_names = list(models.keys())
    n_items = len(all_items)
    print(f"Models: {model_names}")
    print(f"Common items: {n_items}")

    # Load scale definitions
    with open(ADMIN_SESSION) as f:
        session = json.load(f)
    scales = session["measures"]["IPIP300"]["scales"]
    trait_abbrev = {"IPIP300-NEU": "N", "IPIP300-EXT": "E", "IPIP300-OPE": "O",
                    "IPIP300-AGR": "A", "IPIP300-CON": "C"}

    # Build EV matrix
    evs = {}
    for mname in model_names:
        evs[mname] = [models[mname].get(item_id, 3.0) for item_id in all_items]

    # Report denoised scale scores
    print("\n=== Denoised Big Five Scale Scores ===")
    print(f"{'Scale':<30s}", "  ".join(f"{m:>9s}" for m in model_names))
    print("-" * (30 + 11 * len(model_names)))
    for scale_id, scale_def in scales.items():
        item_ids = scale_def["item_ids"]
        rev_ids = set(scale_def["reverse_keyed_item_ids"])
        row = [scale_def["user_readable_name"]]
        for mname in model_names:
            vals = []
            for item_id in item_ids:
                if item_id in models[mname]:
                    ev = models[mname][item_id]
                    if item_id in rev_ids:
                        ev = 6.0 - ev
                    vals.append(ev)
            row.append(sum(vals) / len(vals) if vals else 3.0)
        print(f"{row[0]:<30s}", "  ".join(f"{v:9.3f}" for v in row[1:]))

    # Inter-model correlations (denoised)
    print("\n=== Inter-Model Correlations (Denoised) ===")
    print(f"{'':>12s}", "  ".join(f"{m:>9s}" for m in model_names))
    for mi, m1 in enumerate(model_names):
        row = []
        for mj, m2 in enumerate(model_names):
            ev1 = evs[m1]
            ev2 = evs[m2]
            n = len(ev1)
            mean1 = sum(ev1) / n
            mean2 = sum(ev2) / n
            cov = sum((a - mean1) * (b - mean2) for a, b in zip(ev1, ev2)) / (n - 1)
            var1 = sum((a - mean1) ** 2 for a in ev1) / (n - 1)
            var2 = sum((b - mean2) ** 2 for b in ev2) / (n - 1)
            r = cov / (var1 ** 0.5 * var2 ** 0.5) if var1 > 0 and var2 > 0 else 0
            row.append(r)
        print(f"{m1:>12s}", "  ".join(f"{r:9.3f}" for r in row))

    # Find the most centrist model (smallest total deviation from 3.0)
    centrism = {}
    for mname in model_names:
        centrism[mname] = sum((v - 3.0) ** 2 for v in evs[mname]) / n_items
    most_centrist = min(centrism, key=centrism.get)
    print(f"\nMost centrist model (assistant proxy): {most_centrist} "
          f"(mean sq dev from 3.0: {centrism[most_centrist]:.4f})")

    # Mixture model: subtract assistant proxy
    p_assistant = evs[most_centrist]

    print("\n=== Mixture Model (Denoised, Assistant Proxy = {}) ===".format(most_centrist))
    for mname in model_names:
        if mname == most_centrist:
            continue
        dev = [a - b for a, b in zip(evs[mname], p_assistant)]
        mean_dev = sum(dev) / len(dev)
        std_dev = (sum((d - mean_dev) ** 2 for d in dev) / (len(dev) - 1)) ** 0.5

        # Correlation with assistant
        n = len(dev)
        ev_m = evs[mname]
        mean_m = sum(ev_m) / n
        mean_a = sum(p_assistant) / n
        cov = sum((a - mean_m) * (b - mean_a) for a, b in zip(ev_m, p_assistant)) / (n - 1)
        var_m = sum((a - mean_m) ** 2 for a in ev_m) / (n - 1)
        var_a = sum((b - mean_a) ** 2 for b in p_assistant) / (n - 1)
        r = cov / (var_m ** 0.5 * var_a ** 0.5) if var_m > 0 and var_a > 0 else 0

        print(f"\n{mname}:")
        print(f"  Mean deviation from assistant: {mean_dev:+.3f}")
        print(f"  Std deviation: {std_dev:.3f}")
        print(f"  Correlation with assistant: {r:.3f}")

        # Per-scale residuals
        print(f"  Scale residuals (denoised):")
        for scale_id, scale_def in scales.items():
            item_ids = scale_def["item_ids"]
            rev_ids = set(scale_def["reverse_keyed_item_ids"])
            residuals = []
            for item_id in item_ids:
                if item_id in all_items:
                    idx = all_items.index(item_id)
                    r_val = dev[idx]
                    if item_id in rev_ids:
                        r_val = -r_val
                    residuals.append(r_val)
            if residuals:
                mean_r = sum(residuals) / len(residuals)
                print(f"    {scale_def['user_readable_name']:35s}: {mean_r:+.3f}")

    # Variance decomposition with reliability
    print("\n=== Variance Decomposition (Denoised) ===")
    print(f"  {'model':>12s} {'total_var':>10s} {'shared':>10s} {'unique':>10s} {'ICC':>6s} "
          f"{'reliable':>10s} {'noise':>10s}")

    for mname in model_names:
        if mname == most_centrist:
            continue

        ev_m = evs[mname]
        total_var = sum((v - sum(ev_m) / len(ev_m)) ** 2 for v in ev_m) / (len(ev_m) - 1)

        # Regress out assistant
        mean_m = sum(ev_m) / n_items
        mean_a = sum(p_assistant) / n_items
        cov = sum((a - mean_m) * (b - mean_a)
                   for a, b in zip(ev_m, p_assistant)) / (n_items - 1)
        var_a = sum((b - mean_a) ** 2 for b in p_assistant) / (n_items - 1)
        beta = cov / var_a if var_a > 0 else 0
        predicted = [beta * a + (mean_m - beta * mean_a) for a in p_assistant]
        residual = [m - p for m, p in zip(ev_m, predicted)]
        residual_var = sum(r ** 2 for r in residual) / (n_items - 1)
        shared_var = total_var - residual_var

        # Get ICC from the raw data
        raw = icc_data.get(mname, {})
        # Recompute ICC from variant_evs
        variant_evs_data = raw.get("variant_evs", {})
        if variant_evs_data:
            all_valid = []
            for item_id in all_items:
                vevs = variant_evs_data.get(item_id, [])
                valid = [x for x in vevs if x is not None]
                if len(valid) > 1:
                    all_valid.append(valid)
            if all_valid:
                k = len(all_valid[0])
                nn = len(all_valid)
                gm = sum(sum(r) for r in all_valid) / (nn * k)
                im = [sum(r) / k for r in all_valid]
                ssb = k * sum((m - gm) ** 2 for m in im)
                msb = ssb / (nn - 1)
                ssw = sum(sum((x - im[i]) ** 2 for x in r) for i, r in enumerate(all_valid))
                msw = ssw / (nn * (k - 1))
                icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else 0
            else:
                icc = 1.0
        else:
            icc = 1.0

        reliable_unique = residual_var * icc
        noise = residual_var * (1 - icc)

        print(f"  {mname:>12s} {total_var:10.3f} {shared_var:10.3f} {residual_var:10.3f} "
              f"{icc:6.2f} {reliable_unique:10.3f} {noise:10.3f}")

    # Top disagreement items (denoised)
    print("\n=== Top 15 Items by Cross-Model Variance (Denoised) ===")
    item_vars = []
    for ii, item_id in enumerate(all_items):
        vals = [evs[m][ii] for m in model_names]
        mean_v = sum(vals) / len(vals)
        var_v = sum((v - mean_v) ** 2 for v in vals) / (len(vals) - 1)
        item_vars.append((item_id, var_v, vals))

    item_vars.sort(key=lambda x: x[1], reverse=True)

    # Get item texts
    with open(ADMIN_SESSION) as f:
        sess = json.load(f)
    ipip_items = sess["measures"]["IPIP300"]["items"]

    print(f"  {'item':>10s} {'text':50s} {'var':>6s}  " +
          "  ".join(f"{m:>7s}" for m in model_names))
    print("-" * (70 + 9 * len(model_names)))
    for item_id, var_v, vals in item_vars[:15]:
        text = ipip_items.get(item_id, "?")[:50]
        val_str = "  ".join(f"{v:7.2f}" for v in vals)
        print(f"  {item_id:>10s} {text:50s} {var_v:6.3f}  {val_str}")


if __name__ == "__main__":
    main()
