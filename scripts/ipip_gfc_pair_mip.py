#!/usr/bin/env python3
"""W11 Phase C: solve Okada Appendix C's two-stage MIP on IPIP-NEO-300.

Reformulation: binary-search on the minimax m* parameter rather than
embedding m as a continuous variable. Each feasibility check is a pure
integer program (no LP-relaxation gap), which CBC handles 1-2 orders
of magnitude faster than the literal Okada formulation on a 33k-pair
candidate set.

Algorithm:
  1. Sort candidates by Δsd.
  2. Binary search m in [0, max Δsd]:
       - Restrict admissible pairs to {Δsd ≤ m_test}
       - Solve pure-feasibility integer program: ∃ valid P-pair
         selection respecting Okada's balance constraints?
       - If feasible: tighten upper bound (hi = m_test)
       - If infeasible: tighten lower bound (lo = m_test)
  3. m* = hi at convergence.
  4. Stage 2: minimize Σ Δsd² · x_k over pairs with Δsd ≤ m* + eps,
     same constraints. This is the original Okada Stage 2.

Reads:
  results/desirability/cohort_phase_b_ipip300.json
  instruments/ipip300_annotations.json (deny-list)

Writes:
  instruments/ipip_neo_gfc_P<P>.json — selected pairs, formatted to
  match the Okada GFC-30 instrument shape.

Usage:
    .venv/bin/python scripts/ipip_gfc_pair_mip.py --P 60
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pulp


PHASE_B = "results/desirability/cohort_phase_b_ipip300.json"
ANNOTATIONS = "instruments/ipip300_annotations.json"
TRAITS = ["A", "C", "E", "N", "O"]


def load_inputs(P):
    data = json.load(open(PHASE_B))
    items = data["items"]
    cohort_mean = data["cohort_mean_desirability"]
    deny = set(json.load(open(ANNOTATIONS)).get("deny", {}).keys()) if Path(ANNOTATIONS).exists() else set()

    pool = []
    for it in items:
        if it["id"] in deny: continue
        if it["id"] not in cohort_mean: continue
        pool.append({
            "id": it["id"],
            "text": it["text"],
            "depersonalized": it["depersonalized"],
            "trait": it["trait"],
            "keying": it["keying"],
            "sd": float(cohort_mean[it["id"]]),
        })
    print(f"Item pool: {len(pool)} items (excluded {len(deny)} deny-listed)")
    return pool


def build_candidates(pool):
    """Cross-domain pairs only, sorted by Δsd ascending."""
    pairs = []
    for i, j in combinations(range(len(pool)), 2):
        if pool[i]["trait"] == pool[j]["trait"]:
            continue
        delta = abs(pool[i]["sd"] - pool[j]["sd"])
        mixed = 1 if pool[i]["keying"] != pool[j]["keying"] else 0
        pairs.append((i, j, delta, mixed))
    pairs.sort(key=lambda t: t[2])
    print(f"Cross-domain candidates: {len(pairs)} (sorted by Δsd)")
    return pairs


def add_okada_constraints(prob, x, pool, pairs, admissible_idx, P):
    """Add Okada Appendix C constraints to a PuLP problem.

    x: list of binary vars, one per pair index in `pairs`.
    admissible_idx: set of indices k in `pairs` that are allowed (others fixed to 0).
    """
    # Fix non-admissible pairs to 0
    for k in range(len(pairs)):
        if k not in admissible_idx:
            prob += x[k] == 0, f"fix_{k}"

    # Total
    prob += pulp.lpSum(x[k] for k in admissible_idx) == P, "total_pairs"

    # Each item used at most once
    item_to_pairs = {idx: [] for idx in range(len(pool))}
    for k in admissible_idx:
        i, j, _, _ = pairs[k]
        item_to_pairs[i].append(k)
        item_to_pairs[j].append(k)
    for idx, ks in item_to_pairs.items():
        if ks:
            prob += pulp.lpSum(x[k] for k in ks) <= 1, f"item_{idx}"

    # Domain coverage: each trait appears exactly 2P/5 times
    Nt_target = 2 * P // 5
    for t in TRAITS:
        coeffs = []
        for k in admissible_idx:
            i, j, _, _ = pairs[k]
            c = (1 if pool[i]["trait"] == t else 0) + (1 if pool[j]["trait"] == t else 0)
            if c > 0: coeffs.append(c * x[k])
        if coeffs:
            prob += pulp.lpSum(coeffs) == Nt_target, f"domain_{t}"

    # Trait-pair coverage: each unordered pair appears exactly P/10 times
    Ntt_target = P // 10
    for t1, t2 in combinations(TRAITS, 2):
        ks_in = [k for k in admissible_idx
                 if {pool[pairs[k][0]]["trait"], pool[pairs[k][1]]["trait"]} == {t1, t2}]
        if ks_in:
            prob += pulp.lpSum(x[k] for k in ks_in) == Ntt_target, f"tt_{t1}{t2}"
        elif Ntt_target > 0:
            # No admissible pairs for this trait-pair: infeasible
            prob += pulp.lpSum([]) == Ntt_target, f"tt_{t1}{t2}_infeas"

    # Mixed-keying fraction in [0.4P, 0.6P]
    mixed_sum = pulp.lpSum(pairs[k][3] * x[k] for k in admissible_idx)
    prob += mixed_sum >= 0.4 * P, "mixed_lo"
    prob += mixed_sum <= 0.6 * P, "mixed_hi"

    # Per-domain keying balance: 7*N+ >= 3*N- and vice versa
    for t in TRAITS:
        pos_terms = []
        neg_terms = []
        for k in admissible_idx:
            i, j, _, _ = pairs[k]
            for idx in (i, j):
                if pool[idx]["trait"] == t:
                    if pool[idx]["keying"] == "+":
                        pos_terms.append(x[k])
                    else:
                        neg_terms.append(x[k])
        prob += 7 * pulp.lpSum(pos_terms) >= 3 * pulp.lpSum(neg_terms), f"k_pos_{t}"
        prob += 7 * pulp.lpSum(neg_terms) >= 3 * pulp.lpSum(pos_terms), f"k_neg_{t}"


def check_feasibility(pool, pairs, m_test, P, time_limit=60):
    """Pure feasibility integer program at threshold m_test."""
    admissible = {k for k, (_, _, d, _) in enumerate(pairs) if d <= m_test}
    if len(admissible) < P:
        return False, None
    prob = pulp.LpProblem("feas", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{k}", cat=pulp.LpBinary) for k in range(len(pairs))]
    prob += 0  # constant objective
    add_okada_constraints(prob, x, pool, pairs, admissible, P)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = prob.solve(solver)
    feasible = (status == pulp.LpStatusOptimal)
    selection = [k for k in admissible if x[k].varValue and x[k].varValue > 0.5] if feasible else None
    return feasible, selection


def binary_search_m_star(pool, pairs, P, tol=0.005, time_limit=60):
    """Binary search for smallest m such that a P-pair selection is feasible."""
    print(f"\n=== Stage 1: binary search for m* (P={P}, tol={tol}) ===")
    # Lower bound: enough admissible pairs but probably not feasible
    # Upper bound: clearly enough flexibility
    deltas = [d for _, _, d, _ in pairs]
    lo = 0.0
    hi = max(deltas)

    # Tighten initial hi via an upper-bound check: at hi, must be feasible.
    # Tighten initial lo via lower-bound: at m_low we need at least P admissible pairs;
    # we also need enough +/- per trait, which is a quick necessary check.
    # Skip the necessary-condition check; let the search converge.

    # Cache previously-good selection so we can recover one
    best_selection = None
    best_m = hi

    iteration = 0
    while hi - lo > tol:
        iteration += 1
        mid = (lo + hi) / 2
        n_admiss = sum(1 for d in deltas if d <= mid)
        print(f"  iter {iteration}: testing m={mid:.4f}, admissible={n_admiss}", flush=True)
        t0 = time.time()
        feas, sel = check_feasibility(pool, pairs, mid, P, time_limit=time_limit)
        dt = time.time() - t0
        if feas:
            hi = mid
            best_selection = sel
            best_m = mid
            print(f"    FEASIBLE ({dt:.1f}s)")
        else:
            lo = mid
            print(f"    infeasible ({dt:.1f}s)")
    print(f"\n  m* ≈ {best_m:.4f}  (converged after {iteration} iterations)")
    return best_m, best_selection


def solve_stage2(pool, pairs, P, m_star, eps=1e-3, time_limit=300):
    """Minimize total squared mismatch s.t. all selected Δsd ≤ m_star + eps."""
    print(f"\n=== Stage 2: minimize total Σ Δ² (m ≤ {m_star + eps:.4f}) ===")
    admissible = {k for k, (_, _, d, _) in enumerate(pairs) if d <= m_star + eps}
    print(f"  admissible pairs: {len(admissible)}")
    prob = pulp.LpProblem("stage2", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{k}", cat=pulp.LpBinary) for k in range(len(pairs))]
    prob += pulp.lpSum(pairs[k][2] ** 2 * x[k] for k in admissible)
    add_okada_constraints(prob, x, pool, pairs, admissible, P)
    t0 = time.time()
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    print(f"  Stage 2 status: {pulp.LpStatus[status]}  ({time.time()-t0:.1f}s)")
    selection = [k for k in admissible if x[k].varValue and x[k].varValue > 0.5]
    total_sq = pulp.value(prob.objective) if prob.objective is not None else float("nan")
    return total_sq, selection


def emit_instrument(pool, pairs, selected, P, m_star, total_sq, out_path):
    out_pairs = []
    for block, k in enumerate(selected, 1):
        i, j, delta, mixed = pairs[k]
        a, b = pool[i], pool[j]
        out_pairs.append({
            "block": block,
            "left": {
                "text": a["depersonalized"],
                "original_first_person": a["text"],
                "ipip_id": a["id"],
                "trait": a["trait"], "keying": a["keying"],
                "sd": round(a["sd"], 4),
            },
            "right": {
                "text": b["depersonalized"],
                "original_first_person": b["text"],
                "ipip_id": b["id"],
                "trait": b["trait"], "keying": b["keying"],
                "sd": round(b["sd"], 4),
            },
            "delta_sd": round(delta, 4),
            "mixed_keying": bool(mixed),
        })

    # Diagnostic stats
    domain_counts = {t: 0 for t in TRAITS}
    pair_combo_counts = {}
    keying_per_domain = {t: {"+": 0, "-": 0} for t in TRAITS}
    mixed_total = 0
    deltas = []
    for p in out_pairs:
        for side in ("left", "right"):
            s = p[side]
            domain_counts[s["trait"]] += 1
            keying_per_domain[s["trait"]][s["keying"]] += 1
        combo = tuple(sorted([p["left"]["trait"], p["right"]["trait"]]))
        pair_combo_counts[combo] = pair_combo_counts.get(combo, 0) + 1
        if p["mixed_keying"]:
            mixed_total += 1
        deltas.append(p["delta_sd"])

    out = {
        "instrument": f"IPIP-NEO-GFC-{P}",
        "description": (
            "Desirability-matched graded forced-choice inventory built on "
            "IPIP-NEO-300 items, using cohort-mean (Gemma/Llama/Phi4/Qwen "
            "3B-12B) desirability ratings as input to Okada Appendix C's "
            "constraint set."
        ),
        "citation": "Okada et al. 2026 (arXiv 2602.17262) methodology; constructed 2026-05-13",
        "n_pairs": P,
        "n_items": 2 * P,
        "traits": TRAITS,
        "design_notes": {
            "max_delta_sd": round(m_star, 4),
            "mean_delta_sd": round(float(np.mean(deltas)), 4),
            "total_squared_mismatch": round(float(total_sq), 4),
            "domain_counts": domain_counts,
            "pair_combo_counts": {f"{a}{b}": v for (a, b), v in sorted(pair_combo_counts.items())},
            "keying_per_domain": keying_per_domain,
            "n_mixed_keying": mixed_total,
            "mixed_keying_fraction": round(mixed_total / P, 4),
        },
        "pairs": out_pairs,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")
    print(f"  max Δsd: {m_star:.4f}, mean Δsd: {np.mean(deltas):.4f}")
    print(f"  domain counts: {domain_counts}")
    print(f"  trait-pair coverage: {dict((f'{a}{b}', v) for (a,b),v in sorted(pair_combo_counts.items()))}")
    print(f"  mixed keying: {mixed_total}/{P} ({100*mixed_total/P:.0f}%)")
    print(f"  per-domain keying: {keying_per_domain}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--P", type=int, default=60)
    parser.add_argument("--output", default=None)
    parser.add_argument("--m-tol", type=float, default=0.005,
                        help="Binary-search convergence tolerance for m*.")
    parser.add_argument("--feas-time", type=int, default=60,
                        help="Time limit per feasibility check (seconds).")
    parser.add_argument("--stage2-time", type=int, default=300,
                        help="Time limit for Stage 2 squared-sum minimization.")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"instruments/ipip_neo_gfc_P{args.P}.json"

    pool = load_inputs(args.P)
    pairs = build_candidates(pool)
    m_star, _ = binary_search_m_star(pool, pairs, args.P, tol=args.m_tol, time_limit=args.feas_time)
    total_sq, selection = solve_stage2(pool, pairs, args.P, m_star, time_limit=args.stage2_time)
    emit_instrument(pool, pairs, selection, args.P, m_star, total_sq, args.output)


if __name__ == "__main__":
    main()
