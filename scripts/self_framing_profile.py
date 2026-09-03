"""SELF framing profiles: cohort structure + per-model (2026-09-02).

What the 08-22 sensitivity pass didn't cover: the framings themselves.
Elevation/entropy by framing per model, per-model 6x6 framing agreement
(modal outlier framing, direction of deviation), full 3-way variance
decomposition (model x framing x adjective), cohort-consistent
framing-sensitive adjectives.

First pkit-native analysis script (refactor-core exemplar).

Usage: PYTHONPATH=. .venv/bin/python scripts/self_framing_profile.py
Out:   results/adjectives/self_framing_profile.json
"""
import json

import numpy as np

import pkit

FR = pkit.load.FRAMINGS


def main():
    labels = pkit.load.adjectives()
    paths = pkit.load.self_paths("population")
    models, EV, ENT = [], [], []
    for repo, p in sorted(paths.items()):
        d = json.load(open(p))["results"]
        try:
            ev = [[d[f][a]["ev"] for a in labels] for f in FR]
            en = [[d[f][a].get("entropy", np.nan) for a in labels] for f in FR]
        except (KeyError, TypeError):
            print(f"  [skip] {repo}")
            continue
        models.append(repo.split("/")[-1])
        EV.append(ev)
        ENT.append(en)
    EV = np.array(EV)      # (m, f, a)
    ENT = np.array(ENT)
    m, f, a = EV.shape
    print(f"tensor: {m} models x {f} framings x {a} adjectives")

    # --- elevation / entropy by framing ---------------------------------
    elev = EV.mean(2)                       # (m, f)
    ent = np.nanmean(ENT, axis=2)
    cohort_elev = elev.mean(0)
    cohort_ent = ent.mean(0)

    # --- per-model framing agreement ------------------------------------
    agree = np.zeros((m, f, f))
    for i in range(m):
        agree[i] = np.corrcoef(EV[i])
    row_mean = (agree.sum(2) - 1) / (f - 1)          # mean r with others
    outlier_idx = row_mean.argmin(1)
    stability = row_mean.mean(1)

    # --- 3-way variance decomposition (one obs per cell) ----------------
    g = EV.mean()
    Em, Ef, Ea = EV.mean((1, 2)), EV.mean((0, 2)), EV.mean((0, 1))
    Emf, Ema, Efa = EV.mean(2), EV.mean(1), EV.mean(0)
    am, af, aa = Em - g, Ef - g, Ea - g
    i_mf = Emf - Em[:, None] - Ef[None, :] + g
    i_ma = Ema - Em[:, None] - Ea[None, :] + g
    i_fa = Efa - Ef[:, None] - Ea[None, :] + g
    resid = (EV - g - am[:, None, None] - af[None, :, None] - aa[None, None, :]
             - i_mf[:, :, None] - i_ma[:, None, :] - i_fa[None, :, :])
    ss = {"model": f * a * (am ** 2).sum(),
          "framing": m * a * (af ** 2).sum(),
          "adjective": m * f * (aa ** 2).sum(),
          "model_x_framing": a * (i_mf ** 2).sum(),
          "model_x_adjective": f * (i_ma ** 2).sum(),
          "framing_x_adjective": m * (i_fa ** 2).sum(),
          "threeway_resid": (resid ** 2).sum()}
    tot = sum(ss.values())
    shares = {k: v / tot for k, v in ss.items()}

    # --- cohort-consistent framing-sensitive adjectives -----------------
    # per-adjective framing effect after removing each model's framing
    # elevation: cohort mean of (EV[m,f,a] - elev[m,f]), var over f
    C = EV - elev[:, :, None]
    fa_profile = C.mean(0)                  # (f, a) cohort framing shape
    swing = fa_profile.var(0)
    top = np.argsort(-swing)[:25]
    movers = [{"adjective": labels[i], "swing_sd": float(np.sqrt(swing[i])),
               "by_framing": {FR[j]: round(float(fa_profile[j, i]), 2)
                              for j in range(f)}} for i in top]

    # --- correlates ------------------------------------------------------
    r_stab_ent = float(np.corrcoef(stability, np.nanmean(ent, 1))[0, 1])
    gen_map = {}
    for e in pkit.roster.manifest():
        gen_map[e["repo"].split("/")[-1]] = e.get("generation")
    gens = np.array([gen_map.get(mm, np.nan) for mm in models], float)
    ok = ~np.isnan(gens)
    r_stab_gen = float(np.corrcoef(stability[ok], gens[ok])[0, 1])

    out = {
        "models": models, "framings": FR, "n_adjectives": a,
        "cohort_elevation": dict(zip(FR, cohort_elev.round(3))),
        "cohort_entropy": dict(zip(FR, cohort_ent.round(3))),
        "cohort_agreement": np.mean(agree, 0).round(3).tolist(),
        "variance_shares": {k: round(v, 4) for k, v in shares.items()},
        "per_model": {models[i]: {
            "elevation": dict(zip(FR, elev[i].round(2))),
            "entropy": dict(zip(FR, ent[i].round(2))),
            "framing_mean_r": dict(zip(FR, row_mean[i].round(3))),
            "outlier_framing": FR[outlier_idx[i]],
            "stability": round(float(stability[i]), 3),
        } for i in range(m)},
        "outlier_framing_counts": {fr: int((outlier_idx == j).sum())
                                   for j, fr in enumerate(FR)},
        "top_movers": movers,
        "r_stability_entropy": round(r_stab_ent, 3),
        "r_stability_generation": round(r_stab_gen, 3),
    }
    op = pkit.paths.ADJ / "self_framing_profile.json"
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op)

    print("\ncohort elevation:", {k: round(v, 2) for k, v in out["cohort_elevation"].items()})
    print("cohort entropy:  ", {k: round(v, 2) for k, v in out["cohort_entropy"].items()})
    print("variance shares: ", out["variance_shares"])
    print("outlier framing counts:", out["outlier_framing_counts"])
    print(f"r(stability, entropy) = {r_stab_ent:+.2f}   r(stability, generation) = {r_stab_gen:+.2f}")
    print("top movers:", [mv["adjective"] for mv in movers[:20]])


if __name__ == "__main__":
    main()
