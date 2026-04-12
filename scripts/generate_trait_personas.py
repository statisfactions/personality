#!/usr/bin/env python3
"""Generate synthetic personas with known Big Five trait profiles.

Follows the approach from Okada et al. (2026):
1. Sample trait vectors from N(0, Sigma_human) with empirical Big Five correlations
2. Map z-scores to stanine levels (1-9)
3. Convert to adjective-based persona descriptions using Goldberg markers
4. Output JSON with persona descriptions and ground-truth trait vectors

Adjective markers from Goldberg (1992) / Saucier (1994), as used in the
PsyBORGS framework (Serapio-Garcia et al., 2023).

Usage:
    python3 scripts/generate_trait_personas.py --n 50 --seed 42
    python3 scripts/generate_trait_personas.py --n 100 --seed 42 --output instruments/synthetic_personas.json
"""

import argparse
import json
import numpy as np

# --- Goldberg Big Five markers (high and low poles) ---
# From PsyBORGS admin sessions, originally from Goldberg (1992) 52 markers

MARKERS = {
    "E": {
        "high": ["extraverted", "energetic", "talkative", "bold", "active",
                 "assertive", "friendly", "adventurous and daring", "joyful"],
        "low":  ["introverted", "unenergetic", "silent", "timid", "inactive",
                 "unassertive", "unfriendly", "unadventurous", "gloomy"],
    },
    "A": {
        "high": ["kind", "cooperative", "unselfish", "agreeable", "trustful",
                 "generous", "moral", "honest", "altruistic", "humble",
                 "sympathetic"],
        "low":  ["unkind", "uncooperative", "selfish", "disagreeable",
                 "distrustful", "stingy", "immoral", "dishonest",
                 "unaltruistic", "self-important", "unsympathetic"],
    },
    "C": {
        "high": ["organized", "responsible", "conscientious", "practical",
                 "thorough", "hardworking", "thrifty", "self-efficacious",
                 "orderly", "self-disciplined"],
        "low":  ["disorganized", "irresponsible", "negligent", "impractical",
                 "careless", "lazy", "extravagant", "unsure", "messy",
                 "undisciplined"],
    },
    "N": {
        "high": ["angry", "tense", "nervous", "discontented", "anxious",
                 "irritable", "depressed", "self-conscious", "impulsive",
                 "emotionally unstable"],
        "low":  ["calm", "relaxed", "at ease", "contented", "easygoing",
                 "patient", "happy", "unselfconscious", "level-headed",
                 "emotionally stable"],
    },
    "O": {
        "high": ["intelligent", "analytical", "reflective", "curious",
                 "imaginative", "creative", "sophisticated",
                 "artistically appreciative", "aesthetic",
                 "emotionally aware", "spontaneous",
                 "socially progressive"],
        "low":  ["unintelligent", "unanalytical", "unreflective",
                 "uninquisitive", "unimaginative", "uncreative",
                 "unsophisticated", "artistically unappreciative",
                 "unaesthetic", "emotionally closed", "predictable",
                 "socially conservative"],
    },
}

# Trait order for the correlation matrix
TRAITS = ["A", "C", "E", "N", "O"]

# --- Meta-analytic Big Five correlation matrix ---
# From van der Linden et al. (2010), as used by Okada et al. (2026) Appendix E
# Order: A, C, E, N, O
HUMAN_BIG5_CORR = np.array([
    [ 1.00,  0.43,  0.26, -0.36,  0.21],  # A
    [ 0.43,  1.00,  0.29, -0.43,  0.20],  # C
    [ 0.26,  0.29,  1.00, -0.36,  0.43],  # E
    [-0.36, -0.43, -0.36,  1.00, -0.17],  # N
    [ 0.21,  0.20,  0.43, -0.17,  1.00],  # O
])

# Intensity modifiers mapped to stanine levels (1-9)
# Following PsyBORGS framework
INTENSITY = {
    1: "extremely",     # low pole
    2: "very",          # low pole
    3: "",              # low pole (no modifier)
    4: "a bit",         # low pole
    5: None,            # neutral (neither/nor)
    6: "a bit",         # high pole
    7: "",              # high pole (no modifier)
    8: "very",          # high pole
    9: "extremely",     # high pole
}


def z_to_stanine(z):
    """Convert z-score to stanine (1-9).

    Standard stanine boundaries from Cangelosi (2000).
    """
    boundaries = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
    for i, b in enumerate(boundaries):
        if z < b:
            return i + 1
    return 9


def stanine_to_description(stanine, trait):
    """Convert stanine level to adjective phrase for a trait."""
    if stanine == 5:
        # Neutral: "neither X nor Y" format
        high_adj = MARKERS[trait]["high"]
        low_adj = MARKERS[trait]["low"]
        pairs = []
        for lo, hi in zip(low_adj, high_adj):
            pairs.append(f"neither {lo} nor {hi}")
        return ", ".join(pairs[:-1]) + ", and " + pairs[-1]

    if stanine < 5:
        pole = "low"
        modifier = INTENSITY[stanine]
    else:
        pole = "high"
        modifier = INTENSITY[stanine]

    adjectives = MARKERS[trait][pole]
    if modifier:
        adj_list = [f"{modifier} {adj}" for adj in adjectives]
    else:
        adj_list = adjectives

    return ", ".join(adj_list[:-1]) + ", and " + adj_list[-1]


def generate_persona(z_vector, persona_id):
    """Generate a persona description from a z-score vector.

    Follows Okada et al. (2026) Appendix E/F format exactly:
    - Second person ("You are...")
    - Trait sentence order: O, C, E, A, N
    - Framed with "YOU ARE THE RESPONDENT" prefix and
      "Answer all questions AS THIS PERSON would" suffix

    Args:
        z_vector: array of 5 z-scores [A, C, E, N, O]
        persona_id: string identifier

    Returns:
        dict with persona_id, z_scores, stanines, description, preamble
    """
    stanines = {t: z_to_stanine(z) for t, z in zip(TRAITS, z_vector)}

    # Build trait sentences in Okada's order: O, C, E, A, N
    TRAIT_ORDER = ["O", "C", "E", "A", "N"]
    sentences = []
    for trait in TRAIT_ORDER:
        s = stanines[trait]
        desc = stanine_to_description(s, trait)
        sentences.append(f"You are {desc}")

    description = ". ".join(sentences) + "."

    # Build the full preamble (matching Okada Appendix F.1 exactly)
    preamble = (
        f"YOU ARE THE RESPONDENT. {description} "
        "Answer all questions AS THIS PERSON would."
    )

    return {
        "persona_id": persona_id,
        "z_scores": {t: round(float(z), 3) for t, z in zip(TRAITS, z_vector)},
        "stanines": {t: int(stanines[t]) for t in TRAITS},
        "description": description,
        "preamble": preamble,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Big Five personas with known profiles"
    )
    parser.add_argument("--n", type=int, default=50,
                        help="Number of personas to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str,
                        default="instruments/synthetic_personas.json",
                        help="Output JSON file")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Sample from multivariate normal with human Big Five correlations
    z_vectors = rng.multivariate_normal(
        mean=np.zeros(5),
        cov=HUMAN_BIG5_CORR,
        size=args.n
    )

    personas = []
    for i, z in enumerate(z_vectors):
        persona = generate_persona(z, f"s{i+1}")
        personas.append(persona)

    # Summary statistics
    z_mat = np.array([list(p["z_scores"].values()) for p in personas])
    stanine_mat = np.array([list(p["stanines"].values()) for p in personas])

    output = {
        "n_personas": args.n,
        "seed": args.seed,
        "traits": TRAITS,
        "correlation_matrix": HUMAN_BIG5_CORR.tolist(),
        "personas": personas,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {args.n} personas → {args.output}")
    print(f"\nZ-score summary:")
    for i, t in enumerate(TRAITS):
        print(f"  {t}: mean={z_mat[:,i].mean():.3f}  "
              f"sd={z_mat[:,i].std():.3f}  "
              f"range=[{z_mat[:,i].min():.2f}, {z_mat[:,i].max():.2f}]")
    print(f"\nSample correlations:")
    sample_corr = np.corrcoef(z_mat.T)
    for i, t in enumerate(TRAITS):
        print(f"  {t}: " + "  ".join(f"{sample_corr[i,j]:+.2f}"
                                      for j in range(5)))
    print(f"\nStanine distribution:")
    for i, t in enumerate(TRAITS):
        counts = np.bincount(stanine_mat[:, i], minlength=10)[1:]
        print(f"  {t}: {dict(enumerate(counts, 1))}")


if __name__ == "__main__":
    main()
