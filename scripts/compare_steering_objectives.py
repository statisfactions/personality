#!/usr/bin/env python3
"""Compare three steering vector training objectives:

1. FC-logit:     Maximize logit(A) - logit(B) on a forced-choice prompt
2. Free-text:    Maximize log-prob of first N tokens of high-trait response
3. Persona:      Generate target with persona prompt, then steer vanilla toward it

All produce a δ at the same layer. We compare cosines, FC effectiveness,
and free-text generation quality.

Usage:
    python scripts/compare_steering_objectives.py \
        --model meta-llama/Llama-3.2-3B-Instruct --layer 12 --trait H
"""

import argparse
import gc
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CONTRAST_PAIRS = "instruments/contrast_pairs.json"


def load_model(model_name, device="mps"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()
    print(f"  Loaded, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer


def get_layer_module(model, layer_idx):
    for path in ["model.layers", "model.language_model.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj[layer_idx]
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Can't find layer {layer_idx}")


def make_hook(delta):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + delta.unsqueeze(0).unsqueeze(0),) + output[1:]
        return output + delta.unsqueeze(0).unsqueeze(0)
    return hook_fn


# =============================================================================
# Objective 1: FC logit difference
# =============================================================================
def train_fc_logit(model, tokenizer, pairs, layer_module, hidden_size,
                   device, norm_constraint, n_steps=50, lr=0.05, batch_size=5):
    """Maximize logit(A) - logit(B) on FC prompts."""
    print("\n  === Objective 1: FC-logit ===")

    delta = torch.randn(hidden_size, device=device, dtype=torch.float32) * 0.01
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    a_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_id = tokenizer.encode("B", add_special_tokens=False)[-1]

    prompts = []
    for s, h, l in pairs:
        prompts.append(
            f"Consider what a person most like you would do in the following situation: "
            f"{s}\n\nWhich would you do?\nA) {h}\nB) {l}\n\nRespond with just A or B.\nAnswer:"
        )

    for step in range(n_steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(prompts), min(batch_size, len(prompts)), replace=False)

        handle = layer_module.register_forward_hook(make_hook(delta))
        total_lo = torch.tensor(0.0, device=device, dtype=torch.float32)
        for i in idx:
            inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
            logits = model(**inputs, use_cache=False).logits[0, -1, :]
            total_lo = total_lo + (logits[a_id].float() - logits[b_id].float())
            del inputs
        handle.remove()

        loss = -total_lo / len(idx)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if delta.norm().item() > norm_constraint:
                delta.data *= norm_constraint / delta.norm().item()

        if step % 25 == 0 or step == n_steps - 1:
            print(f"    step {step:3d}  loss={loss.item():+8.3f}  |δ|={delta.norm().item():.3f}")

        del loss, total_lo
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    return delta.detach().clone()


# =============================================================================
# Objective 2: Free-text token likelihood
# =============================================================================
def train_freetext(model, tokenizer, pairs, layer_module, hidden_size,
                   device, norm_constraint, n_target_tokens=15,
                   n_steps=50, lr=0.05, batch_size=3):
    """Maximize log-prob of first N tokens of high-trait response in free-text framing."""
    print(f"\n  === Objective 2: Free-text (first {n_target_tokens} tokens) ===")

    delta = torch.randn(hidden_size, device=device, dtype=torch.float32) * 0.01
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    # Build prompt + target pairs
    items = []
    for s, h, l in pairs:
        prompt = (f"Consider what a person most like you would do in the following "
                  f"situation: {s}\n\nWhat would you do? I would ")
        # Target: first N tokens of the high-trait response
        target_text = h
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)[:n_target_tokens]
        items.append((prompt, target_ids))

    for step in range(n_steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(items), min(batch_size, len(items)), replace=False)

        handle = layer_module.register_forward_hook(make_hook(delta))
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for i in idx:
            prompt, target_ids = items[i]
            # Concatenate prompt + target tokens
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            full_ids = prompt_ids + target_ids
            input_ids = torch.tensor([full_ids], device=device)

            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0]  # (seq_len, vocab)

            # Loss: negative log-prob of each target token given prefix
            prompt_len = len(prompt_ids)
            for j, tid in enumerate(target_ids):
                pos = prompt_len + j - 1  # logits at pos predict token at pos+1
                log_prob = torch.log_softmax(logits[pos].float(), dim=-1)[tid]
                total_loss = total_loss - log_prob

            del outputs, input_ids

        handle.remove()

        total_loss = total_loss / len(idx)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if delta.norm().item() > norm_constraint:
                delta.data *= norm_constraint / delta.norm().item()

        if step % 25 == 0 or step == n_steps - 1:
            print(f"    step {step:3d}  loss={total_loss.item():8.3f}  |δ|={delta.norm().item():.3f}")

        del total_loss
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    return delta.detach().clone()


# =============================================================================
# Objective 3: Persona-guided steering
# =============================================================================
def train_persona(model, tokenizer, pairs, layer_module, hidden_size,
                  device, norm_constraint, trait_desc="extremely honest and principled",
                  n_target_tokens=15, n_steps=50, lr=0.05, batch_size=3):
    """Generate targets with persona prompt, then steer vanilla model toward them."""
    print(f"\n  === Objective 3: Persona-guided ({trait_desc[:30]}...) ===")

    # Step 1: Generate persona responses (no gradients, no hook)
    print("    Generating persona targets...")
    items = []
    for s, h, l in pairs:
        persona_prompt = (
            f"You are someone who is {trait_desc}. "
            f"Consider what you would do in the following situation: {s}\n\n"
            f"What would you do? I would "
        )
        inputs = tokenizer(persona_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=n_target_tokens, temperature=0.0,
                do_sample=False, use_cache=True,
            )
        # Extract just the generated tokens
        gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Vanilla prompt (what we'll steer)
        vanilla_prompt = (
            f"Consider what a person most like you would do in the following "
            f"situation: {s}\n\nWhat would you do? I would "
        )
        items.append((vanilla_prompt, gen_ids, gen_text))

        del inputs, out
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # Show a few targets
    for i in range(min(3, len(items))):
        print(f"    Target {i}: \"{items[i][2][:60]}...\"")

    # Step 2: Optimize δ to reproduce persona targets on vanilla prompt
    print("    Optimizing...")
    delta = torch.randn(hidden_size, device=device, dtype=torch.float32) * 0.01
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        idx = np.random.choice(len(items), min(batch_size, len(items)), replace=False)

        handle = layer_module.register_forward_hook(make_hook(delta))
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for i in idx:
            vanilla_prompt, target_ids, _ = items[i]
            prompt_ids = tokenizer.encode(vanilla_prompt, add_special_tokens=True)
            full_ids = prompt_ids + target_ids
            input_ids = torch.tensor([full_ids], device=device)

            outputs = model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits[0]

            prompt_len = len(prompt_ids)
            for j, tid in enumerate(target_ids):
                pos = prompt_len + j - 1
                log_prob = torch.log_softmax(logits[pos].float(), dim=-1)[tid]
                total_loss = total_loss - log_prob

            del outputs, input_ids

        handle.remove()

        total_loss = total_loss / len(idx)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if delta.norm().item() > norm_constraint:
                delta.data *= norm_constraint / delta.norm().item()

        if step % 25 == 0 or step == n_steps - 1:
            print(f"    step {step:3d}  loss={total_loss.item():8.3f}  |δ|={delta.norm().item():.3f}")

        del total_loss
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    return delta.detach().clone()


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_all(model, tokenizer, deltas, layer_module, eval_pairs, device):
    """Evaluate all deltas on FC accuracy and free-text generation."""
    names = list(deltas.keys())

    # FC evaluation
    a_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    b_id = tokenizer.encode("B", add_special_tokens=False)[-1]

    print(f"\n{'=' * 70}")
    print(f"  FC EVALUATION on {len(eval_pairs)} held-out pairs")
    print(f"{'=' * 70}")

    fc_results = {}
    for name in ["baseline"] + names:
        d = deltas.get(name)
        handle = None
        if d is not None:
            handle = layer_module.register_forward_hook(make_hook(d))

        n_high = 0
        for s, h, l in eval_pairs:
            prompt = (f"Consider what a person most like you would do in the following "
                      f"situation: {s}\n\nWhich would you do?\nA) {h}\nB) {l}\n\n"
                      f"Respond with just A or B.\nAnswer:")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs, use_cache=False).logits[0, -1, :]
            if logits[a_id] > logits[b_id]:
                n_high += 1
            del inputs

        if handle:
            handle.remove()
        fc_results[name if name != "baseline" else "baseline"] = n_high
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    for name, n in fc_results.items():
        print(f"  {name:20s}: {n}/{len(eval_pairs)} high-trait ({n/len(eval_pairs):.0%})")

    # Free-text generation (3 scenarios)
    print(f"\n{'=' * 70}")
    print(f"  FREE-TEXT GENERATION (3 scenarios)")
    print(f"{'=' * 70}")

    test_scenarios = [p[0] for p in eval_pairs[:3]]
    for scenario in test_scenarios:
        prompt = (f"Consider what a person most like you would do in the following "
                  f"situation: {scenario}\n\nWhat would you do? I would ")
        print(f"\n  Scenario: {scenario[:65]}...")

        for name in ["baseline"] + names:
            d = deltas.get(name)
            handle = None
            if d is not None:
                handle = layer_module.register_forward_hook(make_hook(d))

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=50, temperature=0.7,
                    do_sample=True, top_p=0.9, use_cache=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()

            if handle:
                handle.remove()
            label = name if name != "baseline" else "baseline"
            print(f"    {label:20s}: {text[:120]}")

            del inputs, out
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trait", default="H")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--norm-pct", type=float, default=5.0,
                        help="Norm constraint as %% of residual norm")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=25)
    parser.add_argument("--n-target-tokens", type=int, default=15)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    with open(CONTRAST_PAIRS) as f:
        cp = json.load(f)
    trait_data = cp["traits"][args.trait]
    all_pairs = [(p["situation"], p["high"], p["low"]) for p in trait_data["pairs"]]
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:]

    model, tokenizer = load_model(args.model, args.device)
    hidden_size = model.config.hidden_size
    layer_module = get_layer_module(model, args.layer)

    # Measure residual norm
    test_prompt = f"Consider what a person most like you would do: {train_pairs[0][0]}."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    resid_norm = out.hidden_states[args.layer][0, -1, :].float().norm().item()
    del out, inputs
    gc.collect()

    norm_constraint = resid_norm * args.norm_pct / 100.0
    print(f"\nResidual norm at layer {args.layer}: {resid_norm:.1f}")
    print(f"Norm constraint: {args.norm_pct}% = {norm_constraint:.3f}")
    print(f"Train: {len(train_pairs)}, Eval: {len(eval_pairs)}, "
          f"Target tokens: {args.n_target_tokens}")

    # Train all three
    t0 = time.time()
    delta_fc = train_fc_logit(model, tokenizer, train_pairs, layer_module,
                              hidden_size, args.device, norm_constraint,
                              n_steps=args.steps)

    delta_ft = train_freetext(model, tokenizer, train_pairs, layer_module,
                              hidden_size, args.device, norm_constraint,
                              n_target_tokens=args.n_target_tokens,
                              n_steps=args.steps, batch_size=3)

    TRAIT_DESCS = {
        "H": "extremely honest, sincere, and principled",
        "E": "very emotional, sensitive, and anxious",
        "X": "highly extraverted, sociable, and energetic",
        "A": "very agreeable, patient, and forgiving",
        "C": "extremely conscientious, organized, and disciplined",
        "O": "very open-minded, curious, and creative",
    }
    delta_persona = train_persona(model, tokenizer, train_pairs, layer_module,
                                  hidden_size, args.device, norm_constraint,
                                  trait_desc=TRAIT_DESCS.get(args.trait, "high on the trait"),
                                  n_target_tokens=args.n_target_tokens,
                                  n_steps=args.steps, batch_size=3)

    elapsed = time.time() - t0
    print(f"\n  All three objectives trained in {elapsed:.0f}s")

    # Compare directions
    deltas = {"fc-logit": delta_fc, "free-text": delta_ft, "persona": delta_persona}

    print(f"\n{'=' * 70}")
    print(f"  DIRECTION COMPARISON (cosine similarity)")
    print(f"{'=' * 70}")

    units = {}
    for name, d in deltas.items():
        units[name] = d.float() / d.float().norm()

    for n1 in deltas:
        for n2 in deltas:
            if n1 < n2:
                cos = torch.dot(units[n1], units[n2]).item()
                print(f"  {n1:20s} ↔ {n2:20s}: cosine = {cos:+.4f}")

    # Compare with LDA
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        safe = args.model.replace("/", "_")
        data = torch.load(f"results/repe/{safe}_{args.trait}_directions.pt",
                         weights_only=False)
        diffs = data["raw_diffs"]
        d = diffs[:, args.layer, :].numpy()
        n_pairs = d.shape[0]
        X = np.vstack([d / 2, -d / 2])
        y = np.array([1] * n_pairs + [0] * n_pairs)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        lda_d = lda.coef_[0]
        lda_d = lda_d / np.linalg.norm(lda_d)
        lda_t = torch.tensor(lda_d, dtype=torch.float32, device=args.device)

        print(f"\n  vs LDA direction:")
        for name in deltas:
            cos = torch.dot(units[name], lda_t / lda_t.norm()).item()
            print(f"  {name:20s} ↔ {'LDA':20s}: cosine = {cos:+.4f}")
    except Exception as e:
        print(f"  LDA comparison failed: {e}")

    # Evaluate all
    evaluate_all(model, tokenizer, deltas, layer_module, eval_pairs, args.device)


if __name__ == "__main__":
    main()
