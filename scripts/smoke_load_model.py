#!/usr/bin/env python3
"""Line-buffered smoke for new models. Loads model, prints progress to
stdout as it goes, runs one 7-digit Likert forward pass.

Usage: PYTHONPATH=scripts .venv/bin/python scripts/smoke_load_model.py <ModelKey>
"""

import gc
import sys
import time

import torch

print(f"PID {sys.argv}", flush=True)
sys.path.insert(0, "scripts")
from hf_logprobs import load_model, likert_distribution, resolve  # noqa

key = sys.argv[1] if len(sys.argv) > 1 else "Gemma27"
print(f"key={key}, hf={resolve(key)}", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] calling load_model...", flush=True)
t0 = time.time()
model, tok, device = load_model(key)
print(f"[{time.strftime('%H:%M:%S')}] loaded in {time.time()-t0:.1f}s, "
      f"device={device}, dtype={model.dtype}", flush=True)

if device == "mps":
    mem = torch.mps.current_allocated_memory() / 1e9
    print(f"  MPS memory allocated: {mem:.1f} GB", flush=True)

prompt = (
    "For the following pair of statements, indicate which one describes "
    "you more accurately and by how much using a 7-point bipolar scale:\n"
    "1: LEFT statement describes me much more accurately\n"
    "4: About the same\n"
    "7: RIGHT statement describes me much more accurately\n"
    "Return ONLY one integer (1-7).\n\n"
    "++++\nLEFT: Talk to a lot of different people at parties.\n||\n"
    "RIGHT: Retreat from others.\n++++\n"
)
print(f"[{time.strftime('%H:%M:%S')}] running inference...", flush=True)
t0 = time.time()
dist, argmax, h = likert_distribution(
    model, tok, prompt, device,
    digits=("1", "2", "3", "4", "5", "6", "7"),
    use_chat_template=True,
    system_content="You are a bit reserved, a bit quiet.",
)
elapsed = time.time() - t0
print(f"[{time.strftime('%H:%M:%S')}] 1-pair inference: {elapsed:.2f}s, "
      f"argmax={argmax}, entropy={h:.2f}", flush=True)
print(f"  distribution: {dist}", flush=True)

del model, tok
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
print("done", flush=True)
