"""Administers PsyBORGS tests to models on the Orin Ollama server.

Uses subprocess + curl for API calls (Python requests hangs over the
Netbird VPN tunnel — see CLAUDE.md known issues).

Expects OLLAMA_API_KEY to be set in the shell environment.

Example usage:
    export $(grep OLLAMA_API_KEY .env | tr -d "'")
    python inference_scripts/run_ollama_inference.py \
        --admin_session='admin_sessions/local/prod_run_01_hydrated.json' \
        --model_pointer='llama2:7b-chat' \
        --sample
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
from tqdm.auto import tqdm

# point system to project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from psyborgs import survey_bench_lib


# --- Payload generation (self-contained, no outlines dependency) ---

SPID = ['item_preamble_id',
        'item_postamble_id',
        'response_scale_id',
        'response_choice_postamble_id',
        'model_id']


def generate_payload_df(admin_session, model_id):
    """Returns sorted df of prompts, continuations, and info to be scored."""
    payload_list = []
    for measure_iteration in survey_bench_lib.measure_generator(admin_session):
        for prompt_iteration in survey_bench_lib.prompt_generator(
                measure_iteration, admin_session):
            for continuation_iteration in survey_bench_lib.continuation_generator(
                    measure_iteration, admin_session):
                payload_spec = survey_bench_lib.generate_payload_spec(
                    measure_iteration, prompt_iteration,
                    continuation_iteration, 0, model_id)
                payload_list.append(payload_spec)
    return pd.DataFrame(payload_list).sort_values(
        ['prompt_text', 'continuation_text'])


def to_generative_payload(df):
    """Collapse continuations into lists — one row per prompt-SPID combo."""
    return df \
        .sort_index() \
        .groupby(['prompt_text',
                  'measure_id',
                  'measure_name',
                  'scale_id',
                  'item_id'] + SPID) \
        .agg({'continuation_text': list,
              'response_value': list,
              'response_choice': list}) \
        .reset_index()

# --- Ollama API helpers ---

BASE_URL = "https://apollo.quocanmeomeo.io.vn"


def get_api_key():
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        print("ERROR: OLLAMA_API_KEY not set. "
              "Run: export $(grep OLLAMA_API_KEY .env | tr -d \"'\")")
        sys.exit(1)
    return key


def curl_chat_completion(model, messages, api_key, timeout=120):
    """Call the Ollama OpenAI-compatible chat endpoint via curl.

    Returns the assistant's response content string, or None on failure.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "stream": False,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    cmd = [
        "curl", "-s",
        "-m", str(timeout),
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", payload_json,
        f"{BASE_URL}/v1/chat/completions",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def query_with_retry(model, messages, api_key, max_retries=3, timeout=120):
    """Retry wrapper around curl_chat_completion."""
    for attempt in range(max_retries):
        response = curl_chat_completion(model, messages, api_key, timeout)
        if response is not None:
            return response
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return None


# --- Inference logic ---

def constrained_choice(prompt_text, continuations, model, api_key, timeout=120):
    """Ask the model to pick exactly one of the valid response choices.

    Constructs a chat message with the survey prompt and a system instruction
    constraining the model to respond with one of the provided choices verbatim.

    Returns the chosen continuation string, or None if parsing fails.
    """
    choices_str = "\n".join(f"- {c}" for c in continuations)
    system_msg = (
        "You are completing a personality questionnaire. "
        "You must respond with EXACTLY one of the following options, "
        "and nothing else:\n"
        f"{choices_str}\n\n"
        "Respond with only the chosen option text, no extra words."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]

    raw = query_with_retry(model, messages, api_key, timeout=timeout)
    if raw is None:
        return None

    # Try exact match first
    stripped = raw.strip()
    if stripped in continuations:
        return stripped

    # Fuzzy: check if any continuation is contained in the response
    stripped_lower = stripped.lower()
    for c in continuations:
        if c.lower() in stripped_lower:
            return c

    # Last resort: check if response starts with a continuation
    for c in continuations:
        if stripped_lower.startswith(c.lower()):
            return c

    return None


def administer_session_ollama(payload_df, model, api_key, timeout=120):
    """Send prompts to Ollama, grouped by response scale.

    Mirrors administer_session_via_outlines from run_gpt_inference.py.

    Args:
        payload_df: Generative payload df (one row per prompt).
        model: Ollama model name (e.g. 'llama2:7b-chat').
        api_key: Bearer token for the Ollama server.
        timeout: Curl timeout in seconds per request.

    Returns:
        DataFrame with 'model_output' column added.
    """
    response_scales = list(payload_df['response_scale_id'].unique())
    scored_dfs = []

    for response_scale in tqdm(
            response_scales, leave=True,
            desc="Iterating through subpayloads grouped by continuations"):
        grouped_df = payload_df.loc[
            payload_df['response_scale_id'] == response_scale].copy()

        continuations = grouped_df['continuation_text'].iloc[0]
        print(f"Working on continuations [{', '.join(continuations)}]")

        model_answers = []
        failed = 0
        for prompt in tqdm(grouped_df['prompt_text'], leave=True,
                           desc="Prompts"):
            answer = constrained_choice(
                prompt, continuations, model, api_key, timeout=timeout)
            if answer is None:
                failed += 1
                # Fall back to middle option
                answer = continuations[len(continuations) // 2]
            model_answers.append(answer)

        if failed > 0:
            print(f"  Warning: {failed}/{len(grouped_df)} prompts failed to "
                  f"parse; used middle option as fallback.")

        grouped_df['model_output'] = model_answers
        scored_dfs.append(grouped_df)

    return pd.concat(scored_dfs).sort_index()


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Administer PsyBORGS survey to an Ollama-hosted model.")
    parser.add_argument(
        '--admin_session', type=str, required=True,
        help='Path to hydrated admin_session JSON')
    parser.add_argument(
        '--model_pointer', type=str, default='llama2:7b-chat',
        help='Ollama model name (e.g. llama2:7b-chat)')
    parser.add_argument(
        '--job_name', type=str, default='ollama-run',
        help='Experiment name (used in output filenames)')
    parser.add_argument(
        '--job_id', type=str, default='0',
        help='Job ID for tracking')
    parser.add_argument(
        '--sample', action='store_true', default=False,
        help='Only sample 1,000 rows of the payload (for testing)')
    parser.add_argument(
        '--timeout', type=int, default=120,
        help='Curl timeout in seconds per inference call')
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = get_api_key()

    # Model ID for filenames (strip tag colons)
    MODEL_ID = args.model_pointer.replace(':', '-').replace('/', '-')

    # Load admin session
    admin_session = survey_bench_lib.load_admin_session(args.admin_session)

    # Generate payload
    payload_df = generate_payload_df(
        admin_session=admin_session, model_id=args.model_pointer)
    gen_payload = to_generative_payload(payload_df)

    if args.sample:
        print("Sampling only 1,000 prompts.")
        gen_payload = gen_payload.sample(
            min(1000, len(gen_payload)), random_state=42)

    print(f"Payload size: {len(gen_payload)}")

    # Run inference
    results = administer_session_ollama(
        gen_payload, model=args.model_pointer,
        api_key=api_key, timeout=args.timeout)

    # Save results
    if not os.path.exists('results'):
        os.makedirs('results')

    outfile = f"results/results_{args.job_name}_{MODEL_ID}_{args.job_id}.pkl"
    results.to_pickle(outfile)
    print(f"\nResults saved to {outfile}")


if __name__ == '__main__':
    main()
