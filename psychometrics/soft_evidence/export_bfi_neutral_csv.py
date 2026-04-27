"""Export the BFI+neutral pickle to CSV with flattened logprob columns for R analysis."""

import pandas as pd
import numpy as np
import sys

pkl_path = "results_bfi-neutral_llama3.2-3b_0.pkl"
csv_path = "results_bfi-neutral_llama3.2-3b_0.csv"

print(f"Loading {pkl_path}...")
df = pd.read_pickle(pkl_path)
print(f"  {len(df)} rows, {len(df.columns)} columns")

# Flatten option_logprobs dict into separate columns
for k in ['1', '2', '3', '4', '5']:
    df[f'logprob_{k}'] = df['option_logprobs'].apply(
        lambda d: d.get(k, np.nan) if isinstance(d, dict) else np.nan
    )

# Create logprob argmax column (corrected top-1 response)
def get_argmax(d):
    if not isinstance(d, dict) or not d:
        return None
    valid = {k: v for k, v in d.items() if v is not None}
    if not valid:
        return None
    return max(valid, key=valid.get)

df['logprob_argmax'] = df['option_logprobs'].apply(get_argmax)

# Create SPID
df['spid'] = df['item_preamble_id'] + '_' + df['item_postamble_id']

# Flag neutral vs persona preambles
df['is_neutral'] = df['item_preamble_id'].str.startswith('d0-')

# Report contamination
contam = ((df['model_output'] == '3') & (df['logprob_argmax'] != '3')).sum()
n_neutral = df['is_neutral'].sum()
n_persona = (~df['is_neutral']).sum()
print(f"Contaminated rows (model_output=3 but argmax!=3): {contam}")
print(f"Neutral rows: {n_neutral}, Persona rows: {n_persona}")
print(f"logprob_argmax distribution:\n{df['logprob_argmax'].value_counts().sort_index()}")
print(f"\nBy condition:")
print(f"  Neutral: {df.loc[df['is_neutral'], 'logprob_argmax'].value_counts().sort_index().to_dict()}")
print(f"  Persona: {df.loc[~df['is_neutral'], 'logprob_argmax'].value_counts().sort_index().to_dict()}")

# Drop dict column and prompt_text (large) for CSV export
out = df.drop(columns=['option_logprobs', 'prompt_text'])
out.to_csv(csv_path, index=False)
print(f"Exported {len(out)} rows to {csv_path}")
