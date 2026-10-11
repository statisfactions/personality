"""Console view of pkit.channels.pc1_summary (rgb 2026-10-10: data function in
pkit, presentation in the paper; see ../personality-paper/_analysis/channels.py).
Usage: PYTHONPATH=scripts .venv/bin/python scripts/pc1_loading_table.py
"""
import pandas as pd
from pkit import channels as chn

s = chn.pc1_summary()
pd.set_option("display.width", 200)
print(s["cos"].round(2)); print(pd.DataFrame({"share": s["share"], "stability": s["stability"], "r_others": s["r_others"]}).round(2))
for c in chn.CHANNELS:
    print(f"\n{c}\n  +  {', '.join(s['top_pos'][c])}\n  -  {', '.join(s['top_neg'][c])}"
          f"\n  specific +  {', '.join(s['unique_pos'][c])}\n  specific -  {', '.join(s['unique_neg'][c])}")
