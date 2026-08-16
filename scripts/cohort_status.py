"""Blinkenlights: self-refreshing local status page for the wide-n queue.

Loops forever: every ~10s, reads queue_state.json + capture dirs + log
tails + disk, writes tmp/status/index.html (meta-refresh 15s). Serve with:
  python -m http.server 8765 -d tmp/status
Never crashes on partial/missing files — a status page that dies is worse
than no status page.

Usage: .venv/bin/python scripts/cohort_status.py
"""
import datetime
import glob
import html
import json
import os
import shutil
import time

STATE = "results/cohort100/queue_state.json"
QLOG = "results/cohort100/queue.log"
OUT = "tmp/status/index.html"
MANIFEST = "instruments/cohort100_manifest.json"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>cohort100 blinkenlights</title><style>
body {{ background:#0a0f0a; color:#4ade80; font-family:ui-monospace,Menlo,monospace;
       font-size:13px; margin:24px; }}
h1 {{ color:#86efac; font-size:15px; }} h2 {{ color:#86efac; font-size:13px;
      margin:18px 0 6px; }}
table {{ border-collapse:collapse; }} td,th {{ padding:1px 14px 1px 0;
      text-align:left; white-space:nowrap; }}
.dim {{ color:#365d3f; }} .run {{ color:#fde047; }} .fail {{ color:#f87171; }}
.done {{ color:#4ade80; }} .stage {{ color:#93c5fd; }}
pre {{ color:#365d3f; margin:4px 0; }}
</style></head><body>
<h1>cohort100 queue — {now} (auto-refresh 15s)</h1>
<p>{summary}</p>
<h2>in flight</h2>{inflight}
<h2>ledger</h2><table>{ledger}</table>
<h2>captures on disk</h2><p>{captures}</p>
<h2>log tail</h2><pre>{logtail}</pre>
</body></html>"""


def safe_json(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def tail(path, n=14):
    try:
        lines = open(path, errors="replace").readlines()[-n:]
        return html.escape("".join(l for l in lines
                                   if "it/s" not in l and "?B/s" not in l))
    except Exception:
        return "(no log)"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    order = [m["name"] for m in
             safe_json(MANIFEST, {"models": []})["models"]]
    while True:
        st = safe_json(STATE, {})
        counts = {}
        rows = []
        for name in order + [k for k in st if k not in order]:
            v = st.get(name)
            if not v:
                continue
            s = v.get("status", "?")
            counts[s] = counts.get(s, 0) + 1
            cls = {"done": "done", "failed": "fail", "running": "run",
                   "staged": "stage", "downloading": "stage"}.get(s, "dim")
            extra = ""
            if v.get("run_min"):
                extra = f"{v['run_min']:.0f} min"
            if v.get("error"):
                extra = html.escape(str(v["error"])[:90])
            rows.append(f"<tr class={cls}><td>{name}</td><td>{s}</td>"
                        f"<td>{extra}</td></tr>")
        done = counts.get("done", 0)
        total = len(order)
        # honest ETA: observed wall-clock spacing between model completions
        # (the pipeline is download-bound; GPU minutes wildly understate).
        # completion time ~ newest mtime among a model's capture artifacts.
        manifest = safe_json(MANIFEST, {"models": []})["models"]
        stamps = []
        for m in manifest:
            safe_repo = m["repo"].replace("/", "_")
            arts = [f"results/adjectives/selfreport/{safe_repo}_self_full.json",
                    f"results/cohort100/default_enact/{safe_repo}_default.pt",
                    f"results/adjectives/acts/{safe_repo}__pers.pt"]
            ts = [os.path.getmtime(a) for a in arts if os.path.exists(a)]
            if len(ts) == len(arts):
                stamps.append(max(ts))
        eta = ""
        if len(stamps) >= 3:
            stamps.sort()
            span_h = (stamps[-1] - stamps[0]) / 3600
            rate = (len(stamps) - 1) / span_h if span_h > 0 else 0
            if rate > 0:
                eta = (f" | observed {rate:.1f} models/h -> "
                       f"~{(total - done) / rate:.0f} h left (download-bound)")
        free = shutil.disk_usage(".").free / 1e9
        summary = (f"done {done}/{total} | failed {counts.get('failed', 0)}"
                   f" | staged {counts.get('staged', 0)} | disk "
                   f"{free:.0f} GB free{eta}")
        inflight = "<br>".join(
            f"<span class=run>{name}: {v.get('status')}</span>"
            for name, v in st.items()
            if v.get("status") in ("downloading", "running")) or \
            "<span class=dim>(between models)</span>"
        caps = []
        for label, pat in [("self", "results/adjectives/selfreport/*_self_full.json"),
                           ("think", "results/adjectives/selfreport/*_think.json"),
                           ("enact", "results/cohort100/default_enact/*_default.pt"),
                           ("repr", "results/adjectives/acts/*__pers.pt")]:
            caps.append(f"{label}: {len(glob.glob(pat))}")
        page = PAGE.format(
            now=datetime.datetime.now().strftime("%H:%M:%S"),
            summary=summary, inflight=inflight, ledger="".join(rows),
            captures=" | ".join(caps), logtail=tail(QLOG))
        tmp = OUT + ".tmp"
        open(tmp, "w").write(page)
        os.replace(tmp, OUT)
        time.sleep(10)


if __name__ == "__main__":
    main()
