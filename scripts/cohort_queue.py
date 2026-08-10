"""Producer-consumer queue for the wide-n SELF cohort (I/O-bound downloads).

A downloader thread stages HF snapshots ahead of the GPU (bounded by
--max-staged and --min-free-gb); the runner consumes staged models one at a
time (self_adjective_report.py --full), then DELETES the snapshot unless the
weights were already fully cached before this run (so the standing cohort's
models are never evicted). Fully resumable: a model whose *_self_full.json
exists is skipped outright; failed models are quarantined in the state file
and the queue moves on.

Modes:
  --verify      Hub metadata check of every manifest ID (no downloads) —
                catches ID rot / missing gating approvals cheaply. Run first.
  --dry-run     print the plan (skip/run/keep-after) and exit.
  --wait-pid P  don't touch the GPU until process P exits (queue behind an
                in-flight run; downloads proceed meanwhile).

Usage:
  PYTHONPATH=scripts python scripts/cohort_queue.py --verify
  PYTHONPATH=scripts python scripts/cohort_queue.py --wait-pid 44927
State: results/cohort100/queue_state.json
"""
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

MANIFEST = "instruments/cohort100_manifest.json"
STATE = "results/cohort100/queue_state.json"
OUT_DIR = "results/adjectives/selfreport"
LOG_DIR = "results/cohort100/logs"


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {}


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=2)


def result_path(repo):
    return f"{OUT_DIR}/{repo.replace('/', '_')}_self_full.json"


def free_gb(path="."):
    return shutil.disk_usage(path).free / 1e9


def cache_dir_for(repo):
    from huggingface_hub.constants import HF_HUB_CACHE
    return os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))


def fully_cached(repo):
    """True if a complete snapshot already exists locally (don't delete those)."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def verify(models):
    from huggingface_hub import model_info
    bad = 0
    for m in models:
        try:
            info = model_info(m["repo"])
            gate = getattr(info, "gated", False)
            note = " [gated:%s]" % gate if gate else ""
            print(f"  ok   {m['name']:16s} {m['repo']}{note}")
        except Exception as e:
            bad += 1
            print(f"  BAD  {m['name']:16s} {m['repo']}  -> {type(e).__name__}: {e}")
    print(f"{len(models)-bad}/{len(models)} resolve")
    return bad


def downloader(todo, staged, args, st, lock):
    from huggingface_hub import snapshot_download
    for m in todo:
        while staged.qsize() >= args.max_staged:
            time.sleep(20)
        while free_gb() < args.min_free_gb:
            print(f"[dl] low disk ({free_gb():.0f} GB free), waiting", flush=True)
            time.sleep(300)
        pre = fully_cached(m["repo"])
        with lock:
            st[m["name"]] = {"status": "downloading", "preexisting": pre}
            save_state(st)
        try:
            t0 = time.time()
            snapshot_download(m["repo"], ignore_patterns=["*.pth", "*.gguf",
                                                          "original/*",
                                                          "*consolidated*"])
            with lock:
                st[m["name"]]["status"] = "staged"
                st[m["name"]]["dl_min"] = round((time.time() - t0) / 60, 1)
                save_state(st)
            print(f"[dl] staged {m['name']} ({st[m['name']]['dl_min']} min)",
                  flush=True)
            staged.put(m)
        except Exception as e:
            with lock:
                st[m["name"]] = {"status": "failed",
                                 "error": f"download: {type(e).__name__}: {e}"}
                save_state(st)
            print(f"[dl] FAILED {m['name']}: {e}", flush=True)
    staged.put(None)  # sentinel


def run_one(m, args, st, lock):
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(f"{LOG_DIR}/{m['name']}.log", "a")
    env = dict(os.environ, PYTHONPATH="scripts")
    cmd = [sys.executable, "scripts/self_adjective_report.py",
           "--model", m["repo"], "--full"]
    t0 = time.time()
    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                         timeout=args.timeout * 60)
    log.close()
    ok = rc == 0 and os.path.exists(result_path(m["repo"]))
    with lock:
        st[m["name"]]["status"] = "done" if ok else "failed"
        st[m["name"]]["run_min"] = round((time.time() - t0) / 60, 1)
        if not ok:
            st[m["name"]]["error"] = f"runner rc={rc} (see {LOG_DIR}/{m['name']}.log)"
        save_state(st)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-staged", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=150)
    ap.add_argument("--timeout", type=float, default=180,
                    help="per-model run timeout, minutes")
    ap.add_argument("--keep", action="store_true",
                    help="never delete snapshots")
    ap.add_argument("--only", default=None,
                    help="comma list of names or families to include")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--wait-pid", type=int, default=None)
    args = ap.parse_args()

    models = json.load(open(args.manifest))["models"]
    if args.only:
        keys = set(args.only.split(","))
        models = [m for m in models
                  if m["name"] in keys or m["family"] in keys]
    if args.verify:
        sys.exit(1 if verify(models) else 0)

    st = load_state()
    todo = []
    for m in models:
        if os.path.exists(result_path(m["repo"])):
            print(f"[skip] {m['name']}: result exists")
            continue
        if st.get(m["name"], {}).get("status") == "failed":
            print(f"[skip] {m['name']}: quarantined "
                  f"({st[m['name']].get('error', '?')})")
            continue
        todo.append(m)
    if args.limit:
        todo = todo[:args.limit]
    if args.dry_run:
        for m in todo:
            keep = "keep-after" if fully_cached(m["repo"]) else "delete-after"
            print(f"  run  {m['name']:16s} {m['repo']}  [{keep}]")
        print(f"{len(todo)} to run, ~{free_gb():.0f} GB free")
        return
    if not todo:
        print("nothing to do")
        return

    if args.wait_pid:
        print(f"waiting on pid {args.wait_pid} before GPU work "
              "(downloads start now)", flush=True)

    staged = queue.Queue()
    lock = threading.Lock()
    th = threading.Thread(target=downloader,
                          args=(todo, staged, args, st, lock), daemon=True)
    th.start()

    while True:
        m = staged.get()
        if m is None:
            break
        while args.wait_pid and os.path.exists(f"/proc/{args.wait_pid}"):
            time.sleep(60)
        if args.wait_pid:  # macOS: no /proc — poll via kill(0)
            while True:
                try:
                    os.kill(args.wait_pid, 0)
                    time.sleep(60)
                except OSError:
                    break
        print(f"[run] {m['name']}", flush=True)
        try:
            ok = run_one(m, args, st, lock)
        except subprocess.TimeoutExpired:
            with lock:
                st[m["name"]]["status"] = "failed"
                st[m["name"]]["error"] = "timeout"
                save_state(st)
            ok = False
        print(f"[run] {m['name']} -> {'done' if ok else 'FAILED'}", flush=True)
        if not args.keep and not st[m["name"]].get("preexisting"):
            d = cache_dir_for(m["repo"])
            if os.path.exists(d):
                sz = sum(f.stat().st_size for f in
                         __import__("pathlib").Path(d).rglob("*")
                         if f.is_file()) / 1e9
                shutil.rmtree(d)
                print(f"[rm] {m['name']} snapshot freed {sz:.0f} GB "
                      f"({free_gb():.0f} GB free)", flush=True)
    print("=== QUEUE DONE ===", flush=True)
    done = sum(1 for v in st.values() if v.get("status") == "done")
    fail = [(k, v.get("error")) for k, v in st.items()
            if v.get("status") == "failed"]
    print(f"done={done}  failed={len(fail)}")
    for k, e in fail:
        print(f"  failed {k}: {e}")


if __name__ == "__main__":
    main()
