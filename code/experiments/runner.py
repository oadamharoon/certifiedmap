"""Parallel experiment runner: a list of Config dicts -> one pickle per run."""
from __future__ import annotations

import os, sys, pickle, time, hashlib, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multiprocessing import Pool
import numpy as np
from certmap.sim import Simulation, Config

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run_id(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _one(args):
    exp, cfg = args
    path = os.path.join(RESULTS, exp, run_id(cfg) + ".pkl")
    if os.path.exists(path):
        return path
    t = time.time()
    out = Simulation(Config(**cfg)).run()
    out["traj"] = [tr.astype(np.float32) for tr in out["traj"]]
    out["wall"] = time.time() - t
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return path


def run_all(exp: str, cfgs: list[dict], workers: int = 16):
    os.makedirs(os.path.join(RESULTS, exp), exist_ok=True)
    todo = [(exp, c) for c in cfgs]
    print(f"[{exp}] {len(todo)} runs, {workers} workers", flush=True)
    t = time.time()
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_one, todo, chunksize=1)):
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                print(f"[{exp}] {i+1}/{len(todo)} done, {time.time()-t:.0f}s", flush=True)


def load(exp: str):
    import pandas as pd
    rows, runs = [], {}
    d = os.path.join(RESULTS, exp)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pkl"):
            continue
        out = pickle.load(open(os.path.join(d, fn), "rb"))
        rid = fn[:-4]
        runs[rid] = out
        for r in out["log"]:
            rows.append({**r, "rid": rid, **{f"cfg_{k}": v for k, v in out["config"].items()}})
    return pd.DataFrame(rows), runs
