"""Write the durable E50 record: the numbers needed at author-response time without re-running 700 sims.

The per-run pickles are gitignored, so this freezes the derived result (detector separation, margins with
confidence intervals, and the left-uncertified table) into results/e50_record.txt and results/e50_margins.csv.
Regenerate with:  PYTHONPATH=.:experiments python experiments/e50_record.py
"""
import sys, os, io, subprocess, datetime
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from runner import load
from analysis import ci95
import e50_analysis as A

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def detector_table(buf):
    df, runs = load("e45")
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    agg = {}
    for _, r in last.iterrows():
        c = runs[r.rid]["config"]
        nm = A.NAMES.get(A.key_of(c))
        if nm is None:
            continue
        agg.setdefault(nm, {}).setdefault(r.agent, []).append(r.p_hat)
    print("DETECTOR SEPARATION -- end-of-mission p_hat by robot (source: e45 cache)", file=buf)
    print(f"{'condition':22s} {'p_hat by agent':44s} {'max/min':>8s}", file=buf)
    for nm in A.ORDER:
        if nm not in agg:
            continue
        ks = sorted(agg[nm]); m = [float(np.nanmean(agg[nm][k])) for k in ks]
        print(f"{nm:22s} {' '.join(f'{x:.3f}' for x in m):44s} {max(m)/max(min(m),1e-9):>7.2f}x", file=buf)
    print("\n  drift and clean are indistinguishable (~1.4x): a uniformly displaced map contradicts nothing,"
          "\n  so the detector is blind to exactly the fault the certificate is blind to.\n", file=buf)


def margins_csv():
    base = A.collect("e44", lambda c: c["planner"] if float(c.get("denom_delta", 0.0)) == 0.0 else None)
    trust = A.collect("e50", lambda c: float(c["trust_tau"]))
    taus = sorted({k[2] for k in trust})
    rows = ["condition,tau,n,margin,ci95"]
    for nm in A.ORDER:
        S = {s for (n, s, k) in base if n == nm and k == "certified"} & \
            {s for (n, s, k) in base if n == nm and k == "entropy_g"}
        m0 = np.array([base[(nm, s, "certified")] - base[(nm, s, "entropy_g")] for s in sorted(S)])
        a, h = ci95(m0)
        rows.append(f"{nm},plain,{len(m0)},{a:.4f},{h:.4f}")
        for t in taus:
            T = S & {s for (n, s, k) in trust if n == nm and k == t}
            if not T:
                continue
            m1 = np.array([base[(nm, s, "certified")] - trust[(nm, s, t)] for s in sorted(T)])
            b, hh = ci95(m1)
            rows.append(f"{nm},{t},{len(m1)},{b:.4f},{hh:.4f}")
    open(os.path.join(RES, "e50_margins.csv"), "w").write("\n".join(rows) + "\n")
    return len(rows) - 1


if __name__ == "__main__":
    buf = io.StringIO()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    print("E50 -- detector baseline for the coverage-map planner. Durable record for author response.", file=buf)
    print(f"generated {datetime.date.today()} at commit {sha}; 700 runs, 7 conditions x 5 thresholds x 20 seeds.", file=buf)
    print("regenerate the sims with: python experiments/e50_trust_baseline.py <workers>\n", file=buf)
    detector_table(buf)
    old = sys.stdout
    sys.stdout = buf
    try:
        A.main()
    finally:
        sys.stdout = old
    n = margins_csv()
    open(os.path.join(RES, "e50_record.txt"), "w").write(buf.getvalue())
    print(buf.getvalue()[:400])
    print(f"... wrote results/e50_record.txt ({len(buf.getvalue())} chars) and results/e50_margins.csv ({n} rows)")
