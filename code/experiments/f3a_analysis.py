"""F3a analysis: cross-robot committed-label disagreement under drift against a clean team.

Each robot's own certificate commits a label where exactly one of its two e-values clears the Markov
threshold 1/alpha, which is fuse([c], alpha) with a single member. For a pair (i, j) the denominator is the
cells where both gates are open and both robots commit, and the numerator is those where the committed labels
differ. In the drift condition agents 0 and 1 are clean and agent 2 drifts, so (0,2) and (1,2) are the
drifting pairs and (0,1) is a clean pair observing the same world, which pairs the comparison by seed.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from runner import load
from certmap.conformal import AgentCert

ALPHA = 0.05


def own_commit(m):
    """Single-robot certified labels from a robot's own e-values: Gamma = {y : e(y) < 1/alpha}."""
    G = m["e"] < 1.0 / ALPHA
    n = G.sum(1)
    committed = n == 1
    label = np.where(committed, np.where(G[:, 1], 1.0, -1.0), 0.0)
    return committed, label


def main():
    df, runs = load("f3a")
    rows = []
    for rid, out in runs.items():
        cfg = out["config"]
        drift = tuple(cfg.get("drift_agents") or ())
        cond = "drift" if any(d > 0 for d in drift) else "clean"
        bad = {i for i, d in enumerate(drift) if d > 0}
        maps = out["maps"]
        own = [own_commit(m) for m in maps]
        for i, j in itertools.combinations(range(len(maps)), 2):
            ci, li = own[i]; cj, lj = own[j]
            both = maps[i]["gate"] & maps[j]["gate"] & ci & cj
            n = int(both.sum())
            dis = int((li[both] != lj[both]).sum())
            overlap = int((maps[i]["gate"] & maps[j]["gate"]).sum())
            # how often each robot is itself wrong on the shared cells, for reference
            occ = out["occ"].ravel() if out["occ"].ndim > 1 else out["occ"]
            truth = np.where(occ > 0.5, 1.0, -1.0)
            rows.append(dict(rid=rid, seed=cfg["seed"], cond=cond,
                             pair=f"{i}{j}", kind="drifting" if (i in bad or j in bad) else "clean",
                             overlap=overlap, n_both=n,
                             disagree=dis, rate=dis / n if n else np.nan,
                             err_i=float((li[both] != truth[both]).mean()) if n else np.nan,
                             err_j=float((lj[both] != truth[both]).mean()) if n else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "f3a_pairs.csv"), index=False)

    def agg(sub):
        return pd.Series(dict(n_runs=sub.rid.nunique(), pairs=len(sub),
                              overlap=sub.overlap.mean(), n_both=sub.n_both.mean(),
                              rate=sub.rate.mean(), sd=sub.rate.std(),
                              ci95=1.96 * sub.rate.std() / max(np.sqrt(len(sub)), 1)))

    print("\nF3a -- cross-robot committed-label disagreement on overlapping gated cells")
    print("pre-registered rule: R3 feasible if the drifting pair's rate exceeds the clean pair's by >= 3x\n")
    g = R.groupby(["cond", "kind"]).apply(agg, include_groups=False)
    print(g.round(4).to_string())

    clean_ref = R[(R.cond == "clean")].rate.mean()
    drift_pairs = R[(R.cond == "drift") & (R.kind == "drifting")].rate.mean()
    within = R[(R.cond == "drift") & (R.kind == "clean")].rate.mean()
    print(f"\nratio vs clean condition      : {drift_pairs / clean_ref:.2f}x")
    print(f"ratio vs clean pair same world: {drift_pairs / within:.2f}x")

    # paired by seed, using the clean pair (0,1) inside the same drift run as the control
    d = R[(R.cond == "drift") & (R.kind == "drifting")].groupby("seed").rate.mean()
    c = R[(R.cond == "drift") & (R.kind == "clean")].groupby("seed").rate.mean()
    both = pd.concat([d.rename("drifting"), c.rename("clean_pair")], axis=1).dropna()
    both["ratio"] = both.drifting / both.clean_pair
    print("\npaired by seed (same world, same clean teammates):")
    print(both.round(4).to_string())
    print(f"\nmedian paired ratio {both.ratio.median():.2f}x   min {both.ratio.min():.2f}x   max {both.ratio.max():.2f}x")
    verdict = "PASS -- R3 feasible" if both.ratio.median() >= 3.0 else "FAIL -- R3 closed by its own screen"
    print(f"\nVERDICT: {verdict} (rule: median paired ratio >= 3.0)")


if __name__ == "__main__":
    main()
