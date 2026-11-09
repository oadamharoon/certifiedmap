# certmap

Simulator, certificate and experiments for certified multi-robot occupancy mapping: a static study of the
certificate itself, and a changing-map study of what a committed label means once the world moves.

CPU only, no GPU, no ROS. About 2,000 lines in `certmap/` and one small script per experiment in `experiments/`.

## Environment

```bash
conda create -n certmap python=3.11 numpy scipy matplotlib pandas
conda activate certmap
```

Every entry point is run from this directory with `PYTHONPATH` set:

```bash
PYTHONPATH=. python experiments/e2_validity.py 16          # produce runs, 16 worker processes
PYTHONPATH=.:experiments python experiments/e2_analysis.py # read them back
```

Each worker is single-threaded (`OMP_NUM_THREADS=1` is set inside the runner) and one run uses ~150 MB, so the
worker count is simply how many cores you want to give it.

## The 60-second tour

A run is a sequence of **blocks**. In each block every robot:

1. **certifies** — fits its Bayesian occupancy model, forms split-conformal p-values on a query grid, turns them
   into e-values, and fuses its neighbourhood's e-values into one set-valued map (`certmap/conformal.py`);
   each cell comes out *committed free*, *committed occupied*, *undecided*, or *not certifiable* (no data);
2. is **recorded** — every metric is logged per block per robot (`Simulation.record`);
3. **plans** — scores candidate segments with an acquisition function and picks one (`certmap/planner.py`),
   sequentially and greedily across the team;
4. **executes** — drives the segment, scanning as it goes; the first `train_frac` of a segment's scans train the
   model, the rest go to the calibration pool (this split is what keeps the certificate valid).

A run ends when every robot has spent its path budget. `Simulation.run()` returns
`{"config": ..., "log": [per-block records], "traj": ...}` and that dict is what gets cached.

```python
from certmap.sim import Simulation, Config
out = Simulation(Config(seed=0, N=3, topology="mesh", planner="certified", budget=100/3)).run()
out["log"][-1]["c_whole"]      # certified fraction of the map at the end
```

## Layout

| module | contents |
|---|---|
| `certmap/world.py` | procedural 2D worlds (walls with doors, boxes) or a floor plan loaded from `.npy`; query grid; observable-cell mask; `set_occ` for time-varying occupancy |
| `certmap/sensor.py` | 2D LiDAR with range/bearing/pose noise; labelled samples (beam endpoint +1, free space −1); optional label flips (global, range-dependent, per-agent, or inside an unreliable region) |
| `certmap/features.py`, `certmap/model.py` | Bayesian linear occupancy map on local Gaussian features; banded precision, so predictive variance (and the planner's preposterior) is closed-form |
| `certmap/conformal.py` | split-conformal p-values with covariate-shift weights, recalibration on a held-out fold, e-values, the gate, neighbourhood fusion, and an e-BH commitment set logged as a diagnostic only (no reported result uses it); `calibrate_agent_grouped` is the Mondrian (per-reliability-class) variant |
| `certmap/planner.py` | candidate segments (local + frontier), acquisition functions, traversability, team allocation |
| `certmap/sim.py` | `Config`, `Agent`, `Simulation`: the block protocol above, plus the changing-map machinery (moving hazards, traversal guards, persistence certificate) |
| `certmap/viz.py` | partition-map plotting |
| `experiments/runner.py` | `run_all(exp, cfgs, workers)` → one pickle per run; `load(exp)` → a tidy DataFrame of every block record with the config attached as `cfg_*` columns |
| `experiments/analysis.py`, `compare.py` | curves on a common distance grid, 95% CIs, paired differences, Holm correction |

### Reading `sim.py`

It is the only long file (~1,000 lines) and is sectioned by `# ---- ... ----` banners in this order:
topology → moving hazards → sensing → certification → metrics → planning → execution → main loop.
`Config` at the top is grouped by concern, and the changing-map sections (hazards, guard, persistence certificate)
are independent of the static ones: with `n_hazards=0` and `guard="none"` none of that code runs.

## Caching and determinism

`run_all` writes one pickle per run to `results/<exp>/<md5 of the config>.pkl` and **skips runs whose pickle
already exists**, so re-running a script after adding conditions only executes the new ones.

Consequences worth knowing before you edit anything:

* Runs are deterministic: same config → bit-identical log. (`seed` fixes world and start poses, `noise_seed` the
  sensor and planner randomness, `start_seed` re-samples start poses on the same world.)
* The cache key is the hash of the **config dict passed in**, not of the code. If you change simulator behaviour,
  delete the affected pickles, or the next analysis silently mixes old and new behaviour. `DYNAMIC.md` records
  several occasions where that had to be done; superseded generations live in `results/_old_dynamic/`.
* Renaming or re-defaulting a `Config` field orphans every cached run that used it. Reordering fields and editing
  comments is safe (fields are hashed by name).

The result cache (~900 MB, 13k pickles) is **not** in the repository. Logs of past runs (`results/*.log`) and the
small derived CSV/TXT summaries are.

## Experiments

Scripts pair by number: `eN_*.py` produces runs and `eN_analysis.py` reads them back, so an analysis script is
not listed separately below unless it does something its producer does not. `compare.py` and `analysis.py` are
the shared utilities they call (endpoint tables, paired differences against a reference planner with Holm
correction, curves against distance), and `run_queue.sh` runs the first twelve static experiments in sequence.

Static study:

| script | question |
|---|---|
| `e0_reproduce.py` | reproduce the protocol of Mahajan et al. (2026) as a sanity check |
| `e1_objective.py`, `e1b_reference.py`, `e5_score.py`, `e7_ablation.py` | does planning *to certify* beat entropy / frontier / lawnmower for one robot? (it does not — 20-level ablation) |
| `e7_ablation_rev.py` | the same ablation over the revised configuration set |
| `e4_stopping.py`, `e4b_feasibility.py` | post hoc on the `e1` certified runs: predicted against realised certified gain with a gain-rate stopping rule, and the feasibility signal (distance to the nearest undecided cell) |
| `mechanism.py` | why the objectives coincide: how much of the commitment decision is variance-driven |
| `e2_validity.py` | is the certificate valid on the operator's query law, weighted vs unweighted? |
| `e3_team.py`, `e3c_coverage_mask.py`, `e8_joint.py`, `e9_teamsize.py`, `e19_frontier_team.py` | teams: certified state vs plan exchange, coverage mask, coordinated frontier; greedy vs joint allocation; team size |
| `e12_hetero_team.py`, `e13_hetero_sensor.py` | heterogeneous teams: an unreliable teammate, and different sensors (half range, a fifth of the beams). The sensor-capability case is the boundary result: it is inequality without unequal label error, and the certified state gains nothing |
| `e6_flip.py`, `e10_hetero.py` | the label-flip sweep across the miscoverage budget beta = s*alpha, and heteroscedastic sensing where label reliability decays with range |
| `e14_comm.py` | communication payload per block |
| `e15_real_lidar.py` | certificate replayed on real LiDAR logs (Intel, FR079) |
| `e16_combined.py` | exact block-combined certificate (Corollary 2) vs the pooled one. Each block's certificate is calibrated to the uniform law **on the cells it is selected for** (the newly gated ones), which is what the corollary needs; calibrating it to the whole gate instead is not covered by the marginal guarantee. Runs from before that fix are in `results/_old_e16_pretilt/` |
| `e17_graded.py`, `e17b_acquisition.py` | graded certification dichotomy and its forecast (Corollary 2), and the same measurement under a different acquisition |
| `e11_houseexpo.py` | the same questions on 20 real floor plans |
| `e11_reach.py` | reachable observable fraction per floor plan from the start pose, and c relative to it |
| `e18_starts.py` | placement resampling: metrics over many start-pose draws |
| `e32_hetero_uncertainty.py`, `e32b_drop_fixed.py` | heterogeneity beyond label reliability: does the certified state still beat a coverage mask when a teammate is untrustworthy for a *different* reason than flipped labels (poor localisation, among others; N=3 mesh, 100 m each, 20 worlds)? `e32b` reruns the drop-the-untrusted-robot baseline under a *fixed per-robot* budget, since E32 gave the survivors the whole team budget and so assumed a dropped robot's path budget is transferable. Both versions are reported |
| `e33_hetero_robust.py`, `e33b_drift_limit.py` | whether the E32 result survives a *drifting* pose estimate rather than zero-mean per-scan noise, and where drift stops being detectable. Drift yields a map that is wrong but self-consistent, and the certificate detects contradictions, so a uniformly shifted wall contradicts nothing. `e33b` sweeps up to 0.20 m per block so that limit of the guarantee is measured rather than assumed |
| `e34_real_hetero.py` | the heterogeneity mechanism on the real LiDAR logs. With no hardware, the honest real-data instantiation is to split one log into contiguous segments, treat each as a robot, degrade one, and ask what the shared state says about the cells that robot observed. The trajectory is fixed, so this measures the mechanism behind the planning result rather than the planning gain |
| `e35_rrt.py` | the candidate generator the literature expects: a tree-based (RRT-exploration) generator swapped under every acquisition, single robot and three-robot teams, clean and with one unreliable teammate, so the heterogeneity result is shown not to depend on ours |
| `e36_lookahead.py` | whether a non-myopic planner changes the single-robot negative: the same acquisitions under a receding-horizon beam search over 1, 2 and 3 segments, only the first executed, so the horizon is the only thing that varies |
| `e37_ltt_guard.py` | two guarantees for the same evidence. `certified` issues a marginal, all-or-nothing statement per (region x staleness) class at an ACI-adapted level; `ltt` admits one Learn-then-Test threshold over the whole set, an (alpha, delta) statement that needs enough calibration pairs to issue at all and otherwise refuses to the uncertified rule |
| `e38_matrix_fill.py` | fills the gaps in the heterogeneity matrix. The programme was a star design around (N=3, our generator); this crosses team size and generator with the fault types they were not crossed with, so neither "the effect grows with team size" nor "the effect does not depend on our generator" rests on a single fault |
| `e39_policy_agnostic.py` | the traversal certificate as a policy-independent runtime filter. The guard constrains which cells a planner may drive through and never chooses the route, so it should hold its level under any policy: the same guard over a random walk, nearest-frontier, information gain and the certified acquisition, at two hazard speeds, with realised miss rate per entered cell with and without it |
| `e40_topology.py` | communication graph: mesh / ring / star at N=5, unreliable robot at a leaf and at the hub. Tests Prop. 3(i) (validity on any graph) and how the margin tracks how far a bad robot's evidence travels |
| `e41_sensor_budget.py` | the sensor-capability conditions of `e13` repeated at E32's 100 m budget, so the capability null is comparable to the label-error rows |
| `e42_split_pilot.py`, `e43_split_full.py` | the sample-split (provable) certificate: four disjoint calibration folds (claim \| recal-score \| recal \| planning) with the e-value divided by the recalibrated level, so the pooled certificate is provable for the planner we deploy without Assumption 3. `Config.split4=True`; defaults are bit-for-bit unchanged. Costs 0.017-0.092 of certified fraction over the seven conditions of the heterogeneity table, so it is reported, not adopted |
| `e44_provable_denom.py` | exact-alpha variant: divide the e-value by the Chernoff bound of Prop. 4.3 rather than by the measured miscoverage, so coverage is alpha+delta with no leftover factor. `Config.denom_delta=delta`; both arms run so the comparison is paired by world. The factor exceeds the single-robot commitment headroom 1/s, so certified fraction falls 0.95 -> 0.16 and every margin inverts: reported, not adopted. The delta=0 arm also reproduces the seven margins of the heterogeneity table from scratch |
| `e45_effective_size.py` | the effective sample size the recalibration bound is entitled to: the self-normalized n_eff, the Bernstein n_w = (sum w)/max w that the weighted proof needs, and the scan-grouped n_ind that assumes nothing about dependence within a scan. Sets the factor 1+eta quoted in Sec. 4.1 |
| `e46_aci_updates.py` | T_g, the number of blocks each traversal class updates, which sets the excess in Prop. 8.1. Same settings as `e26` |
| `e47_wclip.py`, `e47_analysis.py` | weight clipping: cap the covariate-shift weights at a quantile (`Config.w_clip`) so the tail stops setting n~ = (sum w)/max w. Validity sweep plus the seven heterogeneity conditions, 440 runs. Coverage holds at every level and every margin stays inside its interval, and the correction falls from 1+eta = 2.95 to 1.78 (N=3) |
| `e48_clip_curve.py` | whether a *self-selecting* clip level would choose to clip. Captures the final-block `calibrate_agent` inputs and replays them over a grid of clip quantiles, so the whole curve costs six runs rather than a sweep. The rule minimises alpha(1+eta(q))(1+zeta(q))/(1-zeta(q)); it selects no clipping in all 18 robot-runs, because capping the top 2% of cells removes ~25% of the weight mass |
| `e49_variance.py` | whether a Bernstein step could replace max w. Measures S = sum p^2 m at the selected level: it is 0.156 +/- 0.058 of the worst case eps/n~, so there is ~6x of room, but S's own effective size is 1.39 +/- 0.64, so the quantity that would certify the better regime cannot itself be estimated. Max w is not a loose proof step |
| `e50_trust_baseline.py`, `e50_analysis.py` | a detector baseline for the coverage-map planner: a teammate enters the shared mask only while its self-estimated label-error rate `p_hat` clears `Config.trust_tau` (no oracle). 700 runs over the seven conditions and five thresholds; tau=1.0 is a control that must reproduce the plain arm run for run. At a fixed tau the certified state leads in all seven conditions; the baseline reaches parity only when tau is tuned per fault type |
| `e51_leverage.py` | whether a planner can move n~ at all. Hooks the planner to capture every *candidate* segment's predicted samples, not only the chosen one, and recomputes n~ exactly for each. Over 120 planning steps and 3 seeds the spread across candidates has median 1.013, and a knockout curve shows servicing the 1, 2 or 5 sparsest gated cells leaves n~ unchanged: the weight tail is a plateau of several hundred cells, not a few reachable outliers |
| `f3a_crossrobot.py`, `f3a_analysis.py` | whether cross-robot disagreement sees drift where `p_hat` does not. Each robot's own certificate commits a label from its own e-values; on cells where two gates overlap and both commit, the rate the labels differ is 0.0342 for a drifting pair against 0.0025 for a clean pair in the same world. The disagreeing cells are the drifter's wrong ones, so the signal attributes as well as detects |

F3b, which asks whether a rigid co-registration can absorb the F3a signal, and everything after it live in the follow-up project rather than here.
| `e20_region.py`, `e21_reliability_aware.py`, `e22_mondrian.py` | a localised unreliable region: where errors concentrate, a reliability-aware acquisition, and the per-class (Mondrian) certificate |

Changing-map study (`../DYNAMIC.md`):

| script | question |
|---|---|
| `e23_dynamic_pilot.py`, `e23b_dynamic_fast.py` | do moving hazards break the static certificate, and at what speed? |
| `e24_lapsing.py`, `e25_guard_sweep.py` | lapsing support; first traversal guards (superseded design, kept for the record) |
| `e26_gate2.py` | main table: seven traversal guards × hazard density × speed, the certificate's level sweep, and acquisitions under the guard |
| `e27_frontier_and_goal.py` | safety–progress frontier against the log-odds guard; goal-directed (persistent) hazard motion |
| `e27b_frontier_ext.py` | the same frontier extended upward: log-odds guard at thresholds 0.2/0.3/0.5 (decay 0.9) and the horizon certificate at 0.2, random-walk hazards at speeds 1 and 3, 20 worlds |
| `e28_teams_dynamic.py` | three-robot teams, with and without pooled traversal evidence |
| `e29_houseexpo_dynamic.py` | real floor plans with pedestrian-sized goal-directed hazards |
| `e30_stress.py` | stress regimes: faster, larger and denser hazards |
| `e31_ablation.py`, `e31b_tracker_goal.py` | ablation of the certificate's own settings; tracker baseline with oracle detections |
| `e26_analysis.py` | the analysis for all of the above: `PYTHONPATH=.:experiments python experiments/e26_analysis.py e28` |

`results/dynamic_tables.txt` holds the generated tables for e26–e31.

### What the changing-map metrics mean

Printed by `e26_analysis.py`, per run, summed over robots:

* `miss_cell` — the validity metric: cells that were **occupied when the robot entered them**, over cells entered
  on *certified* blocks. This is what the certificate's level α bounds.
* `strict` — the same over *every cell of an admitted segment*, including those beyond the point where the robot
  stopped. Always ≥ `miss_cell`; reported because the planner admitted those cells too.
* `miss_all` — over all traversals, including the uncertified warm-up and probe blocks.
* `unc_blocks` — blocks moved under the uncertified fallback rule (warm-up + probes).
* `idle` — blocks with no admissible segment (the robot turns and re-scans in place).
* `c_time` — progress: certified static fraction averaged over blocks 10…H on a common horizon (H = 80, or 40 for
  the shorter floor-plan runs), a run that ends earlier keeping its final state. Idling costs time, not budget,
  so progress is measured against *time*, not distance.
* Guards compared: `none`, `inflate` (obstacle inflation), `fresh` (committed free **and** observed in the last
  block — the practitioner's rule), `prob` (decaying log-odds occupancy grid), `cp_track` / `cp_track_oracle`
  (conformal regions around constant-velocity hazard tracks, with map or oracle detections), `certified` (ours),
  `oracle` (true occupancy at planning time).

## Figures and tables

These scripts write into output directories outside this tree: `../paper/figs` and `../paper/tables` for the
wider two-column layout, `../paper_iccps/figs` for the narrower one. Both are local and untracked, so create
them first (`mkdir -p ../paper_iccps/figs`). Pass `acm` as the first argument to `fig_paper.py` to target the
narrower column.

```bash
mkdir -p ../paper/figs ../paper/tables
PYTHONPATH=.:experiments python experiments/fig_paper.py     # validity, dichotomy, coordination figures
PYTHONPATH=.:experiments python experiments/fig_maps2.py     # the map figure
PYTHONPATH=.:experiments python experiments/fig_real.py      # real-log maps
PYTHONPATH=.:experiments python experiments/make_tables.py   # -> ../paper/tables/*.tex
PYTHONPATH=.:experiments python experiments/report.py        # every reported number, from the cache
```

All of these read the cache; none re-runs an experiment (except the figure scripts, which re-simulate the two or
three worlds they draw, because maps are not cached by default — `save_maps=False`).

Every figure script, and what it draws:

| script | figure |
|---|---|
| `fig_paper.py` | validity, dichotomy and coordination panels; pass `acm` as the first argument for the narrower column |
| `fig_mechanism.py` | calibrated gated cells in the (sigma, abs mu) plane, with the commitment regions |
| `fig_protocol.py` | the block protocol: why the certificate stays valid although the planner chooses where to look. Scans over the first 60% of a 5 m segment train the model, the rest calibrate it, so a block's model, gate and recalibrated level are functions of strictly earlier data |
| `fig_unequal.py` | what a coverage map and a certified map disagree about when one robot cannot be trusted: one world, three robots, the third flipping 10% of its labels |
| `fig_hetero.py` | left, the share left explored-but-uncertified by each planner per condition (what the team must come back for); right, the paired difference in certified fraction |
| `fig_topology.py` | whether the margin survives a sparser communication graph, and where it comes from: the three graphs at N=5 with the unreliable robot filled, plus leaf against hub for the star |
| `fig_maps.py`, `fig_maps2.py` | fused partition maps at equal budget: one robot under three objectives, and a team of three with one unreliable teammate under three coordination states |
| `fig_real.py`, `fig_real2.py` | the real logs: pseudo ground truth, the certificate on the law the robot sampled, the certificate on the operator's query law, and three contiguous segments treated as robots fusing e-values |
| `fig_dynamic.py` | left, realised miss rate against the level alpha by guard and hazard density/speed (the validity check); right, the safety-progress plane |
| `fig_guard_maps.py` | one world under three traversal rules, with every cell entered while occupied marked: no guard, obstacle inflation, the persistence certificate |

## Verification

Three scripts read the cache and nothing else, so a claim can be rechecked without re-running a simulation.

```bash
PYTHONPATH=.:experiments python experiments/report.py      # every reported number
PYTHONPATH=.:experiments python experiments/audit.py       # independent re-audit, see below
PYTHONPATH=.:experiments python experiments/e50_record.py  # freeze the E50 result into results/
```

`audit.py` recomputes the reported quantities from the pickles independently of the analysis scripts that
produced them, streaming each run and keeping only the final block per agent, which is what every endpoint
claim is computed from. It carries two checks worth knowing about. The "uncertified" column is the share of
observable cells *not committed*, which is undecided + empty, equivalently 1 - c: comparing it against
`undecided_whole` alone agrees wherever `empty` is small and silently disagrees where it is not. And a row's
margin over the coverage-map planner must equal its uncertified difference, since both are 1 - c; that
identity is the check that cannot drift, and it is what caught a wrong value once introduced by the first
comparison.


## Data

* `data/houseexpo/he*.npy` — 20 floor plans from HouseExpo, rasterised to 0.25 m by `experiments/prep_houseexpo.py`.
* `data/real/*.clf` — Intel and FR079 LiDAR logs (Radish). Not redistributed here; fetch them and drop them in
  this directory to run `e15_real_lidar.py`.
