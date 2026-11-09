# Certified maps for robot exploration

Research code for a multi-robot explorer whose shared state is a *conformally certified* set-valued occupancy
map: every committed cell label carries a distribution-free guarantee, and a team fuses its members' e-values
into one certified map. The code covers validity on the operator's query law under adaptive data collection, a
team decomposition of the certification budget, a certification dichotomy that forecasts how much of a map a
robot can establish, and a second certificate — calibrated on a robot's own traversal outcomes — for acting on
a certified map while the world moves.

CPU-only Python, no dependencies beyond NumPy, SciPy, Pandas and Matplotlib, no GPU. Runs are deterministic.

## Repository map

| path | contents |
|---|---|
| `code/README.md` | what each module and experiment does — **start here** |
| `code/certmap/` | world, sensor, occupancy model, conformal certificate, planners, traversal guard, simulation |
| `code/experiments/` | one script per experiment, plus the analysis and figure scripts |
| `code/data/houseexpo/` | rasterized floor plans, used as real geometry |
| `code/results/` | small derived outputs (summary CSVs and reports); the per-run cache is not tracked |

## Reproducing

```bash
conda create -n certmap python=3.11 numpy scipy matplotlib pandas
conda activate certmap
cd code
PYTHONPATH=. python experiments/e2_validity.py 16      # 16 worker processes
PYTHONPATH=.:experiments python experiments/e2_analysis.py
```

Runs are cached by a hash of their configuration, so re-running a script only executes what is missing. The
cache itself (`code/results/**/*.pkl`, ~900 MB) is not in the repository; the scripts regenerate it. The two
real LiDAR logs (Radish: Intel and FR079) are fetched separately — see [`code/README.md`](code/README.md).




Every reported quantity can be rechecked from the cache without re-simulating, including an independent
re-audit that recomputes them separately from the scripts that produced them:

```bash
PYTHONPATH=.:experiments python experiments/report.py
PYTHONPATH=.:experiments python experiments/audit.py
```

## License

MIT, see [`LICENSE`](LICENSE). The HouseExpo floor plans under `code/data/houseexpo/` are derived from the
HouseExpo dataset and carry their original terms; the Radish LiDAR logs are not redistributed here.
