#!/bin/sh
cd "$(dirname "$0")/.."
for e in e1_objective e1b_reference e2_validity e5_score e6_flip e3_team e3c_coverage_mask e10_hetero e11_houseexpo e8_joint e7_ablation e9_teamsize; do
  python experiments/$e.py 16 > results/$e.log 2>&1
done
echo ALLDONE > results/queue_done.txt
