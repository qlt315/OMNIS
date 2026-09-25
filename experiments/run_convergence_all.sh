#!/bin/bash
# Full convergence run (edit slots/users/seeds as needed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/experiments"
OUT="$ROOT/figures/python figures/convergence"
mkdir -p "$OUT/series"
LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
export MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:."
python3 -u -c "
from convergence_lib import cli_main
cli_main(default_algos=['all'], argv=[
    '--algos','all','--slots','1000','--users','25',
    '--seeds','0','1','2','3','4','--plot'])
" 2>&1 | tee "$LOG"
echo "EXIT:$?"
python3 - <<PY
import csv
from pathlib import Path
rows=list(csv.DictReader(Path(r"$OUT/summary.csv").open()))
print("=== FINAL ===")
for name in ["causal","gdo","cto","ucb","dts","dqn","ppo","mappo"]:
    d={}
    for r in rows:
        if r["name"]==name and r["metric"] in ("reward","acc","vio","ms_per_slot"):
            d[r["metric"]]=f"{float(r['mean']):.3f}+/-{float(r['std']):.3f}"
    if d:
        print(f"{name:8s}", d)
PY
