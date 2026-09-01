#!/usr/bin/env bash
# One-shot driver. Day 1 on the cluster: ./run_all.sh /path/to/imagenet/val
set -euo pipefail
VAL_ROOT="${1:?usage: ./run_all.sh /path/to/imagenet/val}"
N_PER_CLASS="${2:-50}"

echo "=== 0. offline checks ==="
python test_numpy_paths.py

echo "=== 1. subset ==="
python scripts/00_build_subset.py --val-root "$VAL_ROOT" --n-per-class "$N_PER_CLASS"

echo "=== 2. filter sanity (LOOK at figures/filter_sanity.png) ==="
python scripts/01_sanity_filters.py \
  --image "$(python - <<'PY'
import json; print(json.load(open('cache/subset.json'))['paths'][0])
PY
)"

echo "=== 3. go/no-go cache (small, ~20 min) ==="
python scripts/02_cache_profiles.py --models resnet18 \
  --corruptions clean gaussian_noise defocus_blur contrast --severities 5
python scripts/03_oracle_gonogo.py --model resnet18 | tee results/gonogo.txt

echo
echo ">>> READ results/gonogo.txt NOW. If HEADROOM is small, stop and change"
echo ">>> the view family before spending GPU hours on the full sweep."
read -rp ">>> Continue with the full sweep? [y/N] " ok
[[ "$ok" == "y" ]] || exit 0

echo "=== 4. full cache (both models, both view kinds) ==="
python scripts/02_cache_profiles.py --models resnet18 deit_tiny \
  --view-kinds lowpass spatial

echo "=== 5. tune, evaluate, ablate ==="
python scripts/04_tune.py
python scripts/05_evaluate.py
python scripts/06_ablations.py

echo "=== 6. figures and tables ==="
python scripts/07_figures.py
python scripts/08_tables.py

echo "=== done. results/ figures/ report/tables/ are populated ==="
