#!/bin/bash
# Run ONCE on the login node (light work only: checks, downloads, a patch).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROOT"
mkdir -p data results logs pylib

echo "== 1. Python env check"
if ! $PY -c "import torch; print('torch', torch.__version__)"; then
  echo "Roshan's env does not run standalone. Copy the module/activate lines from his job scripts into env.sh:"
  grep -hE "module |source |activate|python" /project2/xiangren_1987/rshashid/group27-brainswap/jobs/* | sort | uniq -c | sort -rn | head -15
  exit 1
fi
for pkg in sklearn datasets; do
  $PY -c "import $pkg" 2>/dev/null || { echo "installing $pkg into pylib/"; \
    $PY -m pip install -q --target "$ROOT/pylib" $([ $pkg = sklearn ] && echo scikit-learn || echo datasets); }
done

echo "== 2. Data (needs internet, so it runs here, not on compute nodes)"
[ -f data/20ng/config.json ]      || $PY src/prep_real_data.py --data_dir data/20ng/
[ -f data/banking77/config.json ] || $PY src/data_pipeline_banking77.py --data_dir data/banking77/

echo "== 3. Leave-one-out fix for hmem"
$PY patch_loo.py src/train_lowdata.py

echo "== Setup done. Next: smoke test (see README)."
