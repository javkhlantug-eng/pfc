#!/bin/bash
# bayasaltugs.com/forecast — Site5 хостинг дээр нэг удаа суулгах скрипт.
# cron-оор минут тутам дуудагдана; дууссаны дараа .setup_done тэмдэг үүсгээд юу ч хийхгүй.
set -u
APP="$HOME/forecast_app"; WEB="$HOME/public_html/forecast"
mkdir -p "$APP/logs"
exec >>"$APP/logs/setup.log" 2>&1
[ -f "$APP/.setup_done" ] && exit 0
exec 9>"$APP/.setup.lock"; flock -n 9 || exit 0
echo "== setup $(date)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8

python3 -m pip install --user -q --upgrade "pip<22" || { echo "pip upgrade failed"; exit 1; }
python3 -m pip install --user -q --prefer-binary "numpy==1.19.5" "pandas==1.1.5" "scikit-learn==0.24.2" "requests<2.28" \
  || { echo "lib install failed (will retry next minute)"; exit 1; }
python3 -c "import numpy,pandas,sklearn,requests;print('libs ok', numpy.__version__, pandas.__version__, sklearn.__version__)" || exit 1

TMP=$(mktemp -d)
curl -sfL https://github.com/javkhlantug-eng/pfc/archive/refs/heads/main.tar.gz | tar xz -C "$TMP" || { echo "download failed"; exit 1; }
SRC="$TMP/pfc-main"
rm -rf "$APP/src" "$APP/site5"; cp -r "$SRC/src" "$SRC/site5" "$APP/"
mkdir -p "$APP/data" "$WEB/data"
for f in "$SRC"/data/*; do [ -f "$APP/data/$(basename "$f")" ] || cp "$f" "$APP/data/"; done
for f in "$SRC"/docs/data/*.json; do [ -f "$WEB/data/$(basename "$f")" ] || cp "$f" "$WEB/data/"; done
sed "s#^const DATA_BASE = .*#const DATA_BASE = '';#" "$SRC/docs/index.html" > "$WEB/index.html"
rm -rf "$TMP"
echo "code + seed data installed"

bash "$APP/site5/run.sh" --force-forecast
touch "$APP/.setup_done"
echo "== setup done $(date)"
