#!/bin/bash
# Цаг тутам cron-оор ажиллана (цаг бүрийн 7 минутад):
#   - өчигдөр/өнөөдрийн ачаалал татах → өнөөдрийн график
#   - Улаанбаатарын 02 цагт: цаг агаар татаж маргаашийн таамаг гаргах
set -u
APP="$HOME/forecast_app"; WEB="$HOME/public_html/forecast"
export PFC_DATA="$APP/data" PFC_DOCS_DATA="$WEB/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
mkdir -p "$APP/logs"
exec 9>"$APP/.run.lock"; flock -n 9 || exit 0
exec >>"$APP/logs/run-$(TZ=Asia/Ulaanbaatar date +%Y-%m).log" 2>&1
H=$(TZ=Asia/Ulaanbaatar date +%H)
echo "== $(TZ=Asia/Ulaanbaatar date '+%F %T') УБ"
cd "$APP/src" || exit 1

python3 collect_load.py --days-back 1


if [ "$H" = "02" ] || [ "${1:-}" = "--force-forecast" ] || [ "${1:-}" = "--update" ] || [ ! -f "$WEB/data/forecast.json" ]; then
  python3 collect_weather.py && python3 forecast.py
fi

python3 today.py
