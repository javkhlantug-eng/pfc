"""Туршилт: Улаанбаатарын температурын урьдчилсан таамгийн бодит алдаа.

Open-Meteo Previous Runs API нь загвар бүрийн өмнө гаргасан таамгийг хадгалдаг:
  temperature_2m_previous_day1 — 1 хоногийн өмнөх ажиллагааны таамаг (~24–47 цагийн өмнө)
  temperature_2m_previous_day2 — 2 хоногийн өмнөх ажиллагааны таамаг (~48–71 цагийн өмнө)
Манай таамаг 02:00-д гарч маргаашийн 01–24 цагийг хамардаг тул ~23–46 цагийн өмнөх
таамаг (previous_day1) хамгийн ойр.

Загвар бүрээр татаж data/exp/tempfc_<model>.csv-д хадгална. Шинжилгээг тусад нь хийнэ.
"""
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from config import DATA, LOCATIONS, TZ

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODELS = ["best_match", "ecmwf_ifs025", "ecmwf_ifs", "ecmwf_aifs025_single", "gfs_seamless",
          "icon_seamless", "jma_seamless", "cma_grapes_global", "ukmo_seamless",
          "meteofrance_seamless", "gem_seamless", "kma_seamless"]
VARS = ["temperature_2m", "temperature_2m_previous_day1", "temperature_2m_previous_day2"]
START = dt.date(2024, 1, 1)
OUT = DATA / "exp"


def get(params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=90)
            if r.status_code == 400:
                return None, r.json().get("reason", r.text[:200])
            r.raise_for_status()
            return pd.DataFrame(r.json()["hourly"]), None
        except requests.RequestException as e:
            if i == tries - 1:
                return None, str(e)[:200]
            time.sleep(10 * (i + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ub = LOCATIONS[0]
    end = dt.date.today() - dt.timedelta(days=1)
    for m in MODELS:
        parts, errs = [], []
        s = START
        while s <= end:
            e = min(s + dt.timedelta(days=90), end)
            df, err = get({"latitude": ub["lat"], "longitude": ub["lon"], "hourly": ",".join(VARS),
                           "models": m, "timezone": TZ, "start_date": s.isoformat(), "end_date": e.isoformat()})
            if df is not None:
                parts.append(df)
            else:
                errs.append("{}: {}".format(s, err))
            s = e + dt.timedelta(days=1)
            time.sleep(0.5)
        if not parts:
            print("{:24s} өгөгдөлгүй ({})".format(m, errs[:1]))
            continue
        d = pd.concat(parts, ignore_index=True)
        d.columns = [c.replace("_" + m, "") for c in d.columns]
        d = d.dropna(subset=[c for c in VARS if c in d.columns], how="all")
        d.to_csv(OUT / "tempfc_{}.csv".format(m), index=False)
        print("{:24s} {} мөр, {} – {}, алдаатай хэсэг {}".format(
            m, len(d), d["time"].min() if len(d) else "-", d["time"].max() if len(d) else "-", len(errs)))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
