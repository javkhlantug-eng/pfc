"""Open-Meteo-оос цагийн температур татна.

data/weather_hourly.csv  — өнгөрсөн (ERA5 reanalysis + сүүлийн хоногуудын хэмжилт/шинжилгээ)
data/weather_forecast.csv — маргааш, нөгөөдрийн таамаг (ажиллуулах бүрт дарагдана)
"""
import argparse
import datetime as dt

import pandas as pd
import requests

from config import DATA, LOCATIONS, WEATHER_VARS, ARCHIVE_URL, FORECAST_URL, LOAD_START, TZ, ub_now, parse_date


def _get(url, params):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df


def _weighted(frames):
    """Хэд хэдэн цэгийн жинлэсэн дундаж. Нэг цэг бол шууд буцаана."""
    if len(frames) == 1:
        return frames[0][0]
    tot = sum(w for _, w in frames) or 1.0
    base = frames[0][0][["time"]].copy()
    for v in WEATHER_VARS:
        base[v] = sum(f[v].values * w for f, w in frames) / tot
    return base


def fetch(url, start=None, end=None, past_days=None, forecast_days=None):
    frames = []
    for loc in LOCATIONS:
        if loc.get("weight", 0) <= 0:
            continue
        p = {"latitude": loc["lat"], "longitude": loc["lon"],
             "hourly": ",".join(WEATHER_VARS), "timezone": TZ}
        if start:
            p.update(start_date=start, end_date=end)
        if past_days is not None:
            p.update(past_days=past_days, forecast_days=forecast_days)
        frames.append((_get(url, p), loc["weight"]))
    return _weighted(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="бүх түүхийг дахин татах")
    a = ap.parse_args()

    out = DATA / "weather_hourly.csv"
    old = pd.read_csv(out, parse_dates=["time"]) if out.exists() and not a.full else pd.DataFrame()
    today = ub_now().date()

    # Archive нь ~5 хоногийн хоцрогдолтой.
    arch_end = today - dt.timedelta(days=6)
    arch_start = (old["time"].max().date() - dt.timedelta(days=10)) if not old.empty else parse_date(LOAD_START)
    parts = [old] if not old.empty else []
    if arch_start <= arch_end:
        parts.append(fetch(ARCHIVE_URL, arch_start.isoformat(), arch_end.isoformat()))

    # Сүүлийн хоногууд + ирэх 3 хоног — forecast API
    fc = fetch(FORECAST_URL, past_days=10, forecast_days=3)
    tomorrow0 = pd.Timestamp(today)  # өнөөдрийн 00:00-оос хойш таамаг гэж үзнэ
    parts.append(fc[fc["time"] < tomorrow0])

    hist = (pd.concat(parts, ignore_index=True)
              .drop_duplicates("time", keep="first")   # archive-ыг давуу эрхтэй
              .sort_values("time").reset_index(drop=True))
    DATA.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out, index=False)
    fc[fc["time"] >= tomorrow0].to_csv(DATA / "weather_forecast.csv", index=False)
    print(f"Цаг агаар: түүх {hist['time'].min()} – {hist['time'].max()}, таамаг {len(fc[fc['time'] >= tomorrow0])} цаг")


if __name__ == "__main__":
    main()
