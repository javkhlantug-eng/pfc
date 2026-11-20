"""Загварын шинж чанарууд.

Таамгийг D өдрийн орой D+1-ийн 24 цагт зориулж гаргана. Тиймээс ачааллын
хоцрогдол нь зөвхөн бүрэн дууссан өдрүүдийг ашиглана:
  (D өдөр таамаг гаргах үед D дуусаагүй байдаг тул D+1-ээс 2 хоногийн өмнөх буюу D-1 бол сүүлийн бүрэн өдөр)
  lag48  — зорилтот өдрөөс 2 хоногийн өмнөх яг тэр цаг
  lag168 — 7 хоногийн өмнөх яг тэр цаг
  prev_day_* — 2 хоногийн өмнөх өдрийн оргил, дундаж, температур
Температур: сургалтад бодит (reanalysis), таамаглахад цаг агаарын таамаг.
"""
import numpy as np
import pandas as pd

from config import DATA

# Тогтмол огноотой баярууд (сар, өдөр)
FIXED = {(1, 1), (3, 8), (6, 1), (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (11, 26), (12, 29)}

FEATURES = [
    "hour", "dow", "doy_sin", "doy_cos", "is_weekend", "is_holiday", "near_holiday",
    "temp", "hdh", "temp_mean24", "temp_mean72", "temp_min_day", "temp_change24",
    "wind_speed", "cloud_cover",
    "lag48", "lag168", "prev_day_peak", "prev_day_mean", "prev_day_temp_mean",
]


def load_holidays() -> set:
    p = DATA / "holidays.csv"
    extra = set(pd.read_csv(p)["date"]) if p.exists() else set()
    return extra


def _holiday_flag(dates: pd.Series, extra: set) -> pd.Series:
    iso = dates.dt.strftime("%Y-%m-%d")
    fixed = [(m, d) in FIXED for m, d in zip(dates.dt.month, dates.dt.day)]
    return (iso.isin(extra) | pd.Series(fixed, index=dates.index)).astype(int)


def build_frame(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """load: date,hour,total_mw (зорилтот өдрүүдэд total_mw хоосон байж болно)
    weather: time, temperature_2m, wind_speed_10m, cloud_cover — тасралтгүй цагийн цуваа."""
    # Ачааллын файлд станцын төрлийн баганууд (wind, solar…) байдаг тул зөвхөн хэрэгтэйг нь авна
    df = load[["date", "hour", "total_mw"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ts"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")  # hour=1 -> 01:00

    w = weather.rename(columns={"temperature_2m": "temp", "wind_speed_10m": "wind_speed"}).copy()
    w["time"] = pd.to_datetime(w["time"])
    w = w.set_index("time").sort_index()
    w = w[~w.index.duplicated()].asfreq(pd.offsets.Hour()).interpolate(limit=6)
    w["temp_mean24"] = w["temp"].rolling(24, min_periods=12).mean()
    w["temp_mean72"] = w["temp"].rolling(72, min_periods=36).mean()
    w["temp_change24"] = w["temp"] - w["temp"].shift(24)
    df = df.merge(w[["temp", "wind_speed", "cloud_cover", "temp_mean24", "temp_mean72", "temp_change24"]],
                  left_on="ts", right_index=True, how="left")

    # Өдрийн хамгийн бага температур (тухайн өдрийн)
    wd = w["temp"].groupby(w.index.normalize()).agg(["min", "mean"])
    df["temp_min_day"] = df["date"].map(wd["min"])
    df["hdh"] = np.clip(18 - df["temp"], 0, None)

    # Цаг/календарь
    df["dow"] = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    extra = load_holidays()
    df["is_holiday"] = _holiday_flag(df["date"], extra)
    prev = _holiday_flag(df["date"] - pd.Timedelta(days=1), extra)
    nxt = _holiday_flag(df["date"] + pd.Timedelta(days=1), extra)
    df["near_holiday"] = ((prev + nxt) > 0).astype(int)

    # Ачааллын хоцрогдол (date, hour түлхүүрээр)
    key = df.set_index(["date", "hour"])["total_mw"]
    def lag(days):
        idx = pd.MultiIndex.from_arrays([df["date"] - pd.Timedelta(days=days), df["hour"]])
        return key.reindex(idx).values
    df["lag48"] = lag(2)
    df["lag168"] = lag(7)

    daily = df.groupby("date")["total_mw"].agg(["max", "mean"])
    d2 = df["date"] - pd.Timedelta(days=2)   # хамгийн сүүлийн бүрэн өдөр
    df["prev_day_peak"] = d2.map(daily["max"])
    df["prev_day_mean"] = d2.map(daily["mean"])
    df["prev_day_temp_mean"] = d2.map(wd["mean"])
    return df
