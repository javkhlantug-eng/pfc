"""Бодитоор гаргасан таамгийн алдаанаас суралцах.

Өмнө нь гаргаж хадгалсан таамгийг (forecast_log.csv, peak_log.csv) бодит гүйцэтгэлтэй
харьцуулж, шинэ таамгийг засна. Өмнөх таамгуудыг ӨӨРЧЛӨХГҮЙ — зөвхөн дараагийн
таамагт хэрэглэнэ.

  алдаа  e = (бодит − загварын түүхий таамаг) / бодит
  засвар c = дундаж(e) × n / (n + K)       — өгөгдөл цөөн үед бага жинтэй (shrinkage)
  шинэ таамаг = түүхий таамаг × (1 + c)

Загварын ТҮҮХИЙ (засвар хийхээс өмнөх) утгаар алдааг тооцдог тул засвар давхардахгүй.
Цаг агаарын таамгийн алдаа ч бодит алдаанд багтдаг тул загвар түүнийг ч тооцож сурна.
"""
import numpy as np
import pandas as pd

WINDOW_DAYS = 14   # сүүлийн хэдэн өдрийн алдааг харах
SHRINK_K = 7       # n өдрийн өгөгдөлтэй үед жин = n / (n + 7)
MAX_CORR = 0.08    # засвар ±8%-иас хэтрэхгүй
MIN_N_BAND = 7     # хүрээг бодит алдаанаас тооцох хамгийн бага өдөр


def _actuals(load):
    a = load[["date", "hour", "total_mw"]].copy()
    a["date"] = a["date"].astype(str)
    return a


def _window(df, target, col="target_date"):
    t = pd.Timestamp(target)
    d = pd.to_datetime(df[col])
    return df[(d < t) & (d >= t - pd.Timedelta(days=WINDOW_DAYS))]


def hourly_correction(log, load, target):
    """Цаг бүрийн засварын коэффициент {цаг: c} ба тооцоонд орсон өдрийн тоо."""
    if log is None or log.empty:
        return {}, 0
    lg = _window(log, target).copy()
    raw = lg["raw_p50"] if "raw_p50" in lg.columns else lg["p50"]
    lg["raw"] = raw.where(raw.notna(), lg["p50"])
    m = lg.merge(_actuals(load), left_on=["target_date", "hour"], right_on=["date", "hour"])
    if m.empty:
        return {}, 0
    m["e"] = (m["total_mw"] - m["raw"]) / m["total_mw"]
    corr = {}
    for h, g in m.groupby("hour"):
        n = len(g)
        corr[int(h)] = float(np.clip(g["e"].mean() * n / (n + SHRINK_K), -MAX_CORR, MAX_CORR))
    return corr, int(m["target_date"].nunique())


def peak_correction(peak_log, load, target):
    """Өдрийн оргилын засвар c, үлдэгдэл алдааны (P10, P90) квантиль, өдрийн тоо."""
    if peak_log is None or peak_log.empty:
        return 0.0, None, 0
    pl = _window(peak_log, target)
    act = _actuals(load)
    cnt = act.groupby("date")["hour"].count()
    daily = act.groupby("date")["total_mw"].max()[cnt >= 24]
    m = pl.merge(daily.rename("actual").reset_index(), left_on="target_date", right_on="date")
    if m.empty:
        return 0.0, None, 0
    e = (m["actual"] - m["raw_p50"]) / m["actual"]
    n = len(e)
    c = float(np.clip(e.mean() * n / (n + SHRINK_K), -MAX_CORR, MAX_CORR))
    band = None
    if n >= MIN_N_BAND:
        r = e - e.mean()
        band = (float(r.quantile(0.1)), float(r.quantile(0.9)))
    return c, band, n


def blend_band(p50, p10, p90, band, n):
    """Загварын хүрээг бодит алдааны хүрээтэй жинлэн холино (өгөгдөл ихсэх тусам бодитод ойртоно)."""
    if band is None:
        return p10, p90
    w = n / (n + 14.0)
    lo = (1 - w) * (p10 / p50 - 1) + w * band[0]
    hi = (1 - w) * (p90 / p50 - 1) + w * band[1]
    return p50 * (1 + min(lo, 0)), p50 * (1 + max(hi, 0))
