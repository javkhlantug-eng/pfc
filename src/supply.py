"""Нийлүүлэлтийн тал: хязгаарлалт илрүүлэх, салхи/нарны таамаг, боломжит чадал.

Хязгаарлалт (түүхэнд ч ашиглана):
  ОХУ-ын импорт хязгаартаа ойр (≥285 МВт) БА дулааны станцууд сүүлийн 60 хоногийн
  дээд хэмжээнийхээ ≥95%-д ажилласан цаг → систем нөөцгүй, хязгаарлалт байж болзошгүй.

Боломжит чадал (цаг бүр):
  дулааны станцын боломжит (сүүлийн 30 хоногийн дээд) + импорт 300 + Зэс-Оюу 80
  + батарей (сүүлийн 30 хоногийн дээд цэнэг өгөлт) + усан станц + салхи, нарны таамаг
"""
import numpy as np
import pandas as pd

import sklearn
if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 0):   # Python 3.6 (Site5)
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

IMPORT_RU_MAX = 300      # ОХУ-аас импортын дээд хязгаар, МВт
ZES_OYU_MAX = 80         # Хятадаас (Зэс-Оюу) импортын дээд хязгаар, МВт
STRESS_IMPORT = 285      # импорт үүнээс дээш = хязгаартаа ойр
STRESS_THERMAL = 0.95    # дулааны станц дээд хэмжээнийхээ энэ хувиас дээш

# Салхи, нарны станцуудын байршил ба жин (нийт үйлдвэрлэлд эзлэх хувь)
WIND_SITES = [
    {"name": "Дорноговь", "lat": 44.89, "lon": 110.14, "w": 1 / 3},
    {"name": "Өмнөговь", "lat": 43.70, "lon": 105.60, "w": 1 / 3},
    {"name": "Улаанбаатар", "lat": 47.65, "lon": 107.10, "w": 1 / 3},
]
SOLAR_SITES = [
    {"name": "Дорноговь", "lat": 44.89, "lon": 110.14, "w": 0.7},
    {"name": "Төв бүс", "lat": 47.92, "lon": 106.92, "w": 0.3},
]
SITE_VARS = ["wind_speed_100m", "shortwave_radiation"]


def site_key(s):
    return "{:.2f}_{:.2f}".format(s["lat"], s["lon"])


def all_sites():
    seen, out = set(), []
    for s in WIND_SITES + SOLAR_SITES:
        k = site_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ---------------- Хязгаарлалт ----------------
def thermal_cap_by_date(load):
    """Өдөр бүрийн «дулааны станцын дээд хэмжээ» — өмнөх 60 хоногийн дээд (тухайн өдрийг оруулахгүй)."""
    dm = load.assign(d=pd.to_datetime(load["date"])).groupby("d")["thermal"].max()
    return dm.rolling(60, min_periods=10).max().shift(1)


def stress_flags(load):
    """Цаг бүрээр True/False — систем нөөцгүй байсан (хязгаарлалт байж болзошгүй)."""
    if "import" not in load.columns or "thermal" not in load.columns:
        return pd.Series(False, index=load.index)
    cap = pd.to_datetime(load["date"]).map(thermal_cap_by_date(load))
    imp = pd.to_numeric(load["import"], errors="coerce")
    th = pd.to_numeric(load["thermal"], errors="coerce")
    return ((imp >= STRESS_IMPORT) & (th >= STRESS_THERMAL * cap)).fillna(False)


# ---------------- Салхи, нарны таамаг ----------------
def _site_frame(weather_sites):
    """Урт хэлбэрийн (time, site, хувьсагчид) өгөгдлийг өргөн хүснэгт болгоно."""
    w = weather_sites.copy()
    w["time"] = pd.to_datetime(w["time"])
    wide = w.pivot_table(index="time", columns="site", values=SITE_VARS)
    wide.columns = ["{}@{}".format(v, s) for v, s in wide.columns]
    return wide.sort_index()


def _re_features(times, wide, kind):
    sites = WIND_SITES if kind == "wind" else SOLAR_SITES
    var = "wind_speed_100m" if kind == "wind" else "shortwave_radiation"
    X = pd.DataFrame(index=pd.DatetimeIndex(times))
    for s in sites:
        col = "{}@{}".format(var, site_key(s))
        X[s["name"]] = wide[col].reindex(X.index).values if col in wide.columns else np.nan
    X["weighted"] = sum(X[s["name"]] * s["w"] for s in sites)
    if kind == "wind":
        X["weighted3"] = np.clip(X["weighted"], 0, 25) ** 3     # салхины чадал ~ хурд³
    X["hour"] = X.index.hour
    doy = X.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return X


def re_forecast(load, weather_sites, target_times):
    """Салхи, нарны цаг тутмын таамаг (МВт). Түүхэн үйлдвэрлэл ↔ тухайн байршлын цаг агаараас сурна."""
    wide = _site_frame(weather_sites)
    hist = load.copy()
    hist["ts"] = pd.to_datetime(hist["date"]) + pd.to_timedelta(hist["hour"], unit="h")
    out = {}
    for kind in ("wind", "solar"):
        y = pd.to_numeric(hist[kind], errors="coerce").clip(lower=0)
        Xh = _re_features(hist["ts"], wide, kind)
        ok = y.notna().values & Xh.notna().all(axis=1).values
        if ok.sum() < 500:
            out[kind] = np.zeros(len(target_times))
            continue
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, random_state=0)
        m.fit(Xh[ok], y.values[ok])
        Xt = _re_features(target_times, wide, kind)
        pred = np.clip(m.predict(Xt.fillna(Xt.median())), 0, None)
        out[kind] = pred
    return out


# ---------------- Боломжит чадал ----------------
def supply_components(load, target_date):
    """Цаг агаараас үл хамаарах хэсгүүд (зорилтот өдрөөс өмнөх 30 хоногоор)."""
    d = pd.to_datetime(load["date"])
    recent = load[(d < pd.Timestamp(target_date)) & (d >= pd.Timestamp(target_date) - pd.Timedelta(days=30))]
    num = lambda c: pd.to_numeric(recent[c], errors="coerce") if c in recent.columns else pd.Series([0.0])
    return {
        "thermal": float(num("thermal").max()),
        "import_ru": float(IMPORT_RU_MAX),
        "zes_oyu": float(ZES_OYU_MAX),
        "battery": float(max(num("battery").max(), 0)),
        "hydro": float(max(num("hydro").max(), 0)),
    }


def hourly_supply(load, weather_sites, target_date):
    """Зорилтот өдрийн 24 цагийн боломжит чадал ба бүрдэл."""
    times = [pd.Timestamp(target_date) + pd.Timedelta(hours=h) for h in range(1, 25)]
    comp = supply_components(load, target_date)
    base = sum(comp.values())
    re = re_forecast(load, weather_sites, times) if weather_sites is not None else {"wind": np.zeros(24), "solar": np.zeros(24)}
    rows = []
    for i, h in enumerate(range(1, 25)):
        w, s = float(re["wind"][i]), float(re["solar"][i])
        rows.append({"hour": h, "wind_fc": round(w, 1), "solar_fc": round(s, 1), "supply": round(base + w + s, 1)})
    return rows, comp
