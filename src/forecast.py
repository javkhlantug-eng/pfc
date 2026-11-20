"""Загвар сургаж, маргаашийн 24 цагийн ачааллыг таамаглаад дашбоардын JSON-уудыг гаргана.

  python src/forecast.py                  # маргаашийн таамаг
  python src/forecast.py --target 2026-11-22
"""
import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd
import sklearn
try:  # scikit-learn < 1.0 (Python 3.6) дээр туршилтын модуль
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingRegressor

_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
L1_LOSS = "absolute_error" if _SK >= (1, 0) else "least_absolute_deviation"

from config import DATA, DOCS_DATA, CAPACITY_MW, ub_now, parse_date
from features import FEATURES, build_frame

BACKTEST_DAYS = 60


HOLDOUT_DAYS = 90


def make_model(leaves=31, min_leaf=30):
    return HistGradientBoostingRegressor(loss=L1_LOSS, max_iter=500, learning_rate=0.04,
                                         max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                                         l2_regularization=1.0, random_state=0)


def _dates(df):
    return df["date"] if "date" in df.columns else pd.Series(df.index, index=df.index)


def fit_with_intervals(train, test, feats, target, **kw):
    """P50 загвар + P10/P90 хүрээ. Хүрээг сүүлийн HOLDOUT_DAYS өдрийн загварт
    үзээгүй алдааны квантилаас (харьцангуй %) тооцно — квантил GBM-ээс бодитой."""
    train = train.dropna(subset=[target])
    d = _dates(train)
    cut = d.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    a, b = train[d <= cut], train[d > cut]
    rel = None
    if len(a) > 100 and len(b) > 10:
        m0 = make_model(**kw).fit(a[feats], a[target])
        rel = (b[target] - m0.predict(b[feats])) / b[target]
    m = make_model(**kw).fit(train[feats], train[target])
    p50 = m.predict(test[feats])
    # Сүүлийн үеийн системтэй хазайлтыг (жишээ нь ачааллын жилийн өсөлт) медиан алдаагаар засна
    lo, mid, hi = (rel.quantile([0.1, 0.5, 0.9]).values if rel is not None else (-0.03, 0.0, 0.03))
    return pd.DataFrame({"p10": p50 * (1 + lo), "p50": p50 * (1 + mid), "p90": p50 * (1 + hi)},
                        index=test.index)


def fit_predict(train, test):
    return fit_with_intervals(train, test, FEATURES, "total_mw")


DAILY_FEATURES = ["dow", "doy_sin", "doy_cos", "is_weekend", "is_holiday", "near_holiday",
                  "temp_min_day", "temp_mean_day", "temp_evening", "temp_mean72_eve",
                  "wind_mean", "prev_day_peak", "prev_day_mean", "prev_day_temp_mean", "peak_lag7"]


def daily_frame(frame):
    """Өдрийн оргилыг шууд таамаглах загварт зориулсан өдрийн мөрүүд.
    (Цагийн муруйн дээд цэг нь бодит оргилоос системтэйгээр бага гардаг тул
    оргилын МВт-ыг тусдаа загвараар гаргана.)"""
    g = frame.groupby("date")
    eve = frame[frame["hour"].between(17, 22)].groupby("date")
    d = pd.DataFrame({
        "peak": g["total_mw"].max(),
        "n": g["total_mw"].count(),
        "temp_mean_day": g["temp"].mean(), "temp_evening": eve["temp"].mean(),
        "temp_mean72_eve": eve["temp_mean72"].mean(), "wind_mean": g["wind_speed"].mean(),
    })
    first = g.first()
    for c in ["dow", "doy_sin", "doy_cos", "is_weekend", "is_holiday", "near_holiday",
              "temp_min_day", "prev_day_peak", "prev_day_mean", "prev_day_temp_mean"]:
        d[c] = first[c]
    d.loc[d["n"] < 24, "peak"] = np.nan
    d["peak_lag7"] = d["peak"].reindex(d.index - pd.Timedelta(days=7)).values
    return d


def fit_predict_daily(train_d, test_d):
    return fit_with_intervals(train_d, test_d, DAILY_FEATURES, "peak", leaves=15, min_leaf=10)


def peak_of(df, col):
    i = df[col].idxmax()
    return int(df.loc[i, "hour"]), float(df.loc[i, col])


def backtest(frame):
    """Сүүлийн BACKTEST_DAYS өдрийг нууж шалгана. Анхаар: энд бодит температур
    ашигладаг тул бодит ажиллагааны алдаа (таамагласан температуртай) арай их байна."""
    hist = frame.dropna(subset=["total_mw"])
    days = sorted(hist["date"].unique())
    if len(days) < BACKTEST_DAYS + 60:
        return None
    cut = days[-BACKTEST_DAYS]
    tr, te = hist[hist["date"] < cut], hist[hist["date"] >= cut].copy()
    te[["p10", "p50", "p90"]] = fit_predict(tr, te)
    te["naive"] = te["lag168"]
    dd = daily_frame(hist)
    dp = fit_predict_daily(dd[dd.index < cut], dd[dd.index >= cut])

    rows = []
    for d, g in te.groupby("date"):
        if len(g) < 24:
            continue
        ah, av = peak_of(g, "total_mw")
        ph, pv = peak_of(g, "p50")
        nh, nv = peak_of(g.dropna(subset=["naive"]), "naive") if g["naive"].notna().any() else (np.nan, np.nan)
        rows.append(dict(date=d, actual_peak=av, actual_hour=ah, pred_peak=float(dp.loc[d, "p50"]),
                         curve_peak=pv, pred_hour=ph, naive_peak=nv, naive_hour=nh,
                         in_band=float(dp.loc[d, "p10"]) <= av <= float(dp.loc[d, "p90"])))
    r = pd.DataFrame(rows)
    mape = lambda a, p: float(np.nanmean(np.abs(a - p) / a) * 100)
    return {
        "days": int(len(r)),
        "period": [str(pd.Timestamp(r["date"].min()).date()), str(pd.Timestamp(r["date"].max()).date())],
        "hourly_mape": round(mape(te["total_mw"], te["p50"]), 2),
        "hourly_mape_naive": round(mape(te["total_mw"], te["naive"]), 2),
        "peak_mae_mw": round(float(np.mean(np.abs(r["actual_peak"] - r["pred_peak"]))), 1),
        "peak_mae_mw_naive": round(float(np.nanmean(np.abs(r["actual_peak"] - r["naive_peak"]))), 1),
        "peak_mae_mw_curve": round(float(np.mean(np.abs(r["actual_peak"] - r["curve_peak"]))), 1),
        "peak_mape": round(mape(r["actual_peak"], r["pred_peak"]), 2),
        "peak_hour_within_1h_pct": round(float((np.abs(r["actual_hour"] - r["pred_hour"]) <= 1).mean() * 100), 1),
        "peak_in_p10_p90_pct": round(float(r["in_band"].mean() * 100), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="YYYY-MM-DD (өгөгдмөл: маргааш, Улаанбаатарын цагаар)")
    ap.add_argument("--no-backtest", action="store_true")
    a = ap.parse_args()

    now = ub_now()
    target = parse_date(a.target) if a.target else now.date() + dt.timedelta(days=1)

    load = pd.read_csv(DATA / "load_hourly.csv", dtype={"date": str})
    load = load[pd.to_datetime(load["date"]).dt.date < target]
    weather = pd.read_csv(DATA / "weather_hourly.csv")
    fc_path = DATA / "weather_forecast.csv"
    if fc_path.exists():
        weather = pd.concat([weather, pd.read_csv(fc_path)]).drop_duplicates("time", keep="first")

    tgt = pd.DataFrame({"date": target.isoformat(), "hour": range(1, 25), "total_mw": np.nan})
    frame = build_frame(pd.concat([load, tgt], ignore_index=True), weather)

    train = frame[(frame["date"].dt.date < target)].dropna(subset=["total_mw"])
    test = frame[frame["date"].dt.date == target].copy()
    if test["temp"].isna().all():
        raise SystemExit(f"{target}-ний температурын таамаг алга. collect_weather.py-г ажиллуулна уу.")
    test[["p10", "p50", "p90"]] = fit_predict(train, test)

    ph, _ = peak_of(test, "p50")                       # оргил цаг — цагийн муруйгаас
    dd = daily_frame(frame[frame["date"].dt.date <= target])
    tdate = pd.Timestamp(target)
    pk = fit_predict_daily(dd[dd.index < tdate], dd.loc[[tdate]]).iloc[0]   # оргил МВт — өдрийн загвараас
    fc = {
        "issued_at": now.isoformat(timespec="minutes"),
        "target_date": target.isoformat(),
        "capacity_mw": CAPACITY_MW,
        "peak": {"hour": ph, "label": f"{ph:02d}:00", "p50": round(float(pk["p50"]), 1),
                 "p10": round(float(pk["p10"]), 1), "p90": round(float(pk["p90"]), 1)},
        "hours": [{"hour": int(r.hour), "label": f"{int(r.hour):02d}:00",
                   "p10": round(r.p10, 1), "p50": round(r.p50, 1), "p90": round(r.p90, 1),
                   "temp": None if pd.isna(r.temp) else round(float(r.temp), 1)}
                  for r in test.itertuples()],
        "train_days": int(train["date"].nunique()),
    }

    # Таамгийн түүх — өчигдрийн таамгийг бодит гүйцэтгэлтэй харьцуулах
    log_p = DATA / "forecast_log.csv"
    new_log = test[["hour", "p10", "p50", "p90"]].assign(target_date=target.isoformat(),
                                                         issued_at=fc["issued_at"]).round(1)
    log = pd.concat([pd.read_csv(log_p), new_log]) if log_p.exists() else new_log
    log = log.drop_duplicates(["target_date", "hour"], keep="last").sort_values(["target_date", "hour"])
    log.to_csv(log_p, index=False)

    # Сүүлийн 7 хоногийн бодит ба таамаг
    hist = frame.dropna(subset=["total_mw"])
    last = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=6)]
    lg = log.assign(date=pd.to_datetime(log["target_date"]))
    last = last.merge(lg[["date", "hour", "p50"]], on=["date", "hour"], how="left")
    recent = [{"date": str(r.date.date()), "hour": int(r.hour), "actual": r.total_mw,
               "forecast": None if pd.isna(r.p50) else r.p50,
               "temp": None if pd.isna(r.temp) else round(float(r.temp), 1)} for r in last.itertuples()]

    # Өдрийн оргил ба температур (бүх түүх) — хамаарлын график
    g = hist.groupby("date")
    daily = pd.DataFrame({"peak": g["total_mw"].max(), "tmin": g["temp"].min(), "tmean": g["temp"].mean(),
                          "peak_hour": g.apply(lambda x: int(x.loc[x["total_mw"].idxmax(), "hour"]))})
    daily_js = [{"date": str(d.date()), "peak": round(r.peak, 1),
                 "tmin": None if pd.isna(r.tmin) else round(r.tmin, 1),
                 "tmean": None if pd.isna(r.tmean) else round(r.tmean, 1), "peak_hour": int(r.peak_hour)}
                for d, r in daily.iterrows()]

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    dump = lambda name, obj: (DOCS_DATA / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    dump("forecast.json", fc)
    dump("recent.json", recent)
    dump("daily.json", daily_js)
    if not a.no_backtest:
        m = backtest(frame[frame["date"].dt.date < target])
        if m:
            m["computed_at"] = fc["issued_at"]
            dump("metrics.json", m)
            print("Шалгалт:", m)
    print(f"{target} оргил: {fc['peak']['label']} цагт ~{fc['peak']['p50']} МВт "
          f"(P10–P90: {fc['peak']['p10']}–{fc['peak']['p90']})")


if __name__ == "__main__":
    main()
