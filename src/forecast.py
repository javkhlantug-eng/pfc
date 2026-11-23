"""Загвар сургаж, маргаашийн 24 цагийн ачааллыг таамаглаад дашбоардын JSON-уудыг гаргана.

  python src/forecast.py                  # маргаашийн таамаг
  python src/forecast.py --target 2026-11-22
"""
import argparse
import datetime as dt
import json
import os
import warnings

import numpy as np
import pandas as pd
import sklearn
if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 0):   # Python 3.6 (Site5)
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
L1_LOSS = "absolute_error" if _SK >= (1, 0) else "least_absolute_deviation"

from config import DATA, DOCS_DATA, CAPACITY_MW, ub_now, parse_date, norm_hours
from features import FEATURES, build_frame
from learning import hourly_correction, peak_correction, blend_band
from supply import stress_flags, hourly_supply

BACKTEST_DAYS = 60


HOLDOUT_DAYS = 90


def make_model(leaves=31, min_leaf=30):
    return HistGradientBoostingRegressor(loss=L1_LOSS, max_iter=500, learning_rate=0.04,
                                         max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
                                         l2_regularization=1.0, random_state=0)


def _dates(df):
    return df["date"] if "date" in df.columns else pd.Series(df.index, index=df.index)


# --- Ачааллын өсөлт ба түүхэнд байгаагүй хүйтнийг тусгах ---------------------
# Модон (gradient boosting) загвар түүхэнд үзсэн хамгийн өндөр утгаас давж таамаглаж
# чаддаггүй, харин Монголын ачаалал ижил температурт жилд ~7% өсөж байна.
# Тиймээс өдрийн ОРГИЛЫГ «delta» горимоор таамаглана:
#   оргил ≈ ойрын түвшин (2 ба 7 хоногийн өмнөх оргилын дундаж)
#           + β × (маргаашийн оройн халаалтын градус − түвшний өдрүүдийн халаалтын градус)
#           + модон загварын засвар (гараг, баяр, салхи г.м.)
# Түвшин нь өсөлтийг, шугаман β нэмэлт хүйтнийг экстраполяц хийнэ.
# Шалгалт (2026-09-22…11-19): оргилын алдаа ±35 → ±18.7 МВт; −38 °C-ийн туршилтад
# 1,786 МВт (хуучин загвар түүхэн дээд 1,774-өөс давж чадахгүй байсан).
# Цагийн муруйд (оргил ЦАГ) энгийн горим хамгийн нарийвчлалтай.
HOURLY_MODE = os.environ.get("PFC_HOURLY_MODE", "plain")
DAILY_MODE = os.environ.get("PFC_DAILY_MODE", "delta")


def _level(df, target):
    cols = ["prev_day_peak", "peak_lag7"] if target == "peak" else ["lag48", "lag168"]
    return df[cols].mean(axis=1).values.astype(float)


def _hdh(x, base=18.0):
    return np.clip(base - np.asarray(x, dtype=float), 0, None)


def _delta_x(df):
    """Маргаашийн халаалтын градус − түвшний өдрүүдийн (2 ба 7 хоногийн өмнөх) халаалтын градус."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # хоёулаа хоосон мөрөнд
        ref = np.nanmean(np.column_stack([_hdh(df["prev_day_temp_mean"]), _hdh(df["temp_lag7_mean"])]), axis=1)
    return _hdh(df["temp_evening"]) - ref


def _delta_base(df, y):
    """оргил ≈ түвшин + β·Δ(халаалтын градус). β-г шугаман регрессээр олно (МВт/°C).
    Шугаман тул түүхэнд байгаагүй хүйтэн болон өсөлтөд экстраполяц хийнэ."""
    lev, dx = _level(df, "peak"), _delta_x(df)
    ok = np.isfinite(lev) & np.isfinite(dx) & np.isfinite(y)
    beta = float(np.dot(dx[ok], y[ok] - lev[ok]) / max(np.dot(dx[ok], dx[ok]), 1e-9))

    def base(x):
        lv, d = _level(x, "peak"), _delta_x(x)
        return lv + beta * np.nan_to_num(d)
    return base, beta


def fit_model(df, feats, target, mode, kw):
    """Сургасан загварын predict(df) функцийг буцаана."""
    y = df[target].values.astype(float)
    gb = make_model(**kw)
    if mode == "delta":
        base_fn, beta = _delta_base(df, y)
        r = y - base_fn(df)
        ok = np.isfinite(r)
        gb.fit(df.loc[ok, feats], r[ok])
        return lambda x: base_fn(x) + gb.predict(x[feats])
    if mode == "ratio":
        lev = _level(df, target)
        ok = np.isfinite(lev)
        gb.fit(df.loc[ok, feats], y[ok] / lev[ok])
        return lambda x: gb.predict(x[feats]) * _level(x, target)
    gb.fit(df[feats], y)
    return lambda x: gb.predict(x[feats])


def fit_with_intervals(train, test, feats, target, mode="plain", **kw):
    """P50 загвар + P10/P90 хүрээ. Хүрээг сүүлийн HOLDOUT_DAYS өдрийн загварт
    үзээгүй алдааны квантилаас (харьцангуй %) тооцно — квантил GBM-ээс бодитой."""
    train = train.dropna(subset=[target])
    if "stressed" in train.columns:   # хязгаарлалт байж болзошгүй үеийг сургалтаас хасна
        train = train[~train["stressed"].astype(bool)]
    d = _dates(train)
    cut = d.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    a, b = train[d <= cut], train[d > cut]
    rel = None
    if len(a) > 100 and len(b) > 10:
        f0 = fit_model(a, feats, target, mode, kw)
        rel = (b[target] - f0(b)) / b[target]
    p50 = fit_model(train, feats, target, mode, kw)(test)
    # Сүүлийн үеийн системтэй хазайлтыг медиан алдаагаар засна
    lo, mid, hi = (rel.quantile([0.1, 0.5, 0.9]).values if rel is not None else (-0.03, 0.0, 0.03))
    return pd.DataFrame({"p10": p50 * (1 + lo), "p50": p50 * (1 + mid), "p90": p50 * (1 + hi)},
                        index=test.index)


def fit_predict(train, test):
    return fit_with_intervals(train, test, FEATURES, "total_mw", mode=HOURLY_MODE)


DAILY_FEATURES = ["dow", "doy_sin", "doy_cos", "is_weekend", "is_holiday", "near_holiday",
                  "temp_min_day", "temp_mean_day", "temp_evening", "temp_mean72_eve",
                  "wind_mean", "prev_day_peak", "prev_day_mean", "prev_day_temp_mean", "peak_lag7", "temp_lag7_mean"]


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
    d["temp_lag7_mean"] = d["temp_mean_day"].reindex(d.index - pd.Timedelta(days=7)).values
    if "stressed" in frame.columns:
        ev = frame[frame["hour"].between(17, 22)].groupby("date")["stressed"].any()
        d["stressed"] = ev.reindex(d.index).fillna(False).astype(bool)
    return d


def fit_predict_daily(train_d, test_d):
    return fit_with_intervals(train_d, test_d, DAILY_FEATURES, "peak", mode=DAILY_MODE, leaves=15, min_leaf=10)


def peak_of(df, col):
    i = df[col].idxmax()
    return int(df.loc[i, "hour"]), float(df.loc[i, col])


RECONCILE_WIDTH = 4   # цаг; 60 өдрийн шалгалтаар 17–22 цагийн алдаа 30.3 → 21.9 МВт


def reconcile_curve(df, peak_p50, width=RECONCILE_WIDTH):
    """Цагийн муруйн оргилыг өдрийн оргилын загвартай тааруулна.
    Цагийн загвар оргилыг дунджаар ~2%-иар бага гаргадаг тул оргил цагийн утгыг өдрийн
    загварын P50 болгож, ойр цагуудыг Гауссын жингээр (exp(-(Δh/width)²)) зөөлөн өргөнө.
    Өглөөний цагууд бараг өөрчлөгдөхгүй."""
    ph, pv = peak_of(df, "p50")
    k = peak_p50 / pv if pv else 1.0
    f = 1 + (k - 1) * np.exp(-((df["hour"].values - ph) / float(width)) ** 2)
    return df[["p10", "p50", "p90"]].mul(f, axis=0)


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

    te["curve_p50"] = te["p50"]
    for d, g in te.groupby("date"):
        te.loc[g.index, ["p10", "p50", "p90"]] = reconcile_curve(g, float(dp.loc[d, "p50"])).values
    rows = []
    for d, g in te.groupby("date"):
        if len(g) < 24:
            continue
        ah, av = peak_of(g, "total_mw")
        ph, pv = peak_of(g.assign(p50=g["curve_p50"]), "p50")
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

    load = norm_hours(pd.read_csv(DATA / "load_hourly.csv", dtype={"date": str}))
    load = load[pd.to_datetime(load["date"]).dt.date < target]
    weather = pd.read_csv(DATA / "weather_hourly.csv")
    fc_path = DATA / "weather_forecast.csv"
    if fc_path.exists():
        weather = pd.concat([weather, pd.read_csv(fc_path)]).drop_duplicates("time", keep="first")

    tgt = pd.DataFrame({"date": target.isoformat(), "hour": range(1, 25), "total_mw": np.nan})
    frame = build_frame(pd.concat([load, tgt], ignore_index=True), weather)
    # Хязгаарлалт байж болзошгүй цагууд (импорт хязгаартаа + дулааны станц дээд хэмжээндээ)
    st = load.assign(stressed=stress_flags(load).values)
    st_key = set(zip(pd.to_datetime(st.loc[st["stressed"], "date"]), st.loc[st["stressed"], "hour"]))
    frame["stressed"] = [(d, h) in st_key for d, h in zip(frame["date"], frame["hour"])]
    print("Хязгаарлалт байж болзошгүй: {} цаг, {} өдөр — сургалтаас хасав".format(
        len(st_key), len({d for d, _ in st_key})))

    train = frame[(frame["date"].dt.date < target)].dropna(subset=["total_mw"])
    test = frame[frame["date"].dt.date == target].copy()
    if test["temp"].isna().all():
        raise SystemExit(f"{target}-ний температурын таамаг алга. collect_weather.py-г ажиллуулна уу.")
    test[["p10", "p50", "p90"]] = fit_predict(train, test)

    # --- Бодит таамгийн алдаанаас суралцах (зөвхөн энэ шинэ таамагт хэрэглэнэ) ---
    log_p, plog_p = DATA / "forecast_log.csv", DATA / "peak_log.csv"
    old_log = pd.read_csv(log_p, dtype={"target_date": str}) if log_p.exists() else None
    old_plog = pd.read_csv(plog_p, dtype={"target_date": str}) if plog_p.exists() else None
    test["raw_p50"] = test["p50"]
    hcorr, h_n = hourly_correction(old_log, load, target)
    if hcorr:
        f = 1 + test["hour"].map(hcorr).fillna(0.0)
        test[["p10", "p50", "p90"]] = test[["p10", "p50", "p90"]].mul(f, axis=0)

    ph, _ = peak_of(test, "p50")                       # оргил цаг — цагийн муруйгаас
    dd = daily_frame(frame[frame["date"].dt.date <= target])
    tdate = pd.Timestamp(target)
    pk = fit_predict_daily(dd[dd.index < tdate], dd.loc[[tdate]]).iloc[0].copy()   # оргил МВт — өдрийн загвараас
    pk_raw = float(pk["p50"])
    pcorr, pband, p_n = peak_correction(old_plog, load, target)
    pk[["p10", "p50", "p90"]] = pk[["p10", "p50", "p90"]] * (1 + pcorr)
    pk["p10"], pk["p90"] = blend_band(pk["p50"], pk["p10"], pk["p90"], pband, p_n)
    # Цагийн муруйг оргилын таамагтай нийцүүлэх (графикт 1,657 биш 1,724 гэх мэт нэг тоо харагдана)
    test["curve_p50"] = test["p50"]
    test[["p10", "p50", "p90"]] = reconcile_curve(test, float(pk["p50"])).values
    # --- Нийлүүлэлт: салхи/нарны таамаг ба цаг тутмын боломжит чадал ---
    sup_rows, sup_comp = None, None
    try:
        ws_p, wsf_p = DATA / "weather_sites_hourly.csv", DATA / "weather_sites_forecast.csv"
        wsites = None
        if ws_p.exists():
            wsites = pd.read_csv(ws_p)
            if wsf_p.exists():
                wsites = pd.concat([wsites, pd.read_csv(wsf_p)]).drop_duplicates(["time", "site"], keep="first")
        sup_rows, sup_comp = hourly_supply(load, wsites, target)
        sup_by_h = {r["hour"]: r for r in sup_rows}
        test["supply"] = test["hour"].map(lambda h: sup_by_h[h]["supply"])
    except Exception as e:  # noqa: BLE001
        print("Нийлүүлэлтийн тооцоо амжилтгүй:", e)

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
        "supply": None if sup_rows is None else {
            "components": {k: round(v, 1) for k, v in sup_comp.items()},
            "hours": sup_rows,
            "peak_margin": round(sup_by_h[ph]["supply"] - float(pk["p90"]), 1),
            "risk_hours": [r["hour"] for r in sup_rows if r["supply"] < float(test.loc[test["hour"] == r["hour"], "p90"].iloc[0])],
        },
        "learning": {"peak_days": p_n, "peak_correction_pct": round(pcorr * 100, 2),
                     "hourly_days": h_n,
                     "hourly_correction_pct": {str(k): round(v * 100, 2) for k, v in sorted(hcorr.items())}},
    }

    # Таамгийн түүх — өчигдрийн таамгийг бодит гүйцэтгэлтэй харьцуулах
    log_cols = ["hour", "p10", "p50", "p90", "raw_p50", "curve_p50"] + (["supply"] if "supply" in test.columns else [])
    new_log = test[log_cols].assign(target_date=target.isoformat(),
                                                                    issued_at=fc["issued_at"]).round(1)
    log = pd.concat([old_log, new_log], sort=False) if old_log is not None else new_log
    log = log.drop_duplicates(["target_date", "hour"], keep="last").sort_values(["target_date", "hour"])
    log.to_csv(log_p, index=False)
    new_pl = pd.DataFrame([{"target_date": target.isoformat(), "issued_at": fc["issued_at"], "hour": ph,
                            "raw_p50": round(pk_raw, 1), "p10": fc["peak"]["p10"], "p50": fc["peak"]["p50"],
                            "p90": fc["peak"]["p90"]}])
    plog = pd.concat([old_plog, new_pl], sort=False) if old_plog is not None else new_pl
    plog.drop_duplicates("target_date", keep="last").sort_values("target_date").to_csv(plog_p, index=False)

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
    print("Суралцах засвар: оргил {:+.2f}% ({} өдөр), цагийн {} өдөр".format(pcorr * 100, p_n, h_n))
    print(f"{target} оргил: {fc['peak']['label']} цагт ~{fc['peak']['p50']} МВт "
          f"(P10–P90: {fc['peak']['p10']}–{fc['peak']['p90']})")


if __name__ == "__main__":
    main()
