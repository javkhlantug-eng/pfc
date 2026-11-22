"""Температурын таамгийн алдааны шинжилгээ (exp_tempfc.py-ийн өгөгдөл дээр).

Ашиглалт: python exp_tempfc_analyze.py <tempfc csv хавтас> <weather_hourly.csv>
"Бодит" = weather_hourly.csv (ERA5) — ачааллын загвар үүн дээр сурсан тул энэ нь зөв жишиг.
Оргилд хамгийн чухал нь оройн (17–22 цаг) дундаж температур.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

src, wpath = Path(sys.argv[1]), Path(sys.argv[2])
act = pd.read_csv(wpath, parse_dates=["time"]).set_index("time")["temperature_2m"].rename("act")
act = act[~act.index.duplicated()]

fc = {}
for p in sorted(src.glob("tempfc_*.csv")):
    m = p.stem.replace("tempfc_", "")
    d = pd.read_csv(p, parse_dates=["time"]).set_index("time")
    d = d[~d.index.duplicated()]
    fc[m] = d


def evening(s):
    s = s[(s.index.hour >= 17) & (s.index.hour <= 22)]
    g = s.groupby(s.index.date)
    return g.mean()[g.count() == 6]


A_eve = evening(act)
rows, eve_err = [], {}
for m, d in fc.items():
    for lead in ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]:
        if lead not in d.columns:
            continue
        f = d[lead].dropna()
        e = (f - act.reindex(f.index)).dropna()
        fe = evening(f)
        ee = (fe - A_eve.reindex(fe.index)).dropna()
        if len(ee) < 60:
            continue
        cold = ee[A_eve.reindex(ee.index) < -15]
        tag = "D1" if lead.endswith("1") else "D2"
        eve_err[(m, tag)] = ee
        rows.append(dict(model=m, lead=tag, days=len(ee), first=str(ee.index.min()),
                         hour_MAE=e.abs().mean(), eve_MAE=ee.abs().mean(), eve_bias=ee.mean(),
                         eve_RMSE=np.sqrt((ee ** 2).mean()), cold_MAE=cold.abs().mean() if len(cold) else np.nan,
                         cold_bias=cold.mean() if len(cold) else np.nan, cold_n=len(cold)))

R = pd.DataFrame(rows).sort_values(["lead", "eve_MAE"])
pd.set_option("display.width", 200)
print("=== Оройн (17–22) дундаж температурын алдаа, °C. cold = бодит < −15°C өдрүүд ===")
print(R.round(2).to_string(index=False))

# Нийтлэг хугацаанд шударга харьцуулалт + хослол + хазайлтын засвар (зөвхөн D1)
d1 = {m: e for (m, t), e in eve_err.items() if t == "D1"}
common = sorted(set.intersection(*[set(e.index) for e in d1.values()])) if d1 else []
print("\nНийтлэг өдрүүд (бүх загвар D1):", len(common), common[0] if common else "", "–", common[-1] if common else "")
if len(common) > 60:
    E = pd.DataFrame({m: e.reindex(common) for m, e in d1.items()})
    F = E.add(A_eve.reindex(common), axis=0)          # загварын оройн таамаг
    a = A_eve.reindex(common)
    res = {m: E[m].abs().mean() for m in E}
    best = sorted(res, key=res.get)
    for k in (2, 3, 4):
        ens = F[best[:k]].mean(axis=1)
        res["ensemble_top{}".format(k)] = (ens - a).abs().mean()
    res["ensemble_all"] = (F.mean(axis=1) - a).abs().mean()

    # Хазайлтын засвар: өмнөх 30 өдрийн дундаж алдааг хасна (зөвхөн өнгөрсөн мэдээлэл — шударга)
    def debias(f, win=30):
        err = f - a
        return f - err.shift(1).rolling(win, min_periods=10).mean().fillna(0)
    for m in best[:3] + ["best_match"]:
        if m in F:
            res[m + "+bias30"] = (debias(F[m]) - a).abs().mean()
    ens3 = F[best[:3]].mean(axis=1)
    res["ensemble_top3+bias30"] = (debias(ens3) - a).abs().mean()
    # Температураас хамаарсан засвар (хүйтэн үед хазайлт өөр байж болно): өмнөх 60 өдрийн шугаман регресс
    def debias_lin(f, win=60):
        out = f.copy()
        for i in range(len(f)):
            lo = max(0, i - win)
            if i - lo < 20:
                continue
            x, y = f.iloc[lo:i].values, a.iloc[lo:i].values
            k, b = np.polyfit(x, y, 1)
            out.iloc[i] = k * f.iloc[i] + b
        return out
    res["ensemble_top3+lin60"] = (debias_lin(ens3) - a).abs().mean()
    if "best_match" in F:
        res["best_match+lin60"] = (debias_lin(F["best_match"]) - a).abs().mean()
    S = pd.Series(res).sort_values()
    print("\n=== Нийтлэг өдрүүд дээрх оройн MAE, °C (бага нь сайн) ===")
    print(S.round(3).to_string())
    print("\nОдоо ашиглаж буй (best_match) →", round(res.get("best_match", np.nan), 3),
          "°C;  хамгийн сайн:", S.index[0], round(S.iloc[0], 3), "°C")
    # Улирлаар
    mon = pd.Series(pd.to_datetime(common).month, index=common)
    win = mon.isin([11, 12, 1, 2, 3])
    print("\nӨвөл (11–3 сар) оройн MAE:")
    for m in list(dict.fromkeys(best[:4] + ["best_match"])):
        if m in E:
            print("  {:24s} {:.2f}".format(m, E[m][win.values].abs().mean()))
    print("  {:24s} {:.2f}".format("ensemble_top3", (ens3 - a)[win.values].abs().mean()))
    print("  {:24s} {:.2f}".format("ensemble_top3+bias30", (debias(ens3) - a)[win.values].abs().mean()))
