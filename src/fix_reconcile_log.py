"""Нэг удаагийн засвар: хадгалсан цагийн таамгийг (forecast_log.csv) тухайн үед зарласан
оргилын таамагтай (peak_log.csv) нийцүүлнэ.

Өмнө нь графикийн 19:00-ийн утга (цагийн загвар, жишээ нь 1,657) зарласан оргилоос
(өдрийн загвар, 1,724) бага харагддаг байсан. Зарласан таамаг нь «19:00-д 1,724» тул
цагийн муруйг түүнд forecast.reconcile_curve-тэй ижил аргаар тааруулна.
Анхны муруй curve_p50 баганад хадгалагдана. Дахин ажиллуулахад юу ч өөрчлөхгүй.
"""
import numpy as np
import pandas as pd

from config import DATA

WIDTH = 4


def main():
    fp, pp = DATA / "forecast_log.csv", DATA / "peak_log.csv"
    if not (fp.exists() and pp.exists()):
        print("лог алга")
        return
    fl = pd.read_csv(fp, dtype={"target_date": str})
    pl = pd.read_csv(pp, dtype={"target_date": str})
    if "curve_p50" not in fl.columns:
        fl["curve_p50"] = np.nan
    n = 0
    for (d, iss), g in fl.groupby(["target_date", "issued_at"]):
        if g["curve_p50"].notna().any():
            continue
        m = pl[(pl["target_date"] == d) & (pl["issued_at"] == iss)]
        if m.empty:
            m = pl[pl["target_date"] == d]
        if m.empty:
            continue
        pk = m.iloc[-1]
        ph, target = int(pk["hour"]), float(pk["p50"])
        row = g[g["hour"] == ph]
        if row.empty or pd.isna(row["p50"].iloc[0]):
            continue
        k = target / float(row["p50"].iloc[0])
        f = 1 + (k - 1) * np.exp(-((g["hour"].values - ph) / float(WIDTH)) ** 2)
        fl.loc[g.index, "curve_p50"] = g["p50"].values
        for c in ["p10", "p50", "p90"]:
            fl.loc[g.index, c] = (g[c].values * f).round(1)
        n += 1
        print("{} ({}): {:02d}:00 {:.1f} → {:.1f} МВт".format(d, iss, ph, float(row["p50"].iloc[0]), target))
    fl.to_csv(fp, index=False)
    print("Засагдсан өдөр:", n)


if __name__ == "__main__":
    main()
