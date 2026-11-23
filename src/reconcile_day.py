"""Нэг өдрийн хадгалсан цагийн таамгийг тухайн өдрийн зарласан оргилын таамагтай
(peak_log.csv) нийцүүлнэ. Хэрэглэгчийн шийдвэрээр зөвхөн 2026-11-23-нд хэрэглэсэн.
Анхны муруй curve_p50 баганад хадгалагдана. Ашиглалт: python reconcile_day.py 2026-11-23"""
import sys

import numpy as np
import pandas as pd

from config import DATA

WIDTH = 4


def main(day):
    fp, pp = DATA / "forecast_log.csv", DATA / "peak_log.csv"
    fl = pd.read_csv(fp, dtype={"target_date": str})
    pl = pd.read_csv(pp, dtype={"target_date": str})
    if "curve_p50" not in fl.columns:
        fl["curve_p50"] = np.nan
    g = fl[fl["target_date"] == day]
    m = pl[pl["target_date"] == day]
    if g.empty or m.empty or g["curve_p50"].notna().any():
        print("өөрчлөх зүйлгүй")
        return
    pk = m.iloc[-1]
    ph, target = int(pk["hour"]), float(pk["p50"])
    v = float(g.loc[g["hour"] == ph, "p50"].iloc[0])
    f = 1 + (target / v - 1) * np.exp(-((g["hour"].values - ph) / float(WIDTH)) ** 2)
    fl.loc[g.index, "curve_p50"] = g["p50"].values
    for c in ["p10", "p50", "p90"]:
        fl.loc[g.index, c] = (g[c].values * f).round(1)
    fl.to_csv(fp, index=False)
    print("{}: {:02d}:00 {:.1f} → {:.1f} МВт".format(day, ph, v, target))


if __name__ == "__main__":
    main(sys.argv[1])
