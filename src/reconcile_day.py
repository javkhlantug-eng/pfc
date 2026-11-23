"""Нэг өдрийн хадгалсан цагийн таамгийг тухайн өдрийн зарласан оргилын таамагтай
(peak_log.csv) нийцүүлнэ. Хэрэглэгчийн шийдвэрээр зөвхөн 2026-11-23-нд хэрэглэсэн.
Анхны муруй curve_p50 баганад хадгалагдана. Ашиглалт: python reconcile_day.py 2026-11-23
--raw: суралцах засвараас өмнөх анхны загварын утгыг (raw_p50) суурь болгоно (2026-11-23-нд хэрэглэсэн)."""
import sys

import numpy as np
import pandas as pd

from config import DATA

WIDTH = 4


def main(day, use_raw=False):
    fp, pp = DATA / "forecast_log.csv", DATA / "peak_log.csv"
    fl = pd.read_csv(fp, dtype={"target_date": str})
    pl = pd.read_csv(pp, dtype={"target_date": str})
    if "curve_p50" not in fl.columns:
        fl["curve_p50"] = np.nan
    g = fl[fl["target_date"] == day]
    m = pl[pl["target_date"] == day]
    if g.empty or m.empty:
        print("өөрчлөх зүйлгүй")
        return
    if use_raw:
        # Одоогийн утгаас анхны муруйг сэргээж (p/curve харьцаагаар), raw_p50 руу шилжүүлнэ
        cur = g["curve_p50"].where(g["curve_p50"].notna(), g["p50"])
        base = g["raw_p50"].where(g["raw_p50"].notna(), cur)
        k0 = cur / g["p50"]
        for c in ["p10", "p50", "p90"]:
            g[c] = (g[c] * k0 * base / cur).values
    elif g["curve_p50"].notna().any():
        print("өөрчлөх зүйлгүй")
        return
    pk = m.iloc[-1]
    ph, target = int(pk["hour"]), float(pk["p50"])
    v = float(g.loc[g["hour"] == ph, "p50"].iloc[0])
    f = 1 + (target / v - 1) * np.exp(-((g["hour"].values - ph) / float(WIDTH)) ** 2)
    if fl.loc[g.index, "curve_p50"].isna().all():
        fl.loc[g.index, "curve_p50"] = g["p50"].values
    for c in ["p10", "p50", "p90"]:
        fl.loc[g.index, c] = (g[c].values * f).round(1)
    fl.to_csv(fp, index=False)
    print("{}: {:02d}:00 {:.1f} → {:.1f} МВт".format(day, ph, v, target))


if __name__ == "__main__":
    main(sys.argv[1], "--raw" in sys.argv)
