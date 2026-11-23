"""fix_reconcile_log.py-ийн өөрчлөлтийг буцаана: 2026-11-24 хүртэлх (хуучин кодоор гаргасан)
таамгуудын цаг бүрийн утгыг анх гаргасан утга руу нь (curve_p50) сэргээнэ. Гаргасан таамгийг
дараа нь өөрчлөхгүй. Дахин ажиллуулахад юу ч өөрчлөхгүй."""
import pandas as pd

from config import DATA

LAST_OLD = "2026-11-24"


def main():
    fp = DATA / "forecast_log.csv"
    fl = pd.read_csv(fp, dtype={"target_date": str})
    if "curve_p50" not in fl.columns:
        print("сэргээх зүйлгүй")
        return
    m = (fl["target_date"] <= LAST_OLD) & fl["curve_p50"].notna() & (fl["curve_p50"] != fl["p50"])
    k = fl.loc[m, "curve_p50"] / fl.loc[m, "p50"]
    for c in ["p10", "p50", "p90"]:
        fl.loc[m, c] = (fl.loc[m, c] * k).round(1)
    fl.loc[fl["target_date"] <= LAST_OLD, "curve_p50"] = float("nan")
    fl.to_csv(fp, index=False)
    print("Сэргээсэн мөр:", int(m.sum()), sorted(fl.loc[m, "target_date"].unique()))


if __name__ == "__main__":
    main()
