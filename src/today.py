"""Өнөөдрийн таамаг ба гүйцэтгэл → docs/data/today.json (цаг тутам ажиллана).

Таамаг нь өмнөх өдрийн 02:00 цагт гаргаж data/forecast_log.csv-д хадгалсан утга.
Гүйцэтгэл нь data/load_hourly.csv-д байгаа өнөөдрийн цагуудын бодит ачаалал.
"""
import datetime as dt
import json

import pandas as pd

from config import DATA, DOCS_DATA, CAPACITY_MW, ub_now


def main():
    now = ub_now()
    today = now.date().isoformat()

    load = pd.read_csv(DATA / "load_hourly.csv", dtype={"date": str})
    act = load[load["date"] == today].set_index("hour")["total_mw"]

    log_p = DATA / "forecast_log.csv"
    fc = pd.DataFrame()
    if log_p.exists():
        log = pd.read_csv(log_p, dtype={"target_date": str})
        fc = log[log["target_date"] == today].set_index("hour")
    issued = None if fc.empty else str(fc["issued_at"].iloc[0])

    hours = []
    for h in range(1, 25):
        a = act.get(h)
        row = {"hour": h, "label": f"{h:02d}:00", "actual": None if a is None or pd.isna(a) else float(a)}
        for k in ("p10", "p50", "p90"):
            row[k] = None if fc.empty or h not in fc.index else float(fc.loc[h, k])
        hours.append(row)

    done = [r for r in hours if r["actual"] is not None]
    with_fc = [r for r in done if r["p50"] is not None]
    summary = {}
    if done:
        pk = max(done, key=lambda r: r["actual"])
        last = done[-1]
        summary.update(actual_peak=pk["actual"], actual_peak_label=pk["label"],
                       last_label=last["label"], last_actual=last["actual"], last_p50=last["p50"])
    if not fc.empty:
        fpk = max((r for r in hours if r["p50"] is not None), key=lambda r: r["p50"])
        summary.update(fc_peak=fpk["p50"], fc_peak_label=fpk["label"], fc_peak_p90=fpk["p90"])
    if with_fc:
        mape = sum(abs(r["actual"] - r["p50"]) / r["actual"] for r in with_fc) / len(with_fc) * 100
        summary.update(mape_so_far=round(mape, 2), hours_compared=len(with_fc))

    out = {"date": today, "updated_at": now.isoformat(timespec="minutes"),
           "forecast_issued_at": issued, "capacity_mw": CAPACITY_MW,
           "summary": summary, "hours": hours}
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / "today.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{today}: {len(done)} цагийн гүйцэтгэл, таамаг {'бий' if issued else 'алга'}")


if __name__ == "__main__":
    main()
