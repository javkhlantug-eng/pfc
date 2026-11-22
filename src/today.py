"""Өнөөдрийн таамаг ба гүйцэтгэл → docs/data/today.json (цаг тутам ажиллана).

Шинэчилсэн таамаг (nowcast): сүүлийн 3 цагт бодит ачаалал өчигдрийн таамгаас хэдэн %
зөрж байгааг (r) тооцоод, үлдсэн цагуудын таамгийг тэр зөрүүгээр засна. Зөрүүний нөлөө
цаг холдох тусам буурна: жин = 0.5^(Δцаг / 8). Өчигдрийн анхны таамаг өөрчлөгдөхгүй.

Таамаг нь өмнөх өдрийн 02:00 цагт гаргаж data/forecast_log.csv-д хадгалсан утга.
Гүйцэтгэл нь data/load_hourly.csv-д байгаа өнөөдрийн цагуудын бодит ачаалал.
"""
import datetime as dt
import json

import pandas as pd

HALF_LIFE_H = 8

from config import DATA, DOCS_DATA, CAPACITY_MW, ub_now, norm_hours, likely_curtailed


def main():
    now = ub_now()
    today = now.date().isoformat()

    load = norm_hours(pd.read_csv(DATA / "load_hourly.csv", dtype={"date": str}))
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

    for r in hours:
        r["curt"] = bool(likely_curtailed(r["actual"], r["p10"]))
    done = [r for r in hours if r["actual"] is not None]
    # Хязгаарлалт байж болзошгүй цагуудыг алдаа болон шинэчлэлийн тооцоонд оруулахгүй
    with_fc = [r for r in done if r["p50"] is not None and not r["curt"]]
    summary = {}
    if done:
        pk = max(done, key=lambda r: r["actual"])
        last = done[-1]
        summary.update(actual_peak=pk["actual"], actual_peak_label=pk["label"],
                       last_label=last["label"], last_actual=last["actual"], last_p50=last["p50"])
    n_curt = sum(1 for r in hours if r["curt"])
    if n_curt:
        summary["curtailed_hours"] = n_curt
    if not fc.empty:
        fpk = max((r for r in hours if r["p50"] is not None), key=lambda r: r["p50"])
        summary.update(fc_peak=fpk["p50"], fc_peak_label=fpk["label"], fc_peak_p90=fpk["p90"])
    if with_fc:
        mape = sum(abs(r["actual"] - r["p50"]) / r["actual"] for r in with_fc) / len(with_fc) * 100
        summary.update(mape_so_far=round(mape, 2), hours_compared=len(with_fc))

    # --- Шинэчилсэн таамаг (nowcast) ---
    if with_fc:
        last3 = with_fc[-3:]
        r = sum(x["actual"] / x["p50"] for x in last3) / len(last3)
        last_h = done[-1]["hour"]
        w = lambda h: 0.5 ** ((h - last_h) / HALF_LIFE_H)
        for row in hours:
            if row["hour"] == last_h:
                row["upd"] = row["actual"]          # шугам сүүлийн бодит цэгээс эхэлнэ
            elif row["hour"] > last_h and row["p50"] is not None:
                row["upd"] = round(row["p50"] * (1 + (r - 1) * w(row["hour"])), 1)
        # Оргил: өчигдрийн өдрийн оргилын таамаг (байвал), эс бөгөөс муруйн дээд цэг
        base_pk, pk_h = summary.get("fc_peak"), int(summary.get("fc_peak_label", "19:00")[:2])
        plog_p = DATA / "peak_log.csv"
        if plog_p.exists():
            pl = pd.read_csv(plog_p, dtype={"target_date": str})
            pl = pl[pl["target_date"] == today]
            if not pl.empty:
                base_pk, pk_h = float(pl["p50"].iloc[0]), int(pl["hour"].iloc[0])
                summary.update(fc_peak=base_pk, fc_peak_label="{:02d}:00".format(pk_h),
                               fc_peak_p90=float(pl["p90"].iloc[0]))
        if base_pk is not None:
            if pk_h > last_h:
                now_pk = base_pk * (1 + (r - 1) * w(pk_h))
                summary.update(now_peak=round(now_pk, 1), now_peak_label="{:02d}:00".format(pk_h))
            else:   # таамагласан оргилын цаг өнгөрсөн
                summary.update(now_peak=summary.get("actual_peak"), now_peak_label=summary.get("actual_peak_label"))
        summary.update(now_dev_pct=round((r - 1) * 100, 2), now_basis_label=done[-1]["label"])

    # --- Шинэчилсэн оргилын түүх: цаг бүрийн шинэчлэлийг бүртгэнэ ---
    if summary.get("now_peak") is not None:
        log_nc(today, summary, now)
    write_history(load)

    out = {"date": today, "updated_at": now.isoformat(timespec="minutes"),
           "forecast_issued_at": issued, "capacity_mw": CAPACITY_MW,
           "summary": summary, "hours": hours}
    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA / "today.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{today}: {len(done)} цагийн гүйцэтгэл, таамаг {'бий' if issued else 'алга'}")


NC_LOG = "nowcast_log.csv"
HIST_DAYS = 14


def log_nc(today, S, now):
    """Тухайн цагийн гүйцэтгэлээр шинэчилсэн оргилыг data/nowcast_log.csv-д нэмнэ."""
    p = DATA / NC_LOG
    row = pd.DataFrame([{"date": today, "basis": S["now_basis_label"], "now_peak": S["now_peak"],
                         "now_peak_label": S["now_peak_label"], "fc_peak": S.get("fc_peak"),
                         "dev_pct": S.get("now_dev_pct"), "logged_at": now.isoformat(timespec="minutes")}])
    log = pd.concat([pd.read_csv(p, dtype={"date": str}), row], sort=False) if p.exists() else row
    log.drop_duplicates(["date", "basis"], keep="last").sort_values(["date", "basis"]).to_csv(p, index=False)


def write_history(load):
    """Сүүлийн 14 өдрийн шинэчлэлийн түүхийг бодит оргилтой хамт docs/data/nowcast_history.json болгоно."""
    p = DATA / NC_LOG
    if not p.exists():
        return
    log = pd.read_csv(p, dtype={"date": str})
    days = sorted(log["date"].unique())[-HIST_DAYS:]
    out = []
    for d in reversed(days):
        g = log[log["date"] == d]
        a = load[load["date"] == d]
        actual = None
        if not a.empty:
            i = a["total_mw"].idxmax()
            actual = {"mw": float(a.loc[i, "total_mw"]), "label": "{:02d}:00".format(int(a.loc[i, "hour"])),
                      "complete": bool(len(a) >= 24)}
        fc = g["fc_peak"].dropna()
        out.append({"date": d, "fc_peak": None if fc.empty else float(fc.iloc[0]), "actual": actual,
                    "cells": {r.basis: {"mw": float(r.now_peak), "label": r.now_peak_label} for r in g.itertuples()}})
    (DOCS_DATA / "nowcast_history.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
