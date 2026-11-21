"""dis.energy.mn-ээс цагийн ачааллыг татаж data/load_hourly.csv-д нэмнэ.

Жишээ:
  python src/collect_load.py --start 2024-01-01          # түүхийг бүхэлд нь
  python src/collect_load.py --days-back 3               # сүүлийн 3 хоногийг шинэчлэх
"""
import argparse
import datetime as dt
import sys
import time

import pandas as pd
import requests

from config import DATA, LOAD_URL, LOAD_START, REQUEST_DELAY_SEC, LOAD_COLS, ub_now, parse_date

OUT = DATA / "load_hourly.csv"
HEADERS = {"User-Agent": "pfc-peak-forecast/1.0 (research)", "Accept": "application/json"}


def fetch_day(day, session):
    r = session.get(LOAD_URL, params={"date": day.isoformat()}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for h in (js.get("data") or {}).get("hours") or []:
        total = h.get("total_consumption_mw")
        if total is None:
            continue
        # API өдрийн сүүлийн цагийг (24:00) "hour": 0, "time": "00:00" гэж өгдөг → 24 болгоно
        row = {"date": day.isoformat(), "hour": int(h["hour"]) or 24, "total_mw": float(total)}
        summ = h.get("station_summary") or {}
        if isinstance(summ, dict):
            for c in LOAD_COLS:
                row[c] = summ.get(c)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD (өгөгдмөл: өнөөдөр)")
    ap.add_argument("--days-back", type=int, help="сүүлийн N хоногийг дахин татах")
    ap.add_argument("--skip-existing", action="store_true",
                    help="24 цаг бүрэн байгаа өдрүүдийг алгасах")
    a = ap.parse_args()

    today = ub_now().date()
    end = parse_date(a.end) if a.end else today
    if a.days_back:
        start = end - dt.timedelta(days=a.days_back)
    else:
        start = parse_date(a.start or LOAD_START)

    old = pd.read_csv(OUT, dtype={"date": str}) if OUT.exists() else pd.DataFrame()
    complete = set()
    if a.skip_existing and not old.empty:
        cnt = old.groupby("date")["hour"].count()
        complete = set(cnt[cnt >= 24].index)

    def save(old, rows):
        new = pd.DataFrame(rows)
        df = pd.concat([old, new], ignore_index=True) if not new.empty else old
        if df.empty:
            return df
        df = df.copy()
        df.loc[df["hour"] == 0, "hour"] = 24          # хуучин файлд хадгалагдсан 0 → 24
        df = (df.drop_duplicates(["date", "hour"], keep="last")
                .sort_values(["date", "hour"]).reset_index(drop=True))
        DATA.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False)
        return df

    s = requests.Session()
    new_rows, errors, n = [], 0, 0
    d = start
    while d <= end:
        n += 1
        if n % 60 == 0:          # урт татлагын явцад үе үе хадгална
            save(old, new_rows)
        if d.isoformat() not in complete:
            for attempt in range(3):
                try:
                    new_rows += fetch_day(d, s)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        errors += 1
                        print(f"[алдаа] {d}: {e}", file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
            time.sleep(REQUEST_DELAY_SEC)
        d += dt.timedelta(days=1)

    new = pd.DataFrame(new_rows)
    df = save(old, new_rows)
    if df.empty:
        print("Өгөгдөл ирсэнгүй.")
        sys.exit(1)
    print(f"{len(new)} мөр нэмэгдлээ, нийт {len(df)} мөр ({df['date'].min()} – {df['date'].max()}), алдаа: {errors}")
    if errors and new.empty:
        sys.exit(1)


if __name__ == "__main__":
    main()
