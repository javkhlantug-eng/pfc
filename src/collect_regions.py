"""dashboard.energy.mn-ээс бүсүүдийн ачааллын агшныг татаж data/regions_hourly.csv-д нэмнэ.

Сайт бүсийн мэдээг зөвхөн хамгийн сүүлийн цагаар харуулдаг (түүх байхгүй) тул
цаг тутам ажиллуулж өөрсдөө түүх хуримтлуулна. Алдаа гарвал бусад ажлыг зогсоохгүй.
"""
import re
import sys

import pandas as pd
import requests

from config import DATA, ub_now

URL = "https://dashboard.energy.mn/departments/dispatch/overview.data"
REGIONS = ["Улаанбаатар", "Дархан-Сэлэнгэ", "Эрдэнэт-Булган", "Багануур-Чойр", "Говь"]
OUT = DATA / "regions_hourly.csv"


def main():
    now = ub_now()
    try:
        r = requests.get(URL, timeout=30, headers={"User-Agent": "pfc-peak-forecast/1.0 (research)"})
        r.raise_for_status()
        t = r.text
    except Exception as e:  # noqa: BLE001
        print("Бүсийн мэдээ татаж чадсангүй:", e)
        return
    mt = re.search(r'mapTime[^0-9]{0,10}([0-9]{2}:[0-9]{2})', t)
    if not mt:
        print("Бүсийн мэдээний цаг олдсонгүй")
        return
    hour = int(mt.group(1)[:2])
    # mapTime нь өнөөдрийн цаг; шөнийн 00:00 бол өчигдрийн 24:00
    date = now.date()
    if hour == 0:
        hour, date = 24, (now - pd.Timedelta(days=1)).date()
    rows = []
    for name in REGIONS:
        m = re.search(re.escape(name) + r'[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)', t)
        if m:
            rows.append({"date": date.isoformat(), "hour": hour, "region": name, "mw": float(m.group(1)),
                         "fetched_at": now.isoformat(timespec="minutes")})
    if not rows:
        print("Бүсийн утга олдсонгүй")
        return
    new = pd.DataFrame(rows)
    df = pd.concat([pd.read_csv(OUT, dtype={"date": str}), new], sort=False) if OUT.exists() else new
    df = df.drop_duplicates(["date", "hour", "region"], keep="last").sort_values(["date", "hour", "region"])
    df.to_csv(OUT, index=False)
    print("Бүс {} {:02d}:00 — {}".format(date, hour, ", ".join("{} {:.0f}".format(x["region"], x["mw"]) for x in rows)))


if __name__ == "__main__":
    main()
