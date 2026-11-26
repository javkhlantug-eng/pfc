"""Томоохон баярын (Шинэ жил, Цагаан сар, Наадам) засвар.

Эдгээр өдрүүд жилд хэдхэн удаа тохиодог тул загвар ачааллын бууралтыг (10–17%) бүрэн
сурч чаддаггүй, ихэвчлэн 50–100 МВт-аар өндөр таамагладаг. Иймд:
  1) тухайн өдрийг «ердийн өдөр» гэж загвараар таамаглаад,
  2) өмнөх жилүүдэд яг тэр баяр ижил гарагийн ердийн өдрүүдээс (±7, ±14 хоног) хэдэн хувиар
     бага байсныг (медиан, n/(n+1) жинтэй) үржүүлнэ.
Өмнөх 2 жилийн шалгалтаар томоохон баярын оргилын алдаа ~57 → ~39 МВт болсон.
Жижиг баяруудын (11-26, 3-8 г.м.) нөлөө 0–3% тул загвар өөрөө (is_holiday) тооцно.
"""
import numpy as np
import pandas as pd

from config import DATA
from features import holiday_name, MAJOR


def _table():
    p = DATA / "holidays.csv"
    return dict(pd.read_csv(p)[["date", "name"]].values) if p.exists() else {}


def major_factor(load, target):
    """(оргилын засвар, дундаж/цагийн засвар, өмнөх тохиолдлын тоо, нэр) эсвэл None."""
    tab = _table()
    name = holiday_name(target, tab)
    if name not in MAJOR:
        return None
    ld = load.copy()
    ld["date"] = ld["date"].astype(str)
    cnt = ld.groupby("date")["hour"].count()
    D = ld.groupby("date")["total_mw"].agg(["max", "mean"])[cnt >= 24]
    D = D[D["mean"] > 300]                       # алдаатай/дутуу өдрийг хасна
    t0 = pd.Timestamp(target)
    rp, rm = [], []
    for d in D.index:
        t = pd.Timestamp(d)
        if t >= t0 - pd.Timedelta(days=20) or holiday_name(t, tab) != name:
            continue
        ref = [(t + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (-14, -7, 7, 14)]
        ref = [r for r in ref if r in D.index and not holiday_name(r, tab)]
        if len(ref) < 2:
            continue
        rp.append(D.loc[d, "max"] / D.loc[ref, "max"].mean() - 1)
        rm.append(D.loc[d, "mean"] / D.loc[ref, "mean"].mean() - 1)
    n = len(rp)
    if n == 0:
        return None
    w = n / (n + 1.0)
    return float(np.median(rp) * w), float(np.median(rm) * w), n, name
