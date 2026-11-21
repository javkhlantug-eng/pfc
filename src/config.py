"""Төслийн тохиргоо. Python 3.6+ дээр ажиллана (Site5 хостинг)."""
import datetime as _dt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("PFC_DATA", str(ROOT / "data")))
DOCS_DATA = Path(os.environ.get("PFC_DOCS_DATA", str(ROOT / "docs" / "data")))

TZ = "Asia/Ulaanbaatar"
UB = _dt.timezone(_dt.timedelta(hours=8))   # Монгол улс зуны цаг хэрэглэдэггүй


def ub_now():
    """Улаанбаатарын одоогийн цаг (серверийн цагийн бүсээс үл хамаарна)."""
    return _dt.datetime.now(UB)


def parse_date(s):
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()

# dis.energy.mn — Төвийн бүсийн эрчим хүчний систем (цагийн мэдээ)
LOAD_URL = "https://dis.energy.mn/generation-chart/history"
LOAD_START = "2024-01-01"          # өгөгдөл ойролцоогоор энэ үеэс эхэлдэг
REQUEST_DELAY_SEC = 1.0            # сервер рүү ачаалал багатай татах

# Цаг агаарын цэгүүд. Жин нь нийт ачаалалд оруулах нөлөө.
# Эхэндээ зөвхөн Улаанбаатар; Дархан, Эрдэнэтийг нэмж туршиж болно.
LOCATIONS = [
    {"name": "Улаанбаатар", "lat": 47.92, "lon": 106.92, "weight": 1.0},
    # {"name": "Дархан", "lat": 49.47, "lon": 105.97, "weight": 0.0},
    # {"name": "Эрдэнэт", "lat": 49.03, "lon": 104.08, "weight": 0.0},
]
WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "cloud_cover"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Системийн боломжит чадал (МВт). Мэдэгдэж байвал оруулна:
# дашбоард дээр таамаг энэ шугамыг давах эрсдэлийг харуулна.
CAPACITY_MW = 1800

# Хязгаарлалтын мэдээ байхгүй тул өгөгдлөөс таамаглана: бодит ачаалал таамгийн P10-аас доош
# унасан БӨГӨӨД боломжит чадлын энэ хувиас дээш байсан цагийг «хязгаарлалт байж болзошгүй» гэнэ.
CURTAIL_FRAC = 0.92


def likely_curtailed(actual, p10):
    """Хязгаарлалт байж болзошгүй эсэх (бодит ачаалал чадалд ойр, таамгаас огцом бага)."""
    if CAPACITY_MW is None or actual is None or p10 is None:
        return False
    return actual < p10 and actual >= CURTAIL_FRAC * CAPACITY_MW

LOAD_COLS = ["thermal", "wind", "solar", "hydro", "battery", "import", "zes_oyu"]


def norm_hours(df):
    """Хуучин өгөгдөлд өдрийн сүүлийн цаг 0 гэж хадгалагдсан байж болно → 24."""
    df = df.copy()
    df.loc[df["hour"] == 0, "hour"] = 24
    return df.drop_duplicates(["date", "hour"], keep="last")
