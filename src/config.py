"""Төслийн тохиргоо."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS_DATA = ROOT / "docs" / "data"

TZ = "Asia/Ulaanbaatar"

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

LOAD_COLS = ["thermal", "wind", "solar", "hydro", "battery", "import", "zes_oyu"]
