# backend/utils.py
import os
import csv
import time
import requests
from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    MODEL_PATH: str = "backend/crop_model.pkl"
    STATE_NPK_PATH: str = "backend/state_npk.csv"
    TRAINING_CSV: str = "backend/Crop_recommendation.csv"
    USER_AGENT: str = "SoilSenseApp (mannats2206@gmail.com)"  # Change to your email for Nominatim

class Logger:
    @staticmethod
    def info(msg: str):
        print(f"[INFO] {msg}")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {msg}")

# ---------------- Geocoding ----------------
class GeoCoder:
    NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

    @staticmethod
    def reverse_geocode(lat: float, lon: float):
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 6,
            "addressdetails": 1
        }
        headers = {"User-Agent": Config().USER_AGENT}
        r = requests.get(GeoCoder.NOMINATIM_URL, params=params, headers=headers, timeout=10)
        return r.json()

    @staticmethod
    def state_from_reverse(result: Dict):
        addr = result.get("address", {})
        return addr.get("state") or addr.get("region")

# ---------------- Weather ----------------
class WeatherService:
    OM_URL = "https://api.open-meteo.com/v1/forecast"

    @staticmethod
    def fetch_current(lat: float, lon: float):
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relativehumidity_2m,precipitation",
            "forecast_days": 1,
            "timezone": "auto"
        }
        r = requests.get(WeatherService.OM_URL, params=params, timeout=10)
        data = r.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relativehumidity_2m", [])
        prec = hourly.get("precipitation", [])

        return {
            "temperature": temps[-1] if temps else None,
            "humidity": hums[-1] if hums else None,
            "precipitation": prec[-1] if prec else None
        }

# ---------------- Soil Mapper ----------------
class SoilMapper:
    def __init__(self, path=None):
        self.path = path or Config().STATE_NPK_PATH
        self.map = {}
        self._load()

    def _normalize(self, name: str) -> str:
        return name.strip().upper()

    def _load(self):
        path_obj = Path(self.path)
        if not path_obj.exists():
            Logger.error(f"state_npk.csv not found at {self.path}")
            return

        with open(path_obj, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                state = self._normalize(row["State"])
                self.map[state] = {
                    "N": float(row["N"]),
                    "P": float(row["P"]),
                    "K": float(row["K"]),
                    "pH": float(row["pH"]),
                }

    def get(self, state_name: str):
        if not state_name:
            return None
        state_name = self._normalize(state_name)
        return self.map.get(state_name)
