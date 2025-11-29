# frontend/api_client.py
import requests
from typing import Optional, Dict, Any

BACKEND_BASE = "http://127.0.0.1:8000"

def get_soil(lat: float, lon: float) -> Dict[str, Any]:
    url = f"{BACKEND_BASE}/soil"
    r = requests.get(url, params={"lat": lat, "lon": lon}, timeout=15)
    r.raise_for_status()
    return r.json()

def get_weather(lat: float, lon: float) -> Dict[str, Any]:
    url = f"{BACKEND_BASE}/weather"
    r = requests.get(url, params={"lat": lat, "lon": lon}, timeout=15)
    r.raise_for_status()
    return r.json()

def predict(lat: Optional[float] = None, lon: Optional[float] = None,
            N: Optional[float] = None, P: Optional[float] = None, K: Optional[float] = None,
            ph: Optional[float] = None, temperature: Optional[float] = None,
            humidity: Optional[float] = None, rainfall: Optional[float] = None,
            user_id: Optional[int] = None, top_k: int = 3) -> Dict[str, Any]:
    url = f"{BACKEND_BASE}/predict"
    # We send as form-encoded params to the POST endpoint
    payload = {}
    if lat is not None: payload["lat"] = lat
    if lon is not None: payload["lon"] = lon
    if N is not None: payload["N"] = N
    if P is not None: payload["P"] = P
    if K is not None: payload["K"] = K
    if ph is not None: payload["ph"] = ph
    if temperature is not None: payload["temperature"] = temperature
    if humidity is not None: payload["humidity"] = humidity
    if rainfall is not None: payload["rainfall"] = rainfall
    if user_id is not None: payload["user_id"] = user_id
    payload["top_k"] = top_k

    r = requests.post(url, params=payload, timeout=20)
    r.raise_for_status()
    return r.json()
