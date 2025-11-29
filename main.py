

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json
import io
import pandas as pd
from datetime import datetime

from backend.ml_model import (
    MLModel,
    SoilSampleModelInput,
    RecommendationEngine
)

from backend.datastore import DataStore
from backend.utils import (
    GeoCoder,
    WeatherService,
    SoilMapper,
    Config,
    Logger
)


# -----------------------------
# INITIALIZE CORE COMPONENTS
# -----------------------------
config = Config()
db = DataStore()
soil_mapper = SoilMapper(path=config.STATE_NPK_PATH)
ml = MLModel(model_path=config.MODEL_PATH)
reco_engine = RecommendationEngine(ml)

app = FastAPI(
    title="SoilSense Backend",
    description="Location-aware Soil Nutrient & Crop Recommendation API",
    version="1.0.0"
)


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "model_available": ml.available()}


@app.get("/health")
def health():
    return {"status": "ok", "model_available": ml.available()}



@app.get("/soil")
def soil_by_latlon(
    lat: float = Query(...),
    lon: float = Query(...)
):
    try:
        geo = GeoCoder.reverse_geocode(lat, lon)
        state = GeoCoder.state_from_reverse(geo)
    except Exception as e:
        Logger.error(str(e))
        raise HTTPException(status_code=500, detail="Reverse geocoding failed.")

    if not state:
        return {"state": None, "nutrients": None}

    nutrients = soil_mapper.get(state)
    return {"state": state, "nutrients": nutrients}


# -----------------------------
# 📌 2. Weather Endpoint
# -----------------------------
@app.get("/weather")
def weather(lat: float = Query(...), lon: float = Query(...)):
    try:
        return WeatherService.fetch_current(lat, lon)
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# 📌 3. Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    N: Optional[float] = None,
    P: Optional[float] = None,
    K: Optional[float] = None,
    ph: Optional[float] = None,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    rainfall: Optional[float] = None,
    user_id: Optional[int] = None,
    top_k: int = 3
):

    sample = SoilSampleModelInput()

    # If lat/lon provided → fill nutrients + weather
    if lat is not None and lon is not None:
        try:
            geo = GeoCoder.reverse_geocode(lat, lon)
            state = GeoCoder.state_from_reverse(geo)
        except:
            state = None

        # Soil nutrients
        if state:
            nutrients = soil_mapper.get(state)
            if nutrients:
                sample.N = nutrients["N"]
                sample.P = nutrients["P"]
                sample.K = nutrients["K"]
                sample.ph = nutrients["pH"]

        # Weather
        w = WeatherService.fetch_current(lat, lon)
        temperature = temperature or w.get("temperature")
        humidity = humidity or w.get("humidity")
        rainfall = rainfall or w.get("precipitation")

    # Override manual values
    sample.N = N or sample.N
    sample.P = P or sample.P
    sample.K = K or sample.K
    sample.ph = ph or sample.ph
    sample.temperature = temperature
    sample.humidity = humidity
    sample.rainfall = rainfall
    sample.user_id = user_id
    sample.lat = lat
    sample.lon = lon

    if not ml.available():
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    try:
        recs = reco_engine.recommend_for_sample(sample, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        db.save_history(
            user_id,
            json.dumps(sample.dict()),
            ",".join(recs)
        )
    except:
        Logger.error("Failed to save history.")

    return {"recommended": recs, "input": sample.dict()}



@app.post("/train")
async def train(file: UploadFile = File(...), label_col: str = "crop"):
    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV format.")

    res = ml.train_and_save(df, label_col=label_col)
    return {"status": "trained", "accuracy": res["accuracy"]}


# -----------------------------
# 📌 5. Upload Model File
# -----------------------------
@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    content = await file.read()

    try:
        with open(config.MODEL_PATH, "wb") as f:
            f.write(content)
        ml._load()
        return {"status": "ok", "message": "Model uploaded & loaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/history/{user_id}")
def history(user_id: int):
    return {"history": db.get_history(user_id)}

