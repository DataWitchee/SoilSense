# backend/ml_model.py
import os
import io
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from backend.utils import Config, Logger

FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


# ------------------------------
# Input model for prediction
# ------------------------------
class SoilSampleModelInput(BaseModel):
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    ph: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    user_id: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


# ------------------------------
# Dataset Loader
# ------------------------------
class DatasetLoader:
    def __init__(self, path=None):
        self.path = path or Config().TRAINING_CSV

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Training data not found: {self.path}")
        return pd.read_csv(self.path)

    def from_bytes(self, data: bytes):
        return pd.read_csv(io.BytesIO(data))


# ------------------------------
# Preprocessor
# ------------------------------
class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        self.scaler.fit(X.values.astype(float))
        self.fitted = True

    def transform(self, X: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted.")
        return self.scaler.transform(X.values.astype(float))


# ------------------------------
# Model Trainer
# ------------------------------
class ModelTrainer:
    def train_from_df(self, df: pd.DataFrame, label_col='crop') -> Dict:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found.")

        y = df[label_col]
        X = df.drop(columns=[label_col])

        # Ensure consistent feature order
        if all(col in X.columns for col in FEATURE_ORDER):
            X = X[FEATURE_ORDER]
        else:
            # fallback: use numeric columns
            X = X.select_dtypes(include=[np.number])

        X = X.fillna(X.mean())

        pre = Preprocessor()
        pre.fit(X)
        X_scaled = pre.transform(X)

        clf = ExtraTreesClassifier(n_estimators=120, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = float(accuracy_score(y_test, preds))

        Logger.info(f"Training Accuracy: {acc}")

        return {
            "model": clf,
            "scaler": pre.scaler,
            "accuracy": acc
        }


# ------------------------------
# ML Model Wrapper
# ------------------------------
class MLModel:
    def __init__(self, model_path=None):
        self.model_path = model_path or Config().MODEL_PATH
        self.clf = None
        self.scaler = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            Logger.info("No model found yet. Train a model first.")
            return

        try:
            obj = joblib.load(self.model_path)
            self.clf = obj["model"]
            self.scaler = obj["scaler"]
            Logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            Logger.error(f"Error loading model: {e}")

    def save(self, model, scaler):
        joblib.dump({"model": model, "scaler": scaler}, self.model_path)
        self.clf = model
        self.scaler = scaler
        Logger.info(f"Model saved to {self.model_path}")

    def available(self):
        return self.clf is not None and self.scaler is not None

    def train_and_save(self, df, label_col='crop'):
        trainer = ModelTrainer()
        res = trainer.train_from_df(df, label_col)
        self.save(res["model"], res["scaler"])
        return {"accuracy": res["accuracy"]}

    def predict(self, sample: SoilSampleModelInput, top_k=3) -> List[str]:
        if not self.available():
            raise RuntimeError("Model not trained yet.")

        vec = []
        for f in FEATURE_ORDER:
            val = getattr(sample, f, None)
            vec.append(0.0 if val is None else float(val))

        X = np.array(vec).reshape(1, -1)
        X = self.scaler.transform(X)

        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(X)[0]
            classes = self.clf.classes_
            top_idx = probs.argsort()[::-1][:top_k]
            return [str(classes[i]) for i in top_idx]

        return [str(self.clf.predict(X)[0])]



class RecommendationEngine:
    def __init__(self, model: MLModel):
        self.model = model

    def recommend_for_sample(self, sample: SoilSampleModelInput, top_k=3):
        return self.model.predict(sample, top_k)
