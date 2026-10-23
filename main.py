import io
import os
import pickle

import librosa
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

MODEL_DIR = "artifacts"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

model = None
scaler = None
encoder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoder

    required_files = [
        "emotion_model.keras",
        "scaler.pkl",
        "label_encoder.pkl"
    ]

    for filename in required_files:
        if not os.path.isfile(os.path.join(MODEL_DIR, filename)):
            raise RuntimeError(
                f"Missing {filename}. Run train_model.py first."
            )

    model = tf.keras.models.load_model(
        f"{MODEL_DIR}/emotion_model.keras"
    )

    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    yield


app = FastAPI(
    title="Speech Emotion Analysis API",
    description="Predict emotion categories from speech audio.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "Speech Emotion Analysis API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict_emotion(
    file: UploadFile = File(...)
):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a WAV audio file."
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum size is 10 MB."
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:
        audio, sr = librosa.load(
            io.BytesIO(content),
            sr=22050,
            duration=3,
            offset=0.5
        )

        if len(audio) < 1000:
            raise ValueError("Audio recording is too short.")

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        features = scaler.transform(features)

        probabilities = model.predict(features, verbose=0)[0]
        index = int(np.argmax(probabilities))

        return {
            "predicted_emotion": str(encoder.classes_[index]),
            "model_score": round(float(probabilities[index]), 4),
            "score_percent": round(
                float(probabilities[index]) * 100, 2
            )
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process the audio. Check the WAV file."
        )
    finally:
        await file.close()