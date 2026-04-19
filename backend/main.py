import json
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model(r"C:\PlantAI\model\plant_model.h5")

with open(r"C:\PlantAI\model\class_mapping.json", "r") as f:
    class_mapping = json.load(f)

@app.get("/")
def root():
    return {"status": "PlantAI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "diseased" if prediction > 0.5 else "healthy"
    confidence = float(prediction) if label == "diseased" else float(1 - prediction)
    return JSONResponse({
        "result": label,
        "confidence": round(confidence * 100, 2)
    })

@app.post("/ndvi")
async def ndvi(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image).astype(float)
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    ndvi = (green - red) / (green + red + 1e-10)
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    ndvi_min = round(float(ndvi.min()), 4)
    ndvi_max = round(float(ndvi.max()), 4)
    ndvi_mean = round(float(ndvi.mean()), 4)
    return JSONResponse({
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "ndvi_mean": ndvi_mean,
        "health_status": "Healthy vegetation" if ndvi_mean > 0.2 else "Stressed vegetation"
    })