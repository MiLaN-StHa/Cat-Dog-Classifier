from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import shutil

app = FastAPI()

model = keras.models.load_model('cat-dog-classification.keras')

# 📁 Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict/")
async def upload_image(file: Annotated[UploadFile, File()]):

    # ✅ Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # ✅ Save file
    destination = UPLOAD_DIR / file.filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Load image from saved file
    image = Image.open(destination).convert("RGB")

    # ✅ Resize
    image = image.resize((224, 224))

    # ✅ Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0

    # ✅ Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predict
    prediction = model.predict(img_array, verbose=0)
    result = np.argmax(prediction)

    # ✅ Output
    if result == 0:
        output = "🐱 Cat"
    else:
        output = "🐶 Dog"

    return {
        "filename": file.filename,
        "prediction": output
    }