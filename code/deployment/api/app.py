from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import joblib
import numpy as np

app = FastAPI(title="Image Colorizer API")

model = ...  # надо заменить путь на настоящий путь 

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("L")  #черно-белое изображение
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # --- Тут вызываем твою модель ---
    img_np = np.array(img)
    colorized_np = model.colorize(img_np) # поменять функцию на свою

    # --- Преобразуем обратно ---
    colorized_img = Image.fromarray(colorized_np.astype("uint8"))

    buf = io.BytesIO()
    colorized_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
