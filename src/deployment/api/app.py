import sys
import os

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from skimage import color
from skimage.transform import resize

# Import your actual model class
from src.models.colorizer import ColorizerNet

app = FastAPI(title="Image Colorizer API")

# --- Load your PyTorch model ---
model_path = "models/best.pt"
model = ColorizerNet(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()  # set to evaluation mode

# --- Transform ---
to_tensor = transforms.ToTensor()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("L")  # grayscale
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # --- Resize to 64x64 ---
    img = img.resize((64, 64), Image.BICUBIC)

    # --- Preprocess ---
    img_tensor = to_tensor(img).unsqueeze(0)  # [1,1,H,W]

    # --- Model inference ---
    with torch.no_grad():
        ab_tensor = model(img_tensor)  # [1,2,H,W]

    # --- Convert L+ab to RGB ---
    L = img_tensor.squeeze(0).squeeze(0).numpy() * 100        # [H,W]
    ab = ab_tensor.squeeze(0).permute(1, 2, 0).numpy() * 128  # [H,W,2]

    if L.shape != ab.shape[:2]:
        ab = resize(ab, (L.shape[0], L.shape[1]), preserve_range=True)

    lab = np.concatenate([L[:, :, np.newaxis], ab], axis=2)
    rgb = color.lab2rgb(lab)

    colorized_img = Image.fromarray((rgb * 255).astype("uint8"))

    # --- Return image ---
    buf = io.BytesIO()
    colorized_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
