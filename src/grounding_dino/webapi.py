import cv2
import supervision as sv
import torch
import os
from typing import List
from GroundingDINO.groundingdino.util.inference import Model
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np

app = FastAPI()

class ImageWithTags(BaseModel):
    image: UploadFile
    tags: List[str]

cwd = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUNDING_DINO_CONFIG_PATH = os.path.join(cwd, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(cwd, "GroundingDINO/groundingdino/checkpoints/GroundingDINO_SwinT_OGC.pth")
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

async def read_image_file(file) -> np.ndarray:
    image = Image.open(io.BytesIO(await file.read()))
    image_array = np.array(image)
    return image_array

@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.post("/detect/")
# async def detect(req: ImageWithTags):
#     image_file = req.image
#     tags = req.tags

#      # Convert the image file to a NumPy array
#     image = await read_image_file(image_file)

#     # detect objects
#     detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=enhance_class_name(class_names=tags),
#         box_threshold=BOX_THRESHOLD,
#         text_threshold=TEXT_THRESHOLD
#     )

#     return detections


    #annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
