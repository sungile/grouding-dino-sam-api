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

print("test")