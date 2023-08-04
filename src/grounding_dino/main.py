import cv2
import torch
import os
from typing import List
from GroundingDINO.groundingdino.util.inference import Model
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np

from typing import List

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

app = FastAPI()

class ImageWithTags(BaseModel):
    image: UploadFile
    tags: List[str]

cwd = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(cwd)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUNDING_DINO_CONFIG_PATH = os.path.join(parent_directory, "dependent_repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(parent_directory, "dependent_repos/GroundingDINO/weights/groundingdino_swint_ogc.pth")
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

print("test")