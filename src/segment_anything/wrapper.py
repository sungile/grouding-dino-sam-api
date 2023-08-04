SAM_ENCODER_VERSION = "vit_h"
from segment_anything import sam_model_registry, SamPredictor
from grounding_dino.main import grounding_dino_model, enhance_class_name
import os



import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

cwd = os.path.dirname(os.path.abspath(__file__))

SAM_CHECKPOINT_PATH = os.path.join(cwd, "weights", "sam_vit_h_4b8939.pth")
SOURCE_IMAGE_PATH = "/usr/sungile/grounding-dino-sam-api/images/bedroom-1.jpg"
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

import cv2
import supervision as sv

# load image
image = cv2.imread(SOURCE_IMAGE_PATH)
CLASSES = ["bed"]
# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=enhance_class_name(class_names=CLASSES),
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print(detections)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

print(annotated_frame)