import base64
import io
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
import cv2
from ultralytics import YOLO


app = FastAPI()

model = YOLO("model/cloths_detection_v1.pt")
def base64_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    image = cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("dataset/uploaded_images/img1.png", image)
    return image

@app.post("/")
async def home():
    return {"Message": "Welcome to my cloths detection portal!"}


@app.post("/detect_objects")
async def detect_objects(image_base64: str):
    image = base64_to_image(image_base64)

    # Preprocess image if necessary
    # cv2.imshow(image)

    # Perform object detection using your model
    # Replace this with your actual object detection logic
    detections = model.predict(image, save=True)  # Assuming a detect method in your model

    # Return the cloths detected Images
    return FileResponse("runs/detect/predict/image0.jpg", media_type="image/jpeg")
