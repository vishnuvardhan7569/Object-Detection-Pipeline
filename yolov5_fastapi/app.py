from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import cv2

app = FastAPI(title="YOLOv5 Object Detection API")

# ---------------- CONFIG ----------------
MODEL_PATH = "best.onnx"
IMG_SIZE = 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

CLASS_NAMES = [
    "Aluminium foil", "Bottle cap", "Bottle", "Broken glass", "Can",
    "Carton", "Cigarette", "Cup", "Lid", "Other litter",
    "Other plastic", "Paper", "Plastic bag - wrapper",
    "Plastic container", "Pop tab", "Straw",
    "Styrofoam piece", "Unlabeled litter"
]

# ---------------- LOAD MODEL ----------------
session = ort.InferenceSession(
    MODEL_PATH, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# ---------------- UTILS ----------------
def preprocess(image: Image.Image):
    original_w, original_h = image.size
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image_resized).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, original_w, original_h


def nms(boxes, scores, threshold):
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, CONF_THRESHOLD, threshold
    )
    return indices.flatten() if len(indices) > 0 else []


def postprocess(outputs, orig_w, orig_h):
    predictions = outputs[0][0]

    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        conf = float(pred[4])
        if conf < CONF_THRESHOLD:
            continue

        class_id = int(np.argmax(pred[5:]))
        cx, cy, w, h = pred[:4]

        # Convert to top-left format (640 scale)
        x = cx - w / 2
        y = cy - h / 2

        # Rescale to original image
        x = int(x * orig_w / IMG_SIZE)
        y = int(y * orig_h / IMG_SIZE)
        w = int(w * orig_w / IMG_SIZE)
        h = int(h * orig_h / IMG_SIZE)

        boxes.append([x, y, w, h])
        scores.append(conf)
        class_ids.append(class_id)

    indices = nms(boxes, scores, NMS_THRESHOLD)

    detections = []
    for i in indices:
        detections.append({
            "class": CLASS_NAMES[class_ids[i]],
            "confidence": round(scores[i], 3),
            "bbox": boxes[i]
        })

    return detections


def draw_boxes(image: Image.Image, detections):
    draw = ImageDraw.Draw(image)

    for det in detections:
        x, y, w, h = det["bbox"]
        label = f"{det['class']} {det['confidence']}"

        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline="lime",
            width=3
        )
        draw.text((x, y - 12), label, fill="lime")

    return image

# ---------------- API ENDPOINTS ----------------
@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor, ow, oh = preprocess(image)
    outputs = session.run(None, {input_name: input_tensor})

    detections = postprocess(outputs, ow, oh)

    return {
        "count": len(detections),
        "detections": detections
    }


@app.post("/detect-image")
async def detect_objects_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor, ow, oh = preprocess(image)
    outputs = session.run(None, {input_name: input_tensor})

    detections = postprocess(outputs, ow, oh)
    image = draw_boxes(image, detections)

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")
