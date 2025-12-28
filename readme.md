# 🗑️ Trash Object Detection System

### YOLOv5 · ONNX · FastAPI

An **end-to-end object detection pipeline** for identifying trash and plastic waste using **YOLOv5**, optimized with **ONNX**, benchmarked on **CPU**, and deployed as a **FastAPI inference service**.

This project demonstrates the **complete machine learning lifecycle**:

**Dataset → Training → Evaluation → Optimization → Benchmarking → Deployment**

---

## 🚀 Project Highlights

* ✅ Trained **YOLOv5** on the **TACO (Trash Annotations in Context) dataset**
* ✅ Achieved **mAP@50 ≈ 0.34** across **18 trash classes**
* ✅ Converted PyTorch model to **ONNX** for optimized CPU inference
* ✅ Benchmarked **PyTorch vs ONNX** performance on CPU
* ✅ Deployed model using **FastAPI**
* ✅ Auto-rescaled bounding boxes to original image resolution
* ✅ JSON + image-based inference outputs

---

## 🧠 Classes Detected (18)

Includes (but not limited to):

* Plastic bag / wrapper
* Bottle, Bottle cap
* Can, Carton
* Paper
* Cigarette
* Cup, Lid
* Straw
* Styrofoam piece
* Other litter
* Other plastic

---

## 🗂️ Project Structure

```text
Object Detection/
│
├── training/                         # Kaggle (GPU)
│   └── taco-yolo.ipynb               # Dataset prep, training, evaluation, ONNX export
│
├── benchmarks/                       # Local (CPU)
│   ├── benchmark_pytorch_cpu.py
│   └── benchmark_onnx_cpu.py
│
├── yolov5_fastapi/                   # Local deployment
│   ├── app.py                        # FastAPI application
│   ├── best.onnx                     # ONNX model (tracked via Git LFS)
│   └── requirements.txt
│
└── README.md
```

📌 **Note**

* Training artifacts and datasets remain on **Kaggle**
* Only **essential inference & benchmarking files** are stored locally
* Dataset files are intentionally **not included** in the repository

---

## 📦 Dataset

* **Name**: TACO – Trash Annotations in Context
* **Source**: Kaggle
* **Format**: YOLO
* **Classes**: 18
* **Splits**: Train / Validation / Test

Dataset was accessed directly from **Kaggle Input (read-only)**.

---

## 🧪 Work Done in Kaggle (GPU)

All computationally intensive steps were performed in a **Kaggle notebook using GPU**.

### 1️⃣ Dataset Preparation

* Verified YOLO directory structure
* Validated `data.yaml` paths and class labels

### 2️⃣ Training YOLOv5

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 30 \
  --data data.yaml \
  --weights yolov5s.pt \
  --name taco_yolov5 \
  --cache ram
```

### 3️⃣ Model Evaluation

* Precision, Recall, and mAP computed per class
* AutoAnchor optimization applied
* Best model saved as `best.pt`

### 4️⃣ Export to ONNX

```bash
python export.py \
  --weights runs/train/taco_yolov55/weights/best.pt \
  --include onnx
```

Output:

```
best.onnx (~27 MB)
```

---

## 📊 CPU Benchmark Results (VS Code)

Benchmarks were performed **locally on CPU**, using the same input resolution (640×640).

| Model   | Avg Inference Time | FPS   |
| ------- | ------------------ | ----- |
| PyTorch | 120.88 ms          | 8.27  |
| ONNX    | 49.53 ms           | 20.19 |

✅ **ONNX provides ~2.4× faster inference on CPU**

---

## 🌐 FastAPI Inference Service

### 🔹 Key Features

* CPU-based ONNX inference
* Automatic bounding box rescaling
* JSON response endpoint
* Image response endpoint with bounding boxes drawn

---

### 🔹 API Endpoints

#### `POST /detect`

Returns detection results in JSON format.

```json
{
  "count": 3,
  "detections": [
    {
      "class": "Can",
      "confidence": 0.973,
      "bbox": [177, 97, 139, 126]
    }
  ]
}
```

#### `POST /detect-image`

Returns the image with bounding boxes and labels rendered.

---

## ▶️ Running the API Locally

```bash
cd yolov5_fastapi
pip install -r requirements.txt
uvicorn app:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🖼️ Output Visualization

* Green bounding boxes
* Class labels with confidence scores
* Boxes correctly mapped to original image resolution

---

## 🧩 Technologies Used

* Python
* YOLOv5
* PyTorch
* ONNX / ONNX Runtime
* FastAPI
* OpenCV
* NumPy
* Kaggle (GPU)
* VS Code (CPU inference & benchmarking)

---

## 🎯 Key Learning Outcomes

* End-to-end object detection pipeline design
* Dataset handling in real-world scenarios
* Model optimization using ONNX
* CPU performance benchmarking
* REST API deployment for ML inference
* Clean separation of training and deployment environments

---

## 📌 Future Enhancements

* Docker containerization
* GPU inference support
* Video stream detection
* Cloud deployment (AWS / Azure / GCP)
* Frontend visualization dashboard

---

## 👤 Author

**Vishnu Vardhan Reddy**
Engineering Student | Full Stack & Machine Learning Enthusiast
