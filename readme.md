# 🗑️ Trash Object Detection System (YOLOv5 + ONNX + FastAPI)

An **end-to-end object detection pipeline** for identifying trash and plastic waste items using **YOLOv5**, optimized with **ONNX**, benchmarked on **CPU**, and deployed as a **FastAPI inference service**.

This project demonstrates the **complete ML lifecycle**:
dataset → training → evaluation → optimization → benchmarking → deployment.

---

## 🚀 Project Highlights

* ✅ Trained **YOLOv5** on the **TACO Trash Dataset**
* ✅ Achieved **mAP@50 ≈ 0.34** on 18 waste classes
* ✅ Exported model to **ONNX** for faster CPU inference
* ✅ Benchmarked **PyTorch vs ONNX** on CPU
* ✅ Built **FastAPI REST API** for real-time inference
* ✅ Auto-rescaled bounding boxes to original image size
* ✅ Visual output with drawn bounding boxes

---

## 🧠 Classes Detected (18)

Examples:

* Plastic bag / wrapper
* Bottle, Bottle cap
* Can, Carton
* Paper
* Cigarette
* Cup, Lid
* Straw
* Styrofoam piece
* Other litter, Other plastic
  …and more.

---

## 🗂️ Folder Structure

```
Object Detection/
│
├── training/
│   ├── taco-yolo.ipynb     # Training + evaluation + export
│
├── benchmarks/
│   ├── benchmark_pytorch_cpu.py
│   ├── benchmark_onnx_cpu.py
│   ├── best.pt
│   └── best.onnx
|
├── yolov5_fastapi/
│   ├── app.py                   # FastAPI application
│   ├── best.onnx                # ONNX model for inference
│   └── requirements.txt
│
└── README.md
```

---

## 📦 Dataset

* **Dataset**: TACO – Trash Annotations in Context
* **Format**: YOLO
* **Source**: Kaggle
* **Classes**: 18
* **Splits**:

  * Train
  * Validation
  * Test

Dataset was **used directly from Kaggle Input (read-only)**.

---

## 🧪 Work Done in Kaggle Notebook

All heavy ML tasks were performed in **Kaggle (GPU)**:

### 1️⃣ Dataset Preparation

* Used YOLO-format dataset (`train/`, `valid/`, `test/`)
* Verified `data.yaml` paths and class names

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

* Precision, Recall, mAP calculated per class
* AutoAnchor optimization applied
* Best weights saved as `best.pt`

### 4️⃣ Export to ONNX

```bash
python export.py \
  --weights runs/train/taco_yolov55/weights/best.pt \
  --include onnx
```

Output:

```
best.onnx (≈27 MB)
```

---

## 📊 Benchmark Results (CPU – VS Code)

Benchmarks were done **locally on CPU** using the same image resolution (640×640).

| Model   | Avg Inference Time | FPS   |
| ------- | ------------------ | ----- |
| PyTorch | 120.88 ms          | 8.27  |
| ONNX    | 49.53 ms           | 20.19 |

✅ **ONNX is ~2.4× faster than PyTorch on CPU**

---

## 🌐 FastAPI Inference Service

### 🔹 Features

* CPU-based ONNX inference
* Automatic bounding box rescaling
* JSON response endpoint
* Image output endpoint with drawn boxes

### 🔹 Endpoints

#### `POST /detect`

Returns detections as JSON.

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

Returns the image with bounding boxes drawn.

---

## ▶️ Running the API Locally

```bash
cd yolov5_fastapi
pip install -r requirements.txt
uvicorn app:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 🖼️ Sample Output

* Bounding boxes drawn in **green**
* Label + confidence shown
* Boxes correctly mapped to **original image size**

---

## 🧩 Technologies Used

* **Python**
* **YOLOv5**
* **PyTorch**
* **ONNX**
* **ONNX Runtime**
* **FastAPI**
* **OpenCV**
* **NumPy**
* **Kaggle GPU**
* **VS Code (CPU benchmarking)**

---

## 🎯 Key Learning Outcomes

* End-to-end object detection workflow
* Model optimization using ONNX
* CPU performance benchmarking
* Production-style API deployment
* Handling real-world datasets
* Clean separation of training and inference environments

---

## 📌 Future Enhancements

* Docker containerization
* GPU inference support
* Video stream detection
* Cloud deployment (AWS / Azure / GCP)
* Frontend dashboard

---

## 👤 Author

**Vishnu Vardhan Reddy**
Engineering Student | Full Stack & ML Enthusiast
