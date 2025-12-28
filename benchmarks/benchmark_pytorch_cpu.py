import pathlib
import torch
import time

# 🔧 FIX: allow PosixPath on Windows
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model (CPU)
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="best.pt",
    force_reload=True  # important
)

model.eval()
model.to("cpu")

# Dummy input
dummy = torch.randn(1, 3, 640, 640)

# Warm-up
for _ in range(10):
    _ = model(dummy)

# Benchmark
runs = 50
start = time.time()
for _ in range(runs):
    _ = model(dummy)
end = time.time()

avg_time = (end - start) / runs
fps = 1 / avg_time

print(f"🔥 PyTorch CPU Avg Inference Time: {avg_time*1000:.2f} ms")
print(f"🔥 PyTorch CPU FPS: {fps:.2f}")
