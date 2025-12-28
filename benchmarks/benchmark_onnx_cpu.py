import onnxruntime as ort
import numpy as np
import time

session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Warm-up
for _ in range(10):
    session.run(None, {input_name: dummy})

# Benchmark
runs = 50
start = time.time()
for _ in range(runs):
    session.run(None, {input_name: dummy})
end = time.time()

avg_time = (end - start) / runs
fps = 1 / avg_time

print(f"⚡ ONNX CPU Avg Inference Time: {avg_time*1000:.2f} ms")
print(f"⚡ ONNX CPU FPS: {fps:.2f}")
