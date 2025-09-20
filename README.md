# 🧠 Real-Time Object Detection with YOLOv8 + COCO + Hailo-8

This project implements an **end-to-end real-time object detection pipeline** using **YOLOv8** trained on the **COCO dataset**, deployed on both desktop GPU and **Raspberry Pi 5 with Hailo-8 AI accelerator**, and integrated with a **PySide6 GUI** for real-time visualization, including DJI Tello drone video streams.

---

## 📦 Installation

### 1. Create Conda environment
```bash
conda create -n yolo-object-det python=3.10 -y
conda activate yolo-object-det
```

### 2. Install PyTorch with CUDA 12.8
```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install dependencies
```bash
pip install ultralytics opencv-python onnx onnxruntime-gpu matplotlib tqdm PySide6 av
```

---

## 📁 Dataset Preparation

### Download COCO 2017
- [COCO 2017](https://cocodataset.org/#download): `train2017.zip`, `val2017.zip`, `annotations_trainval2017.zip`

### Directory structure
```
datasets/coco/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/
│   ├── train2017/
│   └── val2017/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── coco.yaml
```

### Convert COCO JSON → YOLO format
Use the provided `coco_to_yolo.py` script to generate YOLO-style `.txt` labels inside `labels/`.

---

## 🏋️ Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="datasets/coco/coco.yaml",
    epochs=100,
    imgsz=512,
    batch=8,
    fraction=0.1,   # use 10% of COCO (~11k images)
    amp=False,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0
)
```

- Training outputs: `runs/detect/train*/weights/best.pt`

---

## 🔍 Inference (Desktop)

### Test on a video
```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture("test.mp4")

while True:
    ok, frame = cap.read()
    if not ok: 
        break

    frame = cv2.resize(frame, (1280, 720))  # resize for display

    res = model(frame, imgsz=512, conf=0.4)[0]
    
    for b in res.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        cls, conf = int(b.cls[0]), float(b.conf[0])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{model.names[cls]} {conf*100:.1f}%",(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    
    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📤 Model Export

```python
model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", opset=12, dynamic=True)
```

Output: `best.onnx`

---

## 🚀 Raspberry Pi 5 + Hailo-8 Deployment

### 1. Install Hailo SDK
- Download from [Hailo Developer Zone](https://hailo.ai/developer-zone/)

### 2. Compile ONNX → HEF
```bash
hailo_model_zoo compile --model-path best.onnx --target-device hailo8
```

Output: `best.hef`

### 3. Run inference on RPi5
Use HailoRT Python API or C++ SDK to load `.hef` and run inference with the Pi Camera Module 3.

---

## 🛰️ DJI Tello + PySide6 GUI

- **Video stream:** UDP H.264 → decoded with PyAV  
- **Inference:** YOLOv8 ONNX model  
- **GUI:** PySide6 QLabel/QPixmap overlay  

Example GUI code is provided in `gui/app.py`.

---

## 📊 Performance

| Platform                | Model   | FPS  | Notes                  |
|--------------------------|---------|------|------------------------|
| RTX 3050 Ti (4GB VRAM)  | yolov8n | ~30  | 512×512, batch=8       |
| Raspberry Pi 5 + Hailo8 | yolov8n | 15–25| 480p–640p, FP16/INT8   |
| DJI Tello + GUI         | yolov8n | 20+  | 960×720 → 640 downscale|

---

## 📜 License

- COCO dataset: [Creative Commons Attribution 4.0 License](https://cocodataset.org/#termsofuse)  
- Code: MIT License  

---

## 🚦 Roadmap

- [x] COCO dataset preparation  
- [x] YOLOv8 training (Colab + local GPU)  
- [x] Desktop video inference  
- [x] ONNX export  
- [x] Hailo-8 compilation  
- [x] RPi5 deployment  
- [x] PySide6 GUI + Tello integration  

---
