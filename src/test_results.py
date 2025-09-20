from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train7/weights/best.pt")
cap = cv2.VideoCapture("test.mp4")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, (1280, 720))

    res = model(frame, imgsz=512, conf=0.4)[0]
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[cls]} {conf*100:.1f}%", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()