from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

img = cv2.imread("resultado/teste2.jpeg")

results = model.predict(
    img,
    imgsz=640,
    conf=0.15,
    verbose=False
)

total = 0

for r in results:

    print("Boxes:", len(r.boxes))

    total += len(r.boxes)

    for box in r.boxes:

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0,255,0),
            2
        )

print("TOTAL:", total)

cv2.imwrite(
    "debug_deteccao.jpg",
    img
)