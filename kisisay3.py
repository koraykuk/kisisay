from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "ornek3.jpg"
results = model(image_path)[0]

boxes = results.boxes
cls = boxes.cls
conf = boxes.conf
xyxy = boxes.xyxy

person_class_id = 0
person_boxes = xyxy[(cls == person_class_id)]

indices = cv2.dnn.NMSBoxes(
    bboxes=person_boxes.tolist(),
    scores=conf[(cls == person_class_id)].tolist(),
    score_threshold=0.5,
    nms_threshold=0.4
)

true_person_count = len(indices)

print(f"Filtrelenmiş kişi sayısı: {true_person_count}")

annotated_image = results.plot()
cv2.imshow("Doğru Kişi Sayımı", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
