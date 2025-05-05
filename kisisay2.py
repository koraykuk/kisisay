from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "ornek3.jpg"

results = model(image_path)

person_class_id = 0
person_count = sum(1 for c in results[0].boxes.cls if int(c) == person_class_id)

print(f"Fotoğrafta tespit edilen kişi sayısı: {person_count}")

annotated_image = results[0].plot()

cv2.imshow("Tespit Sonucu - Kisiler", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
