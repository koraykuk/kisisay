import cv2

image = cv2.imread("ornek3.jpg")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                        padding=(8, 8), scale=1.05)

for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Tespit edilen kişi sayısı: {len(rects)}")

cv2.imshow("Kisiler", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
