
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Force DirectShow backend on Windows
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
