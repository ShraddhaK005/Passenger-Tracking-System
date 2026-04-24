import cv2
import time
from deepface import DeepFace

# ✅ ADD THIS (face detector)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_identify(camera_index=0, timeout=10):

    cap = cv2.VideoCapture(camera_index)
    time.sleep(2)

    if not cap.isOpened():
        print("Camera failed")
        return None

    start = time.time()
    temp_path = "capture.jpg"

    while time.time() - start < timeout:

        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ✅ ADD FACE CHECK (MAIN FIX)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # 🚨 IF NO FACE → SKIP MATCHING
        if len(faces) == 0:
            print("No face detected")
            continue

        # capture frame ONLY if face exists
        cv2.imwrite(temp_path, frame)

        try:
            result = DeepFace.find(
                img_path=temp_path,
                db_path="dataset",
                model_name="Facenet",
                enforce_detection=False
            )

            cap.release()
            cv2.destroyAllWindows()

            # check if match found
            if len(result) > 0 and not result[0].empty:
                identity_path = result[0].iloc[0]['identity']

                # extract folder name
                person = identity_path.split("\\")[-2]

                print("Detected:", person)
                return person

            else:
                print("No match found")
                return None

        except Exception as e:
            print("Error:", e)

    cap.release()
    cv2.destroyAllWindows()
    return None