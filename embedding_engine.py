import os
import numpy as np
from deepface import DeepFace

# store embeddings in memory
embedding_db = {}

def build_embedding_database(dataset_dir):
    print("Building embedding database...")

    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)

        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)

            try:
                emb = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                detector_backend="opencv",   # 🔥 IMPORTANT
                enforce_detection=False
            )[0]["embedding"]

                embeddings.append(np.array(emb))

            except Exception as e:
                print("Error:", e)

        if embeddings:
            embedding_db[person] = embeddings

    print("Database ready")


def find_match(face_embedding, threshold=1.0):
    best_match = None
    best_distance = 999

    for person, embeddings in embedding_db.items():
        for emb in embeddings:
            dist = np.linalg.norm(face_embedding - emb)

            if dist < best_distance:
                best_distance = dist
                best_match = person

    print("Best Distance:", best_distance)  # 🔥 debug

    # ✅ FINAL DECISION
    if best_distance < threshold:
        return best_match, best_distance
    else:
        return None, best_distance