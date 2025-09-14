from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from datetime import datetime
import os
import cv2
from deepface import DeepFace
import time

app = Flask(__name__)
app.secret_key = "aerotrack_secret"

CSV_PATH = "data/passengers.csv"


# ---- Load & clean CSV ----
def load_csv():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)

        expected_cols = [
            "Name", "TicketNumber", "FlightNumber",
            "Destination", "DepartureTime", "ImagePath", "LastSeenZone"
        ]

        # Add missing columns if any
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        df = df[expected_cols]  # reorder properly
        df = df.fillna("")
        df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
        df["ImagePath"] = df["ImagePath"].str.replace("\\", "/", regex=False)

    else:
        df = pd.DataFrame(columns=[
            "Name", "TicketNumber", "FlightNumber", "Destination",
            "DepartureTime", "ImagePath", "LastSeenZone"
        ])

    return df


df = load_csv()


def save_df():
    df.to_csv(CSV_PATH, index=False)


# ---- Helpers ----
def get_passenger_by_query(query: str):
    matches = df[
        (df["Name"].str.contains(query, case=False, na=False)) |
        (df["TicketNumber"].astype(str) == str(query).strip())
    ]
    return matches.iloc[0] if not matches.empty else None


def resolve_ref_images(passenger_row):
    """Get reference images for passenger."""
    refs = []

    # From CSV path
    if "ImagePath" in passenger_row and passenger_row["ImagePath"].strip():
        p = passenger_row["ImagePath"]
        if os.path.exists(p):
            refs.append(p)

    # Fallback: folder of images
    folder = os.path.join("passenger_images", str(passenger_row["Name"]).replace(" ", "_"))
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                refs.append(os.path.join(folder, f))

    # Deduplicate
    seen, uniq = set(), []
    for r in refs:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def webcam_match_against_refs(ref_paths, zone_name="Zone A"):
    if not ref_paths:
        return False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        return False

    time.sleep(2)  # warm-up
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("[ERROR] Could not read frame")
        return False

    print("[DEBUG] Frame captured")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for ref in ref_paths:
        try:
            result = DeepFace.verify(
                img1_path=rgb_frame,
                img2_path=ref,
                enforce_detection=False,
                model_name="Facenet",
                distance_metric="cosine"
            )
            if result.get("verified") or result.get("distance", 1) < 0.6:
                print(f"[INFO] Match found with {ref}")
                return True
        except Exception as e:
            print("[WARN] DeepFace error:", e)

    return False


def update_last_seen(ticket_number, zone_name):
    global df
    idx = df[df["TicketNumber"].astype(str) == str(ticket_number)].index
    if not idx.empty:
        df.at[idx[0], "LastSeenZone"] = zone_name
        save_df()


# ---- Routes ----
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u == "admin" and p == "admin123":
            session["user"] = u
            return redirect(url_for("dashboard"))
        return render_template("index.html", error="Invalid login!")
    return render_template("index.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    passenger = None
    status = None
    current_zone = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        row = get_passenger_by_query(query)

        if row is None:
            status = "not_in_csv"
            return render_template("dashboard.html", passenger=None, status=status, current_zone=None)

        passenger = row.to_dict()

        current_zone = "Zone A"
        ref_paths = resolve_ref_images(row)
        matched = webcam_match_against_refs(ref_paths, zone_name=current_zone)

        if matched:
            update_last_seen(row["TicketNumber"], current_zone)
            fresh = df[df["TicketNumber"].astype(str) == str(row["TicketNumber"])].iloc[0].to_dict()
            passenger = fresh
            status = "found"
        else:
            status = "not_detected"

    return render_template("dashboard.html", passenger=passenger, status=status, current_zone=current_zone)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route("/testcam")
def testcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return "❌ Webcam not accessible"
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "❌ Failed to grab frame"
    return " Webcam working inside Flask"


if __name__ == "__main__":
    app.run(debug=True)
