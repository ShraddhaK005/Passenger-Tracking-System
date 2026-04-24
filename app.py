import os
import base64
import cv2
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify , Response
)
import pandas as pd
from face_matcher import detect_and_identify
from flask import make_response
from embedding_engine import build_embedding_database

build_embedding_database("dataset")

tracking_history = {}
last_seen_cache = {}
# ----------------------------
# BASIC CONFIG
# ----------------------------

app = Flask(__name__)
app.secret_key = "super-secret-key-change-this"

DATA_DIR = "data"
PASSENGERS_CSV = os.path.join(DATA_DIR, "passengers.csv")
DATASET_DIR = "dataset"  # where passenger image folders will be stored

ZONES = ["Zone A", "Zone B", "Zone C"]  # A = webcam, B/C = external later


# ----------------------------
# UTILS: FILE / DATA HELPERS
# ----------------------------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)


def init_passengers_csv():
    """Create CSV with correct columns if it doesn't exist."""
    if not os.path.exists(PASSENGERS_CSV):
        cols = [
            "passenger_id",
            "full_name",
            "mobile_number",
            "ticket_number",
            "flight_number",
            "airline",
            "destination",
            "departure_datetime",
            "baggage_id",
            "photo_folder",
            "last_seen_zone",
            "last_seen_time",
        ]
        df = pd.DataFrame(columns=cols)
        df.to_csv(PASSENGERS_CSV, index=False)


def load_passengers():
    ensure_dirs()
    init_passengers_csv()
    df = pd.read_csv(PASSENGERS_CSV)

    # Backward compatibility if old CSV without VIP column
    if "is_vip" not in df.columns:
        df["is_vip"] = False

    return df



def save_passengers(df: pd.DataFrame):
    df.to_csv(PASSENGERS_CSV, index=False)


def generate_passenger_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return int(df["passenger_id"].max()) + 1


def parse_departure(dt_str: str) -> datetime | None:
    """Parse 'YYYY-MM-DDTHH:MM' from HTML datetime-local input."""
    if not dt_str:
        return None
    try:
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def calculate_time_remaining(departure_dt: datetime | None):
    if departure_dt is None:
        return None, None
    now = datetime.now()
    delta = departure_dt - now
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return 0, "Flight departure time has already passed."
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return seconds, f"{hours:02d} hr {minutes:02d} min remaining."


def get_zone_counts():
    df = load_passengers()
    counts = {z: 0 for z in ZONES}
    if df.empty:
        return counts
    if "last_seen_zone" not in df.columns:
        return counts
    for z in ZONES:
        counts[z] = int((df["last_seen_zone"] == z).sum())
    return counts


# ----------------------------
# AUTH (SIMPLE STAFF LOGIN)
# ----------------------------

# For now: one hard-coded staff user
STAFF_USER = {
    "username": "admin",
    "password": "admin123",  # change this in real usage
}


def login_required(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return wrapper

# ----------------------------
# LIVE CAMERA STREAMING
# ----------------------------

def generate_frames(camera_source):
    cap = cv2.VideoCapture(camera_source)

    while True:
        success, frame = cap.read()
        if not success:
            break
        if isinstance(camera_source, str):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/<zone>')
def video_feed(zone):

    if zone == "A":
        source = 0  # laptop webcam

    elif zone == "B":
        source = "http://10.133.58.241:8080/video"  # your mobile cam

    else:
        source = 0

    return Response(generate_frames(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if (
            username == STAFF_USER["username"]
            and password == STAFF_USER["password"]
        ):
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


def staff_required(passenger_count: int) -> int:
    """
    Simple rule-based staff suggestion.
    Tune as you like.
    """
    if passenger_count == 0:
        return 0
    if passenger_count <= 10:
        return 1
    elif passenger_count <= 25:
        return 2
    elif passenger_count <= 50:
        return 3
    else:
        return 4


@app.route("/dashboard")
@login_required
def dashboard():
    counts = get_zone_counts()
    total = sum(counts.values())

    # staff suggestion per zone
    staff_suggestions = {
        zone: staff_required(count) for zone, count in counts.items()
    }

    # busiest zone (for highlight)
    busiest_zone = None
    busiest_staff = None
    if counts:
        busiest_zone = max(counts, key=counts.get)
        busiest_staff = staff_suggestions[busiest_zone]

    # Recently seen passengers (sorted by last_seen_time desc)
    df = load_passengers()
    recent_passengers = []
    vip_passengers = []
    flight_summaries = []
    trend_labels = []
    trend_series = {}

    if not df.empty:
        # recent passengers
        if "last_seen_time" in df.columns:
            recent_df = df[
                df["last_seen_time"].notna()
                & (df["last_seen_time"] != "")
            ]
            if not recent_df.empty:
                recent_df = recent_df.sort_values(
                    "last_seen_time", ascending=False
                ).head(5)
                recent_passengers = recent_df.to_dict(orient="records")

        # VIP passengers
        if "is_vip" in df.columns:
            vip_df = df[df["is_vip"] == True]
            if not vip_df.empty:
                vip_df = vip_df.sort_values(
                    "last_seen_time", ascending=False
                ).head(5)
                vip_passengers = vip_df.to_dict(orient="records")

        # Flight-level summaries
        if "flight_number" in df.columns:
            def compute_status(row):
                zone = str(row.get("last_seen_zone", "") or "")
                if zone == "Boarded":
                    return "Boarded"
                elif zone in ZONES:
                    return "Detected"
                else:
                    return "Not Detected"

            df["status"] = df.apply(compute_status, axis=1)

            flights = df["flight_number"].fillna("").unique().tolist()
            for f in flights:
                if not f:
                    continue
                g = df[df["flight_number"] == f]
                total_f = len(g)
                detected_f = (g["status"].isin(["Detected", "Boarded"])).sum()
                boarded_f = (g["status"] == "Boarded").sum()
                not_detected_f = total_f - detected_f
                readiness = int((detected_f / total_f) * 100) if total_f > 0 else 0

                flight_summaries.append({
                    "flight_number": f,
                    "total": total_f,
                    "detected": detected_f,
                    "boarded": boarded_f,
                    "not_detected": not_detected_f,
                    "readiness": readiness,
                })

        # Trend data (zone load per hour for today)
        if "last_seen_time" in df.columns:
            df_seen = df[df["last_seen_time"].notna() & (df["last_seen_time"] != "")]
            if not df_seen.empty:
                df_seen["dt"] = pd.to_datetime(df_seen["last_seen_time"], errors="coerce")
                today = datetime.now().date()
                df_today = df_seen[df_seen["dt"].dt.date == today]

                if not df_today.empty:
                    df_today["hour"] = df_today["dt"].dt.strftime("%H:00")
                    hours = sorted(df_today["hour"].unique().tolist())
                    trend_labels = hours

                    trend_series = {}
                    for z in ZONES:
                        counts_per_hour = []
                        for h in hours:
                            c = ((df_today["last_seen_zone"] == z) & (df_today["hour"] == h)).sum()
                            counts_per_hour.append(int(c))
                        trend_series[z] = counts_per_hour

    return render_template(
        "dashboard.html",
        counts=counts,
        zones=list(counts.keys()),
        count_values=list(counts.values()),
        total_passengers=total,
        staff_suggestions=staff_suggestions,
        busiest_zone=busiest_zone,
        busiest_staff=busiest_staff,
        recent_passengers=recent_passengers,
        vip_passengers=vip_passengers,
        flight_summaries=flight_summaries,
        trend_labels=trend_labels,
        trend_series=trend_series,
    )
@app.route("/passengers")
@login_required
def passengers():
    df = load_passengers()

    flights = []
    airlines = []
    passengers_list = []
    selected_flight = request.args.get("flight", "").strip()
    selected_airline = request.args.get("airline", "").strip()
    detected_count = 0
    not_detected_count = 0

    if not df.empty:
        flights = sorted(df["flight_number"].fillna("").unique().tolist())
        airlines = sorted(df["airline"].fillna("").unique().tolist())

        filtered = df.copy()

        if selected_flight:
            filtered = filtered[filtered["flight_number"] == selected_flight]

        if selected_airline:
            filtered = filtered[filtered["airline"] == selected_airline]

        def compute_status(row):
            zone = str(row.get("last_seen_zone", "") or "")
            if zone == "Boarded":
                return "Boarded"
            elif zone in ZONES:
                return "Detected"
            else:
                return "Not Detected"

        if not filtered.empty:
            filtered["status"] = filtered.apply(compute_status, axis=1)
            detected_count = (filtered["status"].isin(["Detected", "Boarded"])).sum()
            not_detected_count = (filtered["status"] == "Not Detected").sum()
            passengers_list = filtered.to_dict(orient="records")

    return render_template(
        "passengers.html",
        flights=flights,
        airlines=airlines,
        selected_flight=selected_flight,
        selected_airline=selected_airline,
        passengers_list=passengers_list,
        detected_count=detected_count,
        not_detected_count=not_detected_count,
    )




# ----------------------------
# PASSENGER REGISTRATION
# ----------------------------

@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        mobile_number = request.form.get("mobile_number", "").strip()
        ticket_number = request.form.get("ticket_number", "").strip()
        flight_number = request.form.get("flight_number", "").strip()
        airline = request.form.get("airline", "").strip()
        destination = request.form.get("destination", "").strip()
        departure_raw = request.form.get("departure_datetime", "").strip()
        baggage_id = request.form.get("baggage_id", "").strip()
        is_vip = bool(request.form.get("is_vip"))   # NEW


        front_img_b64 = request.form.get("front_image")
        left_img_b64 = request.form.get("left_image")
        right_img_b64 = request.form.get("right_image")

        if not full_name or not ticket_number:
            flash("Full name and ticket number are required.", "error")
            return redirect(url_for("register"))

        if not front_img_b64:
            flash("At least front image is required.", "error")
            return redirect(url_for("register"))

        df = load_passengers()

        # Check if ticket already exists
        if not df.empty and (df["ticket_number"] == ticket_number).any():
            flash("Passenger with this ticket number already exists.", "error")
            return redirect(url_for("register"))

        passenger_id = generate_passenger_id(df)

        # Folder for this passenger
        folder_name = f"{full_name.replace(' ', '_')}_{ticket_number}"
        passenger_folder = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(passenger_folder, exist_ok=True)

        # Save images
        def save_b64_image(b64_str, filename):
            if not b64_str:
                return
            header, _, data = b64_str.partition(",")
            img_bytes = base64.b64decode(data)
            with open(os.path.join(passenger_folder, filename), "wb") as f:
                f.write(img_bytes)

        save_b64_image(front_img_b64, "front.jpg")
        save_b64_image(left_img_b64, "left.jpg")
        save_b64_image(right_img_b64, "right.jpg")

        departure_dt = parse_departure(departure_raw)
        departure_str = departure_dt.isoformat() if departure_dt else ""

        new_row = {
            "passenger_id": passenger_id,
            "full_name": full_name,
            "mobile_number": mobile_number,
            "ticket_number": ticket_number,
            "flight_number": flight_number,
            "airline": airline,
            "destination": destination,
            "departure_datetime": departure_str,
            "baggage_id": baggage_id,
            "photo_folder": folder_name,
            "last_seen_zone": "",
            "last_seen_time": "",
            "is_vip": is_vip, 
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_passengers(df)

        flash("Passenger registered successfully.", "success")
        return redirect(url_for("register"))

    return render_template("register.html")


# ----------------------------
# PASSENGER TRACKING (SEARCH)
# ----------------------------

@app.route("/track", methods=["GET", "POST"])
@login_required
def track():
    result = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        df = load_passengers()

        if df.empty:
            result = {"status": "not_in_db"}

        else:
            row = df.loc[df["ticket_number"] == query]
            if row.empty:
                row = df.loc[df["full_name"].str.lower() == query.lower()]

            if row.empty:
                result = {"status": "not_in_db"}
            else:
                row = row.iloc[0]
                passenger_folder = os.path.join(DATASET_DIR, row["photo_folder"])

                zone_cameras = {
                    "Zone A": 0,
                    "Zone B": "http://10.133.58.241:8080/video"
                }

                detected_zone = None
                now = datetime.now()
                now_str = now.isoformat()
                for zone_name, cam_index in zone_cameras.items():
                    print(f"Scanning {zone_name}...")

                    try:
                        # ✅ correct function call
                        person = detect_and_identify(camera_index=cam_index)

                        print("Detected person:", person)
                        print("Expected:", row["photo_folder"])

                    except Exception as e:
                        print("Error:", e)
                        person = None

                    # ✅ correct matching logic
                    if person and person == row["photo_folder"]:
                        detected_zone = zone_name
                        now_str = datetime.now().isoformat()
                        print("Found in", detected_zone)
                        break
                # -------------------------------
                # 🔥 TRACKING HISTORY + DUPLICATE CONTROL
                # -------------------------------
                passenger_id = row["ticket_number"]

                if detected_zone:
                    last_zone = last_seen_cache.get(passenger_id)

                    # avoid duplicate same-zone spam
                    if last_zone != detected_zone:
                        last_seen_cache[passenger_id] = detected_zone

                        if passenger_id not in tracking_history:
                            tracking_history[passenger_id] = []

                        tracking_history[passenger_id].append({
                            "zone": detected_zone,
                            "time": now.strftime("%H:%M:%S")
                        })

                        # update CSV
                        df.loc[df["ticket_number"] == passenger_id, "last_seen_zone"] = detected_zone
                        df.loc[df["ticket_number"] == passenger_id, "last_seen_time"] = now_str
                        save_passengers(df)

                else:
                    print("⚠ Not found in any zone")

                # -------------------------------
                # 🚨 ALERT SYSTEM
                # -------------------------------
                departure_dt = None
                if isinstance(row.get("departure_datetime"), str) and row["departure_datetime"]:
                    try:
                        departure_dt = datetime.fromisoformat(row["departure_datetime"])
                    except:
                        pass

                seconds, time_msg = calculate_time_remaining(departure_dt)

                urgency = "normal"
                if departure_dt and seconds is not None:
                    if seconds <= 1800 and not detected_zone:
                        urgency = "critical"
                    elif seconds <= 7200:
                        urgency = "warning"

                # -------------------------------
                #  TRACKING DISPLAY
                # -------------------------------
                history = tracking_history.get(passenger_id, [])

                if not detected_zone:
                    status = "registered_not_found"
                    last_seen_zone_val = row.get("last_seen_zone", "")
                    last_seen_time_val = row.get("last_seen_time", "")
                else:
                    status = "found_in_zone"
                    last_seen_zone_val = detected_zone
                    last_seen_time_val = now_str

                result = {
                    "status": status,
                    "full_name": row["full_name"],
                    "ticket_number": row["ticket_number"],
                    "flight_number": row["flight_number"],
                    "destination": row["destination"],
                    "mobile_number": row["mobile_number"],
                    "airline": row["airline"],
                    "baggage_id": row["baggage_id"],
                    "departure_datetime": row["departure_datetime"],
                    "last_seen_zone": last_seen_zone_val,
                    "last_seen_time": last_seen_time_val,
                    "time_message": time_msg,
                    "urgency": urgency,
                    "tracking_history": history   
                }

    return render_template("track.html", result=result)
@app.route("/mark_boarded", methods=["POST"])
@login_required
def mark_boarded():
    ticket = request.form.get("ticket_number", "").strip()
    if not ticket:
        flash("No ticket number provided for override.", "error")
        return redirect(url_for("track"))

    df = load_passengers()
    if df.empty:
        flash("No passengers in database.", "error")
        return redirect(url_for("track"))

    mask = df["ticket_number"] == ticket
    if not mask.any():
        flash("Passenger not found for override.", "error")
        return redirect(url_for("track"))

    now_str = datetime.now().isoformat()
    df.loc[mask, "last_seen_zone"] = "Boarded"
    df.loc[mask, "last_seen_time"] = now_str
    save_passengers(df)

    flash(f"Passenger with ticket {ticket} marked as Boarded.", "success")
    return redirect(url_for("track"))
@app.route("/reset")
def reset():
    import os
    import shutil
    import stat

    def remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # delete CSV
    if os.path.exists("data/passengers.csv"):
        os.remove("data/passengers.csv")

    # delete dataset folder safely
    if os.path.exists("dataset"):
        try:
            shutil.rmtree("dataset", onerror=remove_readonly)
        except Exception as e:
            return f"Error deleting dataset: {e}"

    return "System Reset Done"
@app.route("/export_report")
@login_required
def export_report():
    df = load_passengers()
    if df.empty:
        csv_data = "message\nNo passengers in system\n"
    else:
        def compute_status(row):
            zone = str(row.get("last_seen_zone", "") or "")
            if zone == "Boarded":
                return "Boarded"
            elif zone in ZONES:
                return "Detected"
            else:
                return "Not Detected"

        df["status"] = df.apply(compute_status, axis=1)

        cols = [
            "full_name",
            "ticket_number",
            "flight_number",
            "airline",
            "destination",
            "departure_datetime",
            "last_seen_zone",
            "last_seen_time",
            "status",
            "is_vip",
        ]
        existing_cols = [c for c in cols if c in df.columns]
        export_df = df[existing_cols]

        csv_data = export_df.to_csv(index=False)

    resp = make_response(csv_data)
    resp.headers["Content-Disposition"] = "attachment; filename=daily_passenger_report.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp



# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    ensure_dirs()
    init_passengers_csv()
    app.run(debug=False)
