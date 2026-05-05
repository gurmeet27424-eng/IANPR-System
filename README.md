# IANPR-System
IANPR system with stollen vehicle detection
--------------------------
libraries to be unstalled command
--------------------------
    !pip install ultralytics easyocr gradio opencv-python-headless numpy pillow mysql-connector-python
---------------------
main code
-------------------

    import os #maincode
    import cv2
    import numpy as np
    import easyocr
    import gradio as gr
    from ultralytics import YOLO
    import sqlite3
    from datetime import datetime
    import csv
    from typing import List, Dict, Any, Tuple

    DB_FILE = "anpr_log.db"
    IMAGE_DIR = "plate_images"
    STOLEN_LIST_FILE = "stolen_vehicles.csv"


    try:
    yolo_model = YOLO("yolov8n.pt")
    ocr_reader = easyocr.Reader(['en'], gpu=True) # Set gpu=False if you don't have a CUDA-enabled GPU
    except Exception as e:
    print(f"Error loading models. Check your YOLO and EasyOCR installation/paths: {e}")
    # Exit or handle gracefully in a real application


    STOLEN_PLATES = set()



    def init_stolen_list_file():
    """Initializes the stolen vehicle list file if it doesn't exist."""
    if not os.path.exists(STOLEN_LIST_FILE):
        try:
            with open(STOLEN_LIST_FILE, 'w', newline='') as f:
                pass # Create the empty file
            print(f"Stolen vehicle list file '{STOLEN_LIST_FILE}' initialized.")
        except IOError as e:
             print(f"Critical error creating stolen list file: {e}")

    def load_stolen_list() -> set:
    """Loads plate numbers from the stolen vehicle list file into memory."""
    global STOLEN_PLATES
    STOLEN_PLATES.clear()
    try:
        with open(STOLEN_LIST_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    plate = row[0].strip().upper()
                    if plate:
                        STOLEN_PLATES.add(plate)
        print(f"Loaded {len(STOLEN_PLATES)} stolen plates.")
    except Exception as e:
        print(f"Error loading stolen list: {e}")
    return STOLEN_PLATES

    def init_db():
    """Initializes the SQLite database and creates the detections table."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_score REAL,
            plate_image_path TEXT
        )
        """
        )
        conn.commit()
        conn.close()
        print(f"Database '{DB_FILE}' initialized.")
        except Exception as e:
        print(f"Critical error initializing database: {e}")


    init_db()
    init_stolen_list_file()
    load_stolen_list()



    def preprocess_plate(plate_img: np.ndarray) -> np.ndarray | None:
    """Advanced preprocessing for better OCR accuracy."""
    if plate_img.size == 0: return None

    h, w = plate_img.shape[:2]
    if w < 200:
        scale = 200 / w
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Deskew logic (simplified)
    coords = np.column_stack(np.where(morph > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = 90 + angle
        if abs(angle) > 0.5:
            h, w = morph.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            morph = cv2.warpAffine(morph, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

    return morph

    def extract_text_multiple_methods(plate_crop: np.ndarray) -> Tuple[str, float]:
    """Try multiple preprocessing approaches and return best result."""
    texts = []
    confidences = []

    methods = [
        ("Original", plate_crop),
        ("Preprocessed", preprocess_plate(plate_crop)),
        ("Contrast Enhanced", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)))
    ]

    for name, img in methods:
        if img is not None:
            ocr_result = ocr_reader.readtext(img, detail=1)
            if ocr_result:
                text = " ".join([t[1] for t in ocr_result])
                avg_conf = np.mean([t[2] for t in ocr_result]) if ocr_result else 0.0
                texts.append(text)
                confidences.append(avg_conf)

    if texts:
        best_idx = np.argmax(confidences)
        return texts[best_idx], confidences[best_idx]

    return "", 0.0

    def clean_plate_text(text: str) -> str:
    """Clean and format detected plate text."""
    text = text.replace(" ", "").replace("-", "").replace(".", "")
    text = ''.join(c for c in text if c.isalnum())
    return text.upper()

    def select_plate(selected_option: str) -> str:
    """Extract clean plate number from the selected dropdown option."""
    if not selected_option: return ""

    # Handles both normal and stolen formats
    if selected_option.startswith("🚨🚨"):
        plate_number = selected_option.replace("🚨🚨 **STOLEN: ", "").split("** (confidence:")[0].strip()
    else:
        plate_number = selected_option.split(" (confidence:")[0].strip()
    return plate_number



    def recognize_plate(image: np.ndarray) -> Tuple[np.ndarray | None, str, gr.update, str, List[Dict[str, Any]], str]:
    """
    Enhanced license plate detection with YOLO, OCR, and STOLEN CHECK.
    """
    if image is None:
        return None, "No image provided", gr.update(choices=[], visible=False), "", [], "Status: No Detection"

    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    img = cv2.imread(temp_path)

    # 1. YOLO Detection (for vehicles)
    results = yolo_model.predict(source=temp_path, conf=0.25, iou=0.5, save=False, classes=[2, 3, 5, 7])

    detected_plates = []
    h_img, w_img = img.shape[:2]

    # 2. Contour-based Plate Finding (within image or vehicle regions)
    if not results or len(results[0].boxes) == 0:
        # Fallback: Search entire image for contours
        regions_to_search = [(0, 0, w_img, h_img)]
        summary_base = "Direct contour search used."
    else:
        # Search within detected vehicle bounding boxes (with margin)
        regions_to_search = []
        for result in results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                margin = 20
                regions_to_search.append((max(0, x1 - margin), max(0, y1 - margin), min(w_img, x2 + margin), min(h_img, y2 + margin)))
        summary_base = f"Found {len(regions_to_search)} vehicle region(s)."

    for x1_v, y1_v, x2_v, y2_v in regions_to_search:
        vehicle_crop = img[y1_v:y2_v, x1_v:x2_v]

        gray_vehicle = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_vehicle, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = w_c / float(h_c) if h_c > 0 else 0

            # License plate criteria
            if (2.0 <= aspect_ratio <= 5.0 and w_c > 50 and h_c > 15 and w_c < vehicle_crop.shape[1] * 0.9):

                plate_crop = vehicle_crop[y_c:y_c+h_c, x_c:x_c+w_c]
                text, conf = extract_text_multiple_methods(plate_crop)
                clean_text = clean_plate_text(text)

                if clean_text and conf > 0.3 and len(clean_text) >= 4:
                    # Global coordinates for drawing
                    px, py, pw, ph = x1_v + x_c, y1_v + y_c, w_c, h_c

                    # Save crop for DB log
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    filename = f"{clean_text}_{timestamp_str}.jpg"
                    image_path = os.path.join(IMAGE_DIR, filename)
                    cv2.imwrite(image_path, plate_crop)

                    detected_plates.append({
                        'text': clean_text,
                        'conf': conf,
                        'bbox': (px, py, pw, ph),
                        'image_path': image_path
                    })

    # 3. Finalize Detections
    unique_plates = {}
    for plate in detected_plates:
        text = plate['text']
        if text not in unique_plates or plate['conf'] > unique_plates[text]['conf']:
            unique_plates[text] = plate

    detected_plates = list(unique_plates.values())
    detected_plates.sort(key=lambda x: x['conf'], reverse=True)

    # 4. Stolen Check & UI Preparation
    stolen_status = "Status: OK"
    plate_choices = []
    img_with_all = img.copy()

    for plate in detected_plates:
        text = plate['text']
        conf = plate['conf']
        x, y, w, h = plate['bbox']

        is_stolen = text in STOLEN_PLATES

        # Color for UI drawing
        color = (0, 0, 255) if is_stolen else (0, 255, 0) # Red/Green

        if is_stolen:
            plate_choices.append(f"🚨🚨 **STOLEN: {text}** (confidence: {conf:.2f})")
            stolen_status = "🚨 **MATCH FOUND: STOLEN VEHICLE** 🚨"
        else:
            plate_choices.append(f"{text} (confidence: {conf:.2f})")

        # Draw on image
        cv2.rectangle(img_with_all, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img_with_all, f"{text} ({conf:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 5. Output Handling
    if not detected_plates:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img_rgb, "❌ No license plate detected.", gr.update(choices=[], visible=False), "", [], "Status: No Detection")

    img_rgb = cv2.cvtColor(img_with_all, cv2.COLOR_BGR2RGB)

    summary = stolen_status if stolen_status != "Status: OK" else (
        f"Detected: {detected_plates[0]['text']}" if len(detected_plates) == 1 else
        f"Found {len(detected_plates)} possible plates. Select the correct one."
    )

    default_selection = plate_choices[0]
    default_plate_text = detected_plates[0]['text']

    return (img_rgb, summary, gr.update(choices=plate_choices, value=default_selection, visible=True), default_plate_text, detected_plates, stolen_status)



    def save_to_db(selected_plate_text: str, all_detections_data: List[Dict[str, Any]]) -> str:
    """Saves the selected detection to the SQLite database."""
    if not selected_plate_text or not all_detections_data:
        return "⚠️ No plate selected or no detection data available."

    selected_data = next((d for d in all_detections_data if d['text'] == selected_plate_text), None)

    if selected_data is None:
        return f"❌ Error: Could not find data for '{selected_plate_text}' in session data."

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO detections (plate_number, confidence_score, plate_image_path) VALUES (?, ?, ?)",
            (selected_data['text'], selected_data['conf'], selected_data['image_path'])
        )
        conn.commit()
        conn.close()
        return f"✅ Saved '{selected_data['text']}' to database successfully!"
    except Exception as e:
        return f"❌ Database Error: {str(e)}"

    def insert_stolen_plate(plate_number: str) -> str:
    """Adds a new plate number to the stolen list file."""
    plate_number = plate_number.strip().upper()
    if not plate_number:
        return "⚠️ Plate number cannot be empty."

    try:
        # Check if already in the set
        if plate_number in STOLEN_PLATES:
            return f"ℹ️ Plate '{plate_number}' is already in the stolen list."

        with open(STOLEN_LIST_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([plate_number])

        # Update in-memory list
        STOLEN_PLATES.add(plate_number)
        return f"✅ Plate '{plate_number}' added to stolen list. Total Stolen: {len(STOLEN_PLATES)}"

    except Exception as e:
        return f"❌ File Handling Error: {str(e)}"

    def view_detection_log() -> str:
    """Fetches and displays the last 10 detection logs."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT plate_number, confidence_score, detection_timestamp FROM detections ORDER BY id DESC LIMIT 10")
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "No detections logged yet."

        output = "### 📖 Recent Detections Log (Last 10)\n"
        output += "| Plate Number | Confidence | Timestamp |\n"
        output += "|---|---|---|\n"
        for plate, conf, ts in results:
            output += f"| **{plate}** | {conf:.2f} | {ts.split('.')[0]} |\n"
        return output

    except Exception as e:
        return f"❌ Error retrieving log: {str(e)}"

    def view_stolen_log() -> str:
    """Fetches detections from the log that match any plate in the current STOLEN_PLATES set."""
    # 1. Reload the stolen list just to be safe
    current_stolen_plates = load_stolen_list()

    if not current_stolen_plates:
        return "No plates currently marked as stolen in the system (`stolen_vehicles.csv` is empty)."

    # Create placeholders for the SQL IN clause
    placeholders = ', '.join('?' for _ in current_stolen_plates)
    plate_list = tuple(current_stolen_plates)

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Execute the targeted SQL query
        query = f"""
        SELECT plate_number, confidence_score, detection_timestamp, plate_image_path
        FROM detections
        WHERE plate_number IN ({placeholders})
        ORDER BY detection_timestamp DESC
        """

        cursor.execute(query, plate_list)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "No matching detections found in the log for the currently marked stolen plates."

        # Format output
        output = "### 🚨 Stolen Vehicle Detection History 🚨\n"
        output += "| Plate Number | Confidence | Timestamp | Image Path |\n"
        output += "|---|---|---|---|\n"
        for plate, conf, ts, path in results:
            output += f"| **{plate}** | {conf:.2f} | {ts.split('.')[0]} | {path} |\n"
        return output

    except Exception as e:
        return f"❌ Error retrieving stolen vehicle log: {str(e)}"


    with gr.Blocks(theme=gr.themes.Soft(), title="ANPR System") as iface:
    gr.Markdown("# 🚨 Enhanced ANPR with Database Logging & Stolen Vehicle Check 🚨")
    gr.Markdown("Upload a photo of a vehicle to detect and read license plates. The system uses **YOLOv8** for vehicle detection and **EasyOCR** with advanced pre-processing for high accuracy.")

    # State variable to hold all detection data between steps
    detection_data_state = gr.State([])

    with gr.Tab("Plate Recognition & Logging"):

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload Car Image", sources=['upload', 'webcam'])
                detect_btn = gr.Button("🔎 **Detect License Plate**", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Detected Image")
                stolen_vehicle_status = gr.Textbox(label="🚨 **Stolen Vehicle Check Status** 🚨", lines=1, interactive=False, value="Status: ...")
                detection_status = gr.Textbox(label="Detection Status", lines=2, interactive=False)

        with gr.Row():
            plate_selector = gr.Dropdown(
                label="Select Correct Plate (if multiple detected)",
                choices=[],
                visible=False,
                interactive=True
            )

        selected_plate = gr.Textbox(
            label="Selected Plate Number",
            lines=1,
            interactive=False
        )

        gr.Markdown("---")

        # Logging Buttons
        with gr.Row():
            save_btn = gr.Button("💾 **Save Confirmed Plate to Database**", variant="secondary")
            db_status = gr.Textbox(label="Database Status", placeholder="Click 'Save' to log the selected plate...")

        # Event handlers for Recognition Tab
        detect_btn.click(
            fn=recognize_plate,
            inputs=[input_image],
            outputs=[output_image, detection_status, plate_selector, selected_plate, detection_data_state, stolen_vehicle_status]
        )

        plate_selector.change(
            fn=select_plate,
            inputs=[plate_selector],
            outputs=[selected_plate]
        )

        save_btn.click(
            fn=save_to_db,
            inputs=[selected_plate, detection_data_state],
            outputs=[db_status]
        )

    with gr.Tab("Log Viewer & Stolen Management"):

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ➕ Add Stolen Plate")
                stolen_plate_input = gr.Textbox(label="Plate Number to Mark as Stolen (e.g., AB123CD)")
                add_stolen_btn = gr.Button("➕ **Add to Stolen List**", variant="stop")
                stolen_status_output = gr.Textbox(label="Stolen List Update Status", placeholder="List updated when button is clicked.")

                add_stolen_btn.click(
                    fn=insert_stolen_plate,
                    inputs=stolen_plate_input,
                    outputs=[stolen_status_output]
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📜 Log Viewing Options")

                with gr.Row():
                    log_btn = gr.Button("📖 **View Recent Detection Log**", variant="secondary")
                    view_stolen_log_btn = gr.Button("🚨 **View DETECTED Stolen Vehicle History** 🚨", variant="stop")

                log_output = gr.Markdown("Log data will appear here...")

                log_btn.click(
                    fn=view_detection_log,
                    outputs=[log_output]
                )

                view_stolen_log_btn.click(
                    fn=view_stolen_log,
                    outputs=[log_output]
                )

        gr.Markdown("----------")

        # Button to check the in-memory list contents
        current_stolen_btn = gr.Button("🔄 **Reload & View Current Stolen Plates**")
        current_stolen_list = gr.Markdown("Current stolen list from `stolen_vehicles.csv` will appear here...")

        current_stolen_btn.click(
            fn=lambda: "### Current Stolen Plates (from CSV):\n\n" + "\n".join([f"* **{p}**" for p in load_stolen_list()]),
            outputs=[current_stolen_list]
        )


    iface.launch(share=True)
