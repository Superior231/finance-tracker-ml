"""
Flask API untuk Receipt Parser (PaddleOCR + Random Forest)
Versi ini sudah mendukung CORS agar bisa dipanggil langsung dari website (JS fetch)
"""

import os
import json
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Set environment variable Paddle sebelum import library Paddle
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
from ml_receipt_parser import MLReceiptParser

app = Flask(__name__)

# Aktifkan CORS untuk semua route -> browser dari domain lain (website Anda) bisa akses API ini.
# Kalau mau dibatasi hanya untuk domain tertentu, ganti origins="*" jadi origins=["https://situsanda.com"]
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================================
# INITIALIZATION (Dijalankan sekali saat server booting)
# ============================================================================
MODEL_PATH = "models/receipt_parser_rf.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained Random Forest model tidak ditemukan di {MODEL_PATH}. "
        "Sila lakukan training terlebih dahulu."
    )

print("\n--- INITIALIZING RECEIPT PARSER FLASK API ---")

print("Loading PaddleOCR...")
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_unclip_ratio=1.2,
    enable_mkldnn=True,
)
print("✓ PaddleOCR Loaded successfully.")

print(f"Loading trained ML Parser from: {MODEL_PATH}")
receipt_parser = MLReceiptParser(model_path=MODEL_PATH)
print("✓ ML Parser Loaded successfully.\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_ocr_result(predict_results, source_name: str):
    """Ekstrak dict OCR dari objek hasil PaddleOCR (menangani berbagai versi output)."""
    if not predict_results:
        return None

    ocr_obj = predict_results[0] if isinstance(predict_results, list) else predict_results
    ocr_result = None

    if hasattr(ocr_obj, "to_dict"):
        ocr_result = ocr_obj.to_dict()
    elif hasattr(ocr_obj, "rec_texts"):
        ocr_result = {
            "rec_texts": ocr_obj.rec_texts,
            "rec_boxes": getattr(ocr_obj, "rec_boxes", []),
            "rec_scores": getattr(ocr_obj, "rec_scores", []),
            "input_path": source_name,
        }
    elif isinstance(ocr_obj, dict):
        ocr_result = ocr_obj
    else:
        temp_dir = "temp_ocr"
        os.makedirs(temp_dir, exist_ok=True)
        ocr_obj.save_to_json(temp_dir)

        base_name = os.path.splitext(os.path.basename(source_name))[0]
        json_files = [
            f for f in os.listdir(temp_dir)
            if f.startswith(base_name) and f.endswith(".json")
        ]

        if json_files:
            with open(os.path.join(temp_dir, json_files[0]), "r", encoding="utf-8") as f:
                ocr_result = json.load(f)
            os.remove(os.path.join(temp_dir, json_files[0]))

    return ocr_result


def run_pipeline(img, source_name: str):
    """Jalankan OCR + ML parsing untuk satu gambar (numpy array BGR)."""
    predict_results = ocr.predict(img)
    ocr_result = extract_ocr_result(predict_results, source_name)

    if ocr_result is None or "rec_texts" not in ocr_result:
        return {
            "success": False,
            "error": "Cannot extract OCR data from internal result object",
        }, 500

    parsed_result = receipt_parser.parse(ocr_result)

    if parsed_result.get("status") == "Success✅":
        return {"success": True, "data": parsed_result}, 200
    else:
        return {"success": False, "data": parsed_result}, 200


# ============================================================================
# API ROUTES
# ============================================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint untuk memastikan API berjalan dengan baik"""
    return jsonify({
        "status": "healthy",
        "message": "Receipt Parser API (PaddleOCR + Random Forest) is ready."
    }), 200


@app.route("/api/parse-receipt", methods=["POST"])
def parse_receipt():
    """
    Endpoint utama untuk parsing nota pembayaran.

    Mendukung DUA cara kirim gambar:
    1. multipart/form-data dengan key 'image'  (form upload biasa dari HTML <form>)
    2. application/json dengan key 'image_base64' (kalau website kirim base64 string)
    """
    try:
        img = None
        source_name = "upload.jpg"

        # --- Cara 1: multipart/form-data ---
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No file selected"}), 400

            source_name = file.filename
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # --- Cara 2: JSON base64 ---
        elif request.is_json:
            body = request.get_json(silent=True) or {}
            b64_data = body.get("image_base64")

            if not b64_data:
                return jsonify({
                    "success": False,
                    "error": "Field 'image_base64' tidak ditemukan di body JSON"
                }), 400

            # Buang prefix data URL kalau ada, misal "data:image/jpeg;base64,...."
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            file_bytes = np.frombuffer(base64.b64decode(b64_data), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_name = body.get("filename", "upload.jpg")

        else:
            return jsonify({
                "success": False,
                "error": "Kirim gambar via form-data (key: 'image') atau JSON (key: 'image_base64')"
            }), 400

        if img is None:
            return jsonify({"success": False, "error": "Invalid image file format"}), 400

        payload, status_code = run_pipeline(img, source_name)
        return jsonify(payload), status_code

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    # host="0.0.0.0" supaya bisa diakses dari device lain di jaringan yang sama, bukan cuma localhost
    app.run(host="0.0.0.0", port=5000, debug=False)
    