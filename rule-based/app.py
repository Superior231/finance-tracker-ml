import os

# Nonaktifkan pengecekan konektivitas ke model hoster saat inisialisasi PaddleOCR
# (harus di-set SEBELUM import paddleocr)
# CATATAN: pesan error menyebut "DISABLE_MODEL_SOURCE_CHECK", tapi variabel yang
# benar-benar dibaca oleh PaddleX adalah PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK
# (lihat: https://github.com/PaddlePaddle/PaddleOCR/issues/17598)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
from parser import ReceiptParser
import json
import glob

# ----------------------
# Folder Paths
# ----------------------
IMG_DIR = "img"
OUTPUT_DIR = "output"
RESULTS_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ekstensi gambar yang didukung
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Model
model = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_unclip_ratio=1.2,                          # Diturunkan agar box tidak gampang nyambung
    enable_mkldnn=True,                                 # Mengakselerasi komputasi pada CPU Intel/AMD
)

# Post-processing -> Parsing with Fuzzy Matching
parser = ReceiptParser()

# Ambil semua file gambar di dalam folder img/
img_paths = sorted([
    p for p in glob.glob(os.path.join(IMG_DIR, "*"))
    if p.lower().endswith(IMG_EXTENSIONS)
])

if not img_paths:
    print(f"| Tidak ada gambar ditemukan di folder '{IMG_DIR}/'")

for IMG_PATH in img_paths:
    base_name = os.path.splitext(os.path.basename(IMG_PATH))[0]

    try:
        # Predict
        predict = model.predict(IMG_PATH)
        for res in predict:
            res.save_to_json(OUTPUT_DIR)

        FILE_NAME = os.path.join(OUTPUT_DIR, f"{base_name}_res.json")

        with open(FILE_NAME, encoding="utf-8") as f:
            data = json.load(f)

        print("="*50)
        print(f"| Test: {FILE_NAME}")
        print("="*50)

        result = parser.parse(data)

        if not result['items']:
            print(f"| Status: Error ❌")
            print(f"| Detail: Tidak ada item yang terdeteksi.")
            print("-"*50)
            continue  # Lanjut ke gambar berikutnya

        print("{:<3} {:<30} {:<6} {:<6}".format(" No.", "Name", "Qty", "Price"))
        for i, item in enumerate(result['items'], 1):
            print(" {:<3} {:<30} {:<6} {:>6,}".format(
                i, item['item_name'], item['qty'], item['price']
            ))

        total = sum(item['qty'] * item['price'] for item in result['items'])

        print("-"*50)
        print(" {:<3} {:<30} {:<6} {:>6,}".format("", "", "Total", total))
        print("-"*50)
        print(f"| Status: {result['status']}")
        print("-"*50)

        parser.save_parsed_to_json(
            result['items'],
            os.path.join(RESULTS_DIR, os.path.basename(FILE_NAME))
        )

    except Exception as e:
        # Tangkap error apapun (mis. gagal baca file, gagal parsing, dll)
        # agar batch tetap lanjut ke gambar berikutnya, tidak berhenti total.
        print("="*50)
        print(f"| Test: {base_name}")
        print("="*50)
        print(f"| Status: Error ❌")
        print(f"| Detail: Gagal memproses gambar ini ({type(e).__name__}: {e})")
        print("-"*50)
        continue
