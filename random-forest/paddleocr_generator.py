# Generate JSON from PaddleOCR
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import json
from paddleocr import PaddleOCR


model = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_unclip_ratio=1.2,
    enable_mkldnn=True,
)

# Single
# img_path = 'datasets/train/nota6.jpg'
# predict = model.predict(img_path)
# for res in predict:
#     res.save_to_json('output')
#     res.save_to_img("output")

# Batch
# for img_file in os.listdir('datasets/train/'):
#     if img_file.endswith(('.jpg', '.jpeg', '.png')):
#         img_path = os.path.join('datasets/train', img_file)
#         predict = model.predict(img_path)
#         for res in predict:
#             res.save_to_json('output')

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir("datasets/train"):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join("datasets/train", img_file)

        # Baca ukuran gambar
        img = cv2.imread(img_path)
        image_height, image_width = img.shape[:2]

        # OCR
        predict = model.predict(img_path)

        for res in predict:
            result = res.json["res"]
            result["image_width"]  = image_width
            result["image_height"] = image_height

            json_name = os.path.splitext(img_file)[0] + "_res.json"

            with open(
                os.path.join(output_dir, json_name),
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        print(f"Saved: {json_name}")
