from paddleocr import PaddleOCR
from parser import ReceiptParser
import json
import os

# Model
model = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    # text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Path
IMG_PATH = "img/nota26.jpg"

# Predict
predict = model.predict(IMG_PATH)
for res in predict:
    res.save_to_json("output") 

# Post-processing -> Parsing with Fuzzy Matching
parser = ReceiptParser()

FILE_NAME = "output/nota26_res.json"
with open(FILE_NAME) as f:
    data = json.load(f)
print("="*50)
print(f"| Test: {FILE_NAME}")
print("="*50)

result = parser.parse(data)

if result['items']:
    # for item in result['items']:
        # print(f"✓ {item['item_name']}: {item['qty']}x @ Rp {item['price']:,}")

    print("{:<3} {:<30} {:<6} {:<6}".format(" No.", "Name", "Qty", "Price"))
    for i, item in enumerate(result['items'], 1):
        print(" {:<3} {:<30} {:<6} {:>6,}".format(
            i, item['item_name'], item['qty'], item['price']
        ))
    print("-"*50)
    print(f"| Status: {result['status']}")
    print("-"*50)

    parser.save_parsed_to_json(result['items'], f"results/{os.path.basename(FILE_NAME)}")
else:
    print(f"| Status: Error ❌")
    print(f"| Detail: Tidak ada item yang terdeteksi.")
    print("-"*50)
