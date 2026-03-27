"""
Debug script untuk test inference step by step
"""

import json
from ml_receipt_parser import MLReceiptParser
from paddleocr import PaddleOCR

# Load model
print("Loading model...")
parser = MLReceiptParser(model_path='models/receipt_parser_rf.pkl')
print("Model loaded successfully!\n")

# Test dengan OCR result yang sudah ada
print("="*60)
print("TEST 1: Using existing OCR result")
print("="*60)

# Load OCR result from file
ocr_file = 'output/nota9_res.json'
print(f"Loading: {ocr_file}")

with open(ocr_file, 'r', encoding='utf-8') as f:
    ocr_result = json.load(f)

print(f"OCR Result keys: {list(ocr_result.keys())}")
print(f"Number of texts: {len(ocr_result.get('rec_texts', []))}")
print(f"Number of boxes: {len(ocr_result.get('rec_boxes', []))}")
print(f"Number of scores: {len(ocr_result.get('rec_scores', []))}")

# Try parsing
print("\nParsing...")
result = parser.parse(ocr_result)

print(f"\nStatus: {result['status']}")
print(f"Total items: {result['total_items']}")

if 'error' in result:
    print(f"\nError: {result['error']}")
    if 'traceback' in result:
        print("\nTraceback:")
        print(result['traceback'])

if 'debug_info' in result:
    print(f"\nDebug Info:")
    for key, value in result['debug_info'].items():
        print(f"  {key}: {value}")

if result['items']:
    print("\nItems found:")
    for i, item in enumerate(result['items'], 1):
        print(f"  {i}. {item['item_name']}: {item['qty']}x @ Rp {item['price']:,}")

# Test dengan fresh OCR
print("\n" + "="*60)
print("TEST 2: Fresh OCR from image")
print("="*60)

image_path = 'img/nota9.jpg'
print(f"Processing: {image_path}")

# Run OCR
print("Running OCR...")
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

predict_results = ocr.predict(image_path)
print(f"OCR completed. Results type: {type(predict_results)}")

if isinstance(predict_results, list):
    ocr_obj = predict_results[0]
    print(f"First result type: {type(ocr_obj)}")
    print(f"Has __dict__: {hasattr(ocr_obj, '__dict__')}")
    
    # Try to get attributes
    if hasattr(ocr_obj, '__dict__'):
        print(f"Attributes: {list(vars(ocr_obj).keys())}")
    
    # Convert to dict
    if hasattr(ocr_obj, 'rec_texts'):
        ocr_dict = {
            'rec_texts': ocr_obj.rec_texts,
            'rec_boxes': ocr_obj.rec_boxes,
            'rec_scores': ocr_obj.rec_scores,
        }
        print(f"\nConverted to dict successfully")
        print(f"Number of texts: {len(ocr_dict['rec_texts'])}")
        
        # Parse
        print("\nParsing...")
        result2 = parser.parse(ocr_dict)
        
        print(f"Status: {result2['status']}")
        print(f"Total items: {result2['total_items']}")
        
        if 'error' in result2:
            print(f"\nError: {result2['error']}")
        
        if 'debug_info' in result2:
            print(f"\nDebug Info:")
            for key, value in result2['debug_info'].items():
                print(f"  {key}: {value}")
        
        if result2['items']:
            print("\nItems found:")
            for i, item in enumerate(result2['items'], 1):
                print(f"  {i}. {item['item_name']}: {item['qty']}x @ Rp {item['price']:,}")
    else:
        print("Cannot find rec_texts attribute")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
