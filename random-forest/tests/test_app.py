"""
Simple test untuk debug inference issue
Run: python test_app.py
"""

import json
import os
from ml_receipt_parser import MLReceiptParser

print("="*60)
print("TESTING ML RECEIPT PARSER")
print("="*60)

# Step 1: Load model
print("\n1. Loading model...")
try:
    parser = MLReceiptParser(model_path='models/receipt_parser_rf.pkl')
    parser.verbose = True  # Enable debug mode
    print("✓ Model loaded successfully")
    print(f"  Feature names: {len(parser.feature_names)} features")
    print(f"  Classes: {parser.label_encoder.classes_}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Step 2: Load OCR result
print("\n2. Loading OCR result...")
ocr_file = 'output/nota9_res.json'

if not os.path.exists(ocr_file):
    print(f"✗ File not found: {ocr_file}")
    print("Please run PaddleOCR first to generate OCR results")
    exit(1)

with open(ocr_file, 'r', encoding='utf-8') as f:
    ocr_result = json.load(f)

print(f"✓ OCR result loaded")
print(f"  Keys: {list(ocr_result.keys())}")
print(f"  Texts: {len(ocr_result.get('rec_texts', []))}")
print(f"  Boxes: {len(ocr_result.get('rec_boxes', []))}")
print(f"  Scores: {len(ocr_result.get('rec_scores', []))}")

# Step 3: Test prediction
print("\n3. Testing prediction...")
print("-"*60)

try:
    result = parser.parse(ocr_result)
    
    print(f"\n✓ Parsing completed")
    print(f"  Status: {result['status']}")
    print(f"  Total items: {result['total_items']}")
    
    if 'error' in result:
        print(f"\n✗ Error occurred:")
        print(f"  {result['error']}")
        if 'traceback' in result:
            print("\nFull traceback:")
            print(result['traceback'])
    
    if 'debug_info' in result:
        print(f"\nDebug Information:")
        debug = result['debug_info']
        print(f"  Total predictions: {debug.get('total_predictions', 0)}")
        print(f"  Items before validation: {debug.get('total_items_before_validation', 0)}")
        print(f"  Label distribution: {debug.get('label_distribution', {})}")
    
    if result['items']:
        print(f"\n✓ Items found: {len(result['items'])}")
        print("\nParsed Items:")
        print("-"*60)
        for i, item in enumerate(result['items'], 1):
            print(f"{i:2d}. {item['item_name']:<30} | Qty: {item['qty']:3d} | Price: Rp {item['price']:>10,}")
        print("-"*60)
    else:
        print("\n⚠ No valid items found")
        print("\nPossible reasons:")
        print("  1. Model predicted all elements as 'ignore'")
        print("  2. Items failed validation (qty/price out of range)")
        print("  3. Model needs retraining with better annotated data")
        
        # Show what was predicted
        if 'debug_info' in result:
            dist = result['debug_info'].get('label_distribution', {})
            if dist:
                print("\nLabel distribution from predictions:")
                for label, count in dist.items():
                    print(f"  {label}: {count}")

except Exception as e:
    print(f"\n✗ Error during parsing:")
    print(f"  {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)

print("\nNext steps:")
print("1. Check if model was trained with correct data")
print("2. Verify annotation quality in training data")
print("3. Check if validation thresholds are too strict")
print("4. Enable verbose mode to see detailed predictions")