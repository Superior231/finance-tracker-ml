"""
Inference Application untuk Receipt Parser
Menggunakan trained Random Forest model
"""

import os
import json
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from typing import Dict, List
from paddleocr import PaddleOCR
from ml_receipt_parser import MLReceiptParser
from colorama import Fore, Back, Style, init


init(autoreset=True)


class ReceiptInferenceApp:
    """Application untuk inference menggunakan trained model"""
    
    def __init__(self, model_path: str):
        """
        Initialize inference app
        
        Args:
            model_path: path ke trained model
        """
        print(f"\n{Back.BLUE}{Fore.WHITE} INITIALIZING RECEIPT PARSER {Style.RESET_ALL}\n")
        
        # Load OCR model
        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_unclip_ratio=1.2,                          # Diturunkan agar box tidak gampang nyambung
            enable_mkldnn=True,                                 # Mengakselerasi komputasi pada CPU Intel/AMD
        )
        print(f"{Fore.GREEN}✓ PaddleOCR loaded{Style.RESET_ALL}")
        
        # Load ML parser
        print(f"Loading trained model from: {model_path}")
        self.parser = MLReceiptParser(model_path=model_path)
        print(f"{Fore.GREEN}✓ ML Parser loaded{Style.RESET_ALL}\n")
    
    def process_image(self, image_path: str, save_ocr: bool = True, 
                     output_dir: str = "output") -> Dict:
        """
        Process receipt image end-to-end
        
        Args:
            image_path: path ke gambar receipt
            save_ocr: apakah menyimpan OCR result
            output_dir: directory untuk save hasil
            
        Returns:
            Dictionary berisi items dan metadata
        """
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}Processing: {image_path}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # Step 1: OCR
        print("Step 1: Running OCR...")
        predict_results = self.ocr.predict(image_path)
        
        if not predict_results:
            print(f"{Fore.RED}Error: OCR failed{Style.RESET_ALL}")
            return {'status': 'ocr_failed', 'items': []}
        
        # Extract OCR result - PaddleOCR returns object, not dict
        if isinstance(predict_results, list):
            ocr_obj = predict_results[0]
        else:
            ocr_obj = predict_results
        
        # Try different methods to extract data from OCR object
        ocr_result = None
        
        # Method 1: Check if it has to_dict() method
        if hasattr(ocr_obj, 'to_dict'):
            ocr_result = ocr_obj.to_dict()
        
        # Method 2: Try direct attribute access
        elif hasattr(ocr_obj, 'rec_texts'):
            ocr_result = {
                'rec_texts': ocr_obj.rec_texts,
                'rec_boxes': ocr_obj.rec_boxes if hasattr(ocr_obj, 'rec_boxes') else [],
                'rec_scores': ocr_obj.rec_scores if hasattr(ocr_obj, 'rec_scores') else [],
                'input_path': image_path
            }
        
        # Method 3: Try to access via dir() to find hidden attributes
        elif hasattr(ocr_obj, '__class__'):
            # Get all non-private attributes
            attrs = [a for a in dir(ocr_obj) if not a.startswith('_')]
            
            # Look for rec_texts in attributes
            if 'rec_texts' in attrs:
                ocr_result = {
                    'rec_texts': getattr(ocr_obj, 'rec_texts', []),
                    'rec_boxes': getattr(ocr_obj, 'rec_boxes', []),
                    'rec_scores': getattr(ocr_obj, 'rec_scores', []),
                    'input_path': image_path
                }
            else:
                # Try to save to JSON and reload
                temp_dir = 'temp_ocr'
                os.makedirs(temp_dir, exist_ok=True)
                ocr_obj.save_to_json(temp_dir)
                
                # Find the saved JSON file
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                json_files = [f for f in os.listdir(temp_dir) if f.startswith(base_name) and f.endswith('.json')]
                
                if json_files:
                    with open(os.path.join(temp_dir, json_files[0]), 'r', encoding='utf-8') as f:
                        ocr_result = json.load(f)
                    # Clean up temp file
                    os.remove(os.path.join(temp_dir, json_files[0]))
        
        # Method 4: Already a dict
        elif isinstance(ocr_obj, dict):
            ocr_result = ocr_obj
        
        # If all methods failed
        if ocr_result is None or 'rec_texts' not in ocr_result:
            print(f"{Fore.RED}Error: Cannot extract OCR data from result object{Style.RESET_ALL}")
            return {'status': 'ocr_extraction_failed', 'items': []}
        
        texts = ocr_result.get('rec_texts', [])
        print(f"{Fore.GREEN}✓ OCR completed: {len(texts)} text elements detected{Style.RESET_ALL}")
        
        # Save OCR result if requested
        if save_ocr:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            ocr_filename = os.path.splitext(base_name)[0] + '_ocr.json'
            ocr_path = os.path.join(output_dir, ocr_filename)
            
            with open(ocr_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_result, f, indent=2, ensure_ascii=False)
            
            print(f"  OCR result saved to: {ocr_path}")
        
        # Step 2: ML Parsing
        print("\nStep 2: Parsing with ML model...")
        result = self.parser.parse(ocr_result)
        
        if result['status'] == 'Success✅':
            print(f"{Fore.GREEN}✓ Parsing completed: {result['total_items']} items found{Style.RESET_ALL}")
        elif result['status'] == 'error':
            print(f"{Fore.RED}✗ Parsing error: {result.get('error', 'Unknown error')}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ Parsing status: {result['status']}{Style.RESET_ALL}")
        
        return result
    
    def display_results(self, result: Dict):
        """
        Display parsing results in formatted table
        
        Args:
            result: result dictionary dari parse()
        """
        # Show debug info if error
        if result['status'] == 'error':
            print(f"\n{Fore.RED}{'='*60}")
            print("ERROR DETAILS")
            print(f"{'='*60}{Style.RESET_ALL}")
            print(f"Error: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(f"\n{Fore.YELLOW}Traceback:{Style.RESET_ALL}")
                print(result['traceback'])
            return
        
        # Show debug info
        if 'debug_info' in result:
            debug = result['debug_info']
            print(f"\n{Fore.CYAN}Debug Info:{Style.RESET_ALL}")
            print(f"  Total predictions: {debug.get('total_predictions', 0)}")
            print(f"  Items before validation: {debug.get('total_items_before_validation', 0)}")
            print(f"  Label distribution: {debug.get('label_distribution', {})}")
        
        if not result['items']:
            print(f"\n{Fore.YELLOW}No valid items found{Style.RESET_ALL}")
            if result.get('status') == 'no_items_found':
                print("\nPossible reasons:")
                print("  - All predictions were 'ignore' label")
                print("  - Items didn't pass validation (check qty/price ranges)")
                print("  - Model needs more training data")
            return
        
        print(f"\n{Back.GREEN}{Fore.BLACK} PARSING RESULTS {Style.RESET_ALL}")
        print(f"\nTotal Items: {result['total_items']}")
        print(f"Status: {result['status']}\n")
        
        # # Table header
        # print(f"{Fore.CYAN}{'No.':<5} {'Item Name':<35} {'Qty':<6} {'Price':>12}{Style.RESET_ALL}")
        # print("-" * 60)
        
        # # Table rows
        # total_price = 0
        # for i, item in enumerate(result['items'], 1):
        #     item_name = item['item_name'][:33] + '..' if len(item['item_name']) > 35 else item['item_name']
        #     qty = item['qty']
        #     price = item['price']
        #     subtotal = qty * price
        #     total_price += subtotal
            
        #     print(f"{i:<5} {item_name:<35} {qty:<6} Rp {price:>10,}")
        
        # print("-" * 60)

        # Define column widths
        col_no = 5
        col_name = 35
        col_qty = 6
        col_price = 10
        col_sub = 10
        
        # Table header
        print(
            f"{Fore.CYAN}"
            f"{'No.':<{col_no}}  "
            f"{'Item Name':<{col_name}}  "
            f"{'Qty':<{col_qty}}  "
            f"{'Price':>{col_price}}  "
            f"{'Subtotal':>{col_sub}}"
            f"{Style.RESET_ALL}"
        )

        print("-" * (col_no + col_name + col_qty + col_price + col_sub + 10))
        
        # Table rows
        total_price = 0
        for i, item in enumerate(result['items'], 1):
            name = item["item_name"]
            if len(name) > col_name:
                name = name[:col_name - 2] + ".."

            qty = item["qty"]
            price = item["price"]
            subtotal = qty * price
            total_price += subtotal

            print(
                f"{i:<{col_no}}  "
                f"{name:<{col_name}}  "
                f"{qty:<{col_qty}}  "
                f"Rp{price:>{col_price-2},}  "
                f"Rp{subtotal:>{col_sub-2},}"
            )
        
        # Footer
        print("-" * (col_no + col_name + col_qty + col_price + col_sub + 10))
        print(f"{'Total:'}{Fore.GREEN} Rp {total_price:,}{Style.RESET_ALL}")
    
    def save_results(self, result: Dict, output_path: str):
        """
        Save parsing results to JSON
        
        Args:
            result: result dictionary
            output_path: output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result['items'], f, indent=2, ensure_ascii=False)
        
        print(f"\n{Fore.GREEN}✓ Results saved to: {output_path}{Style.RESET_ALL}")
    
    def batch_process(self, image_dir: str, output_dir: str = "results"):
        """
        Process multiple images in batch
        
        Args:
            image_dir: directory berisi gambar-gambar receipt
            output_dir: directory untuk save hasil
        """
        print(f"\n{Back.MAGENTA}{Fore.WHITE} BATCH PROCESSING {Style.RESET_ALL}\n")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [
            f for f in os.listdir(image_dir) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            print(f"{Fore.RED}No images found in {image_dir}{Style.RESET_ALL}")
            return
        
        print(f"Found {len(image_files)} images\n")
        
        results_summary = []
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, filename)
            
            print(f"\n[{i}/{len(image_files)}] {filename}")
            
            # Process image
            result = self.process_image(image_path, save_ocr=True, output_dir=output_dir)
            
            # Display results
            self.display_results(result)
            
            # Save results
            result_filename = os.path.splitext(filename)[0] + '_parsed.json'
            result_path = os.path.join(output_dir, result_filename)
            self.save_results(result, result_path)
            
            # Add to summary
            results_summary.append({
                'filename': filename,
                'status': result['status'],
                'total_items': result['total_items'],
                'items': result['items']
            })
        
        # Save batch summary
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'='*60}{Style.RESET_ALL}")
        print(f"Processed: {len(image_files)} images")
        print(f"Results saved to: {output_dir}")
        print(f"Summary saved to: {summary_path}")


def compare_with_rule_based(ml_result: Dict, rule_based_result: Dict):
    """
    Compare ML-based result with rule-based result
    
    Args:
        ml_result: hasil dari ML parser
        rule_based_result: hasil dari rule-based parser
    """
    print(f"\n{Back.YELLOW}{Fore.BLACK} COMPARISON: ML vs RULE-BASED {Style.RESET_ALL}\n")
    
    ml_items = ml_result.get('items', [])
    rb_items = rule_based_result.get('items', [])
    
    print(f"ML Parser:         {len(ml_items)} items")
    print(f"Rule-based Parser: {len(rb_items)} items")
    
    # Find matching and different items
    ml_set = set((item['item_name'], item['qty'], item['price']) for item in ml_items)
    rb_set = set((item['item_name'], item['qty'], item['price']) for item in rb_items)
    
    matching = ml_set & rb_set
    ml_only = ml_set - rb_set
    rb_only = rb_set - ml_set
    
    print(f"\nMatching items:    {len(matching)}")
    print(f"ML only:           {len(ml_only)}")
    print(f"Rule-based only:   {len(rb_only)}")
    
    if ml_only:
        print(f"\n{Fore.CYAN}Items found by ML only:{Style.RESET_ALL}")
        for item in ml_only:
            print(f"  - {item[0]}: {item[1]}x @ Rp {item[2]:,}")
    
    if rb_only:
        print(f"\n{Fore.MAGENTA}Items found by Rule-based only:{Style.RESET_ALL}")
        for item in rb_only:
            print(f"  - {item[0]}: {item[1]}x @ Rp {item[2]:,}")
    
    # Accuracy calculation
    if len(rb_items) > 0:
        accuracy = len(matching) / len(rb_items) * 100
        print(f"\n{Fore.GREEN}Agreement: {accuracy:.1f}%{Style.RESET_ALL}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    MODEL_PATH = "models/receipt_parser_rf.pkl"
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           RECEIPT PARSER - INFERENCE APPLICATION              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"{Fore.RED}Error: Model not found at {MODEL_PATH}{Style.RESET_ALL}")
        print("\nPlease train the model first:")
        print("  python train_pipeline.py")
        sys.exit(1)
    
    # Initialize app
    app = ReceiptInferenceApp(MODEL_PATH)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python app.py single <image_path>")
        print("  Batch mode:   python app.py batch <image_dir>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "single":
        if len(sys.argv) < 3:
            print(f"{Fore.RED}Error: Please provide image path{Style.RESET_ALL}")
            sys.exit(1)
        
        image_path = sys.argv[2]
        
        if not os.path.exists(image_path):
            print(f"{Fore.RED}Error: Image not found: {image_path}{Style.RESET_ALL}")
            sys.exit(1)
        
        # Process single image
        result = app.process_image(image_path)
        app.display_results(result)
        
        # Save results
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_parsed.json'
        output_path = os.path.join('results', output_filename)
        app.save_results(result, output_path)
    
    elif mode == "batch":
        if len(sys.argv) < 3:
            print(f"{Fore.RED}Error: Please provide image directory{Style.RESET_ALL}")
            sys.exit(1)
        
        image_dir = sys.argv[2]
        
        if not os.path.exists(image_dir):
            print(f"{Fore.RED}Error: Directory not found: {image_dir}{Style.RESET_ALL}")
            sys.exit(1)
        
        # Batch processing
        app.batch_process(image_dir, output_dir='results')
    
    else:
        print(f"{Fore.RED}Unknown mode: {mode}{Style.RESET_ALL}")
        print("Use 'single' or 'batch'")
        sys.exit(1)
