"""
Interactive Annotation Tool untuk labeling OCR data
Tool ini memudahkan proses annotation manual untuk training data
"""

import json
import os
from typing import Dict, List
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)


class AnnotationTool:
    """Interactive tool untuk annotation OCR data"""
    
    def __init__(self):
        self.label_colors = {
            'item_name': Fore.GREEN,
            'qty': Fore.CYAN,
            'price': Fore.YELLOW,
            'ignore': Fore.RED
        }
    
    def annotate_receipt(self, ocr_file: str, output_file: str):
        """
        Annotate satu receipt secara interaktif
        
        Args:
            ocr_file: path ke file OCR result JSON
            output_file: path output untuk annotated data
        """
        # Load OCR result
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_result = json.load(f)
        
        texts = ocr_result.get('rec_texts', [])
        boxes = ocr_result.get('rec_boxes', [])
        image_width = ocr_result.get("image_width", 0)
        image_height = ocr_result.get("image_height", 0)
        
        if not texts:
            print(f"{Fore.RED}Error: No text found in OCR result")
            return
        
        print(f"\n{Back.BLUE}{Fore.WHITE} ANNOTATION TOOL {Style.RESET_ALL}")
        print(f"File: {ocr_file}")
        print(f"Image Size : {image_width} x {image_height}")
        print(f"Total elements: {len(texts)}\n")
        print("Labels:")
        print(f"  {Fore.GREEN}1{Style.RESET_ALL} = item_name (nama produk)")
        print(f"  {Fore.CYAN}2{Style.RESET_ALL} = qty (kuantitas)")
        print(f"  {Fore.YELLOW}3{Style.RESET_ALL} = price (harga)")
        print(f"  {Fore.RED}4{Style.RESET_ALL} = ignore (diabaikan)")
        print(f"\nCommands: {Fore.MAGENTA}s{Style.RESET_ALL}=save, {Fore.MAGENTA}q{Style.RESET_ALL}=quit, {Fore.MAGENTA}u{Style.RESET_ALL}=undo\n")
        print("="*60)
        
        history = []

        # Resume annotation jika file sudah ada
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            annotations = saved_data.get("annotations", [])
            if annotations:
                i = max(a["index"] for a in annotations) + 1
            else:
                i = 0

        else:
            annotations = []
            i = 0

        if i >= len(texts):
            print(
                f"{Fore.GREEN}"
                "This file has already been fully annotated."
                f"{Style.RESET_ALL}"
            )
            return

        print(
            f"{Fore.CYAN}"
            f"Progress : {len(annotations)}/{len(texts)} "
            f"({len(annotations)/len(texts)*100:.1f}%)"
            f"{Style.RESET_ALL}"
        )

        if len(annotations) > 0:
            print(
                f"{Fore.YELLOW}"
                f"Resume annotation from element {i+1}"
                f"{Style.RESET_ALL}"
            )

        while i < len(texts):
            text = texts[i]
            box  = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
            x    = box[0]
            y    = box[1]
            w    = box[2] - box[0]
            h    = box[3] - box[1]
            
            # Display current element
            print(f"\n[{i+1}/{len(texts)}] {Fore.WHITE}{Style.BRIGHT}{text}{Style.RESET_ALL}")
            print(f"Position : x={x}, y={y}, w={w}, h={h}")

            if image_width > 0 and image_height > 0:
                print(
                    f"Relative : "
                    f"x={x/image_width:.3f}, "
                    f"y={y/image_height:.3f}, "
                    f"w={w/image_width:.3f}, "
                    f"h={h/image_height:.3f}"
                )
            
            # Show context (previous & next)
            if i > 0:
                print(f"  Prev: {Fore.LIGHTBLACK_EX}{texts[i-1]}{Style.RESET_ALL}")
            if i < len(texts) - 1:
                print(f"  Next: {Fore.LIGHTBLACK_EX}{texts[i+1]}{Style.RESET_ALL}")
            
            # Get label
            label_input = input(f"\nLabel (1-4 / s / q / u): ").strip().lower()
            
            if label_input == 'q':
                self._save_annotations(ocr_result, annotations, output_file)
                print(f"\n{Fore.GREEN}Progress saved. Exiting annotation.{Style.RESET_ALL}")
                return False
            
            elif label_input == 's':
                self._save_annotations(ocr_result, annotations, output_file)
                print(f"\n{Fore.GREEN}Saved! Continue annotating...{Style.RESET_ALL}")
                # i += 1
                continue
            
            elif label_input == 'u':
                if history:
                    i, annotations = history.pop()
                    print(f"{Fore.YELLOW}Undo: back to element {i+1}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Nothing to undo{Style.RESET_ALL}")
                continue
            
            # Map input to label
            label_map = {
                '1': 'item_name',
                '2': 'qty',
                '3': 'price',
                '4': 'ignore'
            }
            
            if label_input in label_map:
                label = label_map[label_input]
                
                # Save to history for undo
                history.append((i, annotations.copy()))
                
                # Add annotation
                annotations.append({
                    'index': i,
                    'text': text,
                    'label': label
                })
                
                color = self.label_colors[label]
                print(f"{color}✓ Labeled as: {label}{Style.RESET_ALL}")
                i += 1
            else:
                print(f"{Fore.RED}Invalid input! Use 1-4, s, q, or u{Style.RESET_ALL}")
        
        # Save final result
        self._save_annotations(ocr_result, annotations, output_file)
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"Annotation completed! ✅")
        print(f"Saved to: {output_file}")
        print(f"{'='*60}{Style.RESET_ALL}")
        return True
    
    def _save_annotations(self, ocr_result: Dict, annotations: List[Dict], output_file: str):
        """Save annotations to JSON file"""
        annotated_data = {
            'ocr_result': ocr_result,
            'annotations': annotations
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotated_data, f, indent=2, ensure_ascii=False)
    
    def show_annotation_summary(self, annotation_file: str):
        """Display summary dari annotated file"""
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        
        # Count labels
        label_counts = {}
        for ann in annotations:
            label = ann['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n{Back.BLUE}{Fore.WHITE} ANNOTATION SUMMARY {Style.RESET_ALL}")
        print(f"File: {annotation_file}")
        print(f"Total annotations: {len(annotations)}\n")
        
        for label, count in sorted(label_counts.items()):
            color = self.label_colors.get(label, Fore.WHITE)
            print(f"  {color}{label:12s}: {count:3d}{Style.RESET_ALL}")
        
        print()


class BatchAnnotator:
    """Batch annotation untuk multiple files"""
    
    def __init__(self):
        self.tool = AnnotationTool()
    
    def annotate_batch(self, ocr_dir: str, output_dir: str):
        """
        Annotate semua file dalam directory
        
        Args:
            ocr_dir: directory berisi OCR result JSON files
            output_dir: directory output untuk annotated files
        """
        # Get all JSON files
        json_files = [f for f in os.listdir(ocr_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"{Fore.RED}No JSON files found in {ocr_dir}{Style.RESET_ALL}")
            return
        
        print(f"\n{Back.GREEN}{Fore.BLACK} BATCH ANNOTATION {Style.RESET_ALL}")
        print(f"Found {len(json_files)} files in {ocr_dir}")
        print(f"Output directory: {output_dir}\n")

        stopped = False
        
        for i, filename in enumerate(json_files, 1):
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"Processing file {i}/{len(json_files)}: {filename}")
            print(f"{'='*60}{Style.RESET_ALL}")
            
            ocr_file = os.path.join(ocr_dir, filename)
            output_file = os.path.join(output_dir, filename.replace('.json', '_annotated.json'))
            
            # Check if already annotated
            if os.path.exists(output_file):
                # Load annotation yang sudah ada
                with open(output_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)

                annotations = saved_data.get("annotations", [])

                # Load OCR untuk mengetahui total elemen
                with open(ocr_file, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)

                total_elements = len(ocr_data.get("rec_texts", []))

                # Jika sudah selesai, skip otomatis
                if len(annotations) >= total_elements:
                    print(f"{Fore.GREEN}File '{filename}' has already been fully annotated.{Style.RESET_ALL}")
                    continue

                # Jika belum selesai, lanjut otomatis
                print(f"{Fore.CYAN}Resuming {filename}{Style.RESET_ALL}")
            
            try:
                result = self.tool.annotate_receipt(ocr_file, output_file)
                if result is False:
                    stopped = True
                    print(f"{Fore.YELLOW}Batch annotation stopped.{Style.RESET_ALL}")
                    break
            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Batch annotation interrupted{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error processing {filename}: {e}{Style.RESET_ALL}")
                continue
        
        if not stopped:
            print(f"\n{Fore.GREEN}Batch annotation completed!{Style.RESET_ALL}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python annotation_tool.py single <ocr_file> <output_file>")
        print("  Batch mode:  python annotation_tool.py batch <ocr_dir> <output_dir>")
        print("  Summary:     python annotation_tool.py summary <annotation_file>")
        sys.exit(1)
    
    mode = sys.argv[1]
    tool = AnnotationTool()
    
    if mode == "single":
        if len(sys.argv) < 4:
            print("Error: Provide ocr_file and output_file")
            sys.exit(1)
        ocr_file = sys.argv[2]
        output_file = sys.argv[3]
        tool.annotate_receipt(ocr_file, output_file)
    
    elif mode == "batch":
        if len(sys.argv) < 4:
            print("Error: Provide ocr_dir and output_dir")
            sys.exit(1)
        ocr_dir = sys.argv[2]
        output_dir = sys.argv[3]
        batch_tool = BatchAnnotator()
        batch_tool.annotate_batch(ocr_dir, output_dir)
    
    elif mode == "summary":
        if len(sys.argv) < 3:
            print("Error: Provide annotation_file")
            sys.exit(1)
        annotation_file = sys.argv[2]
        tool.show_annotation_summary(annotation_file)
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)