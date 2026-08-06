"""
ML-based Receipt Parser using Random Forest
Untuk skripsi: Post-processing OCR hasil PaddleOCR dengan Machine Learning
"""

import re
import json
import pickle
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Random Forest Only
class PureFeatureExtractor:
    """
    Feature extractor TANPA rule-based heuristic.
    Hanya statistik teks, layout, dan konteks spasial.
    """

    def extract_features(
            self,
            element: dict,
            prev_element: dict | None,
            next_element: dict | None,
            line_elements: list[dict]
        ) -> dict:

        text = element["text"]
        img_w = max(element["image_width"], 1)
        img_h = max(element["image_height"], 1)

        features = {
            # Text Features
            "text_length": len(text),
            "num_words": len(text.split()),
            "num_digits": sum(c.isdigit() for c in text),
            "num_alpha": sum(c.isalpha() for c in text),
            "num_special": sum(not c.isalnum() and not c.isspace() for c in text),

            "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
            "alpha_ratio": sum(c.isalpha() for c in text) / max(len(text), 1),
            "upper_ratio": sum(c.isupper() for c in text) / max(len(text), 1),

            # Position Features
            "x_pos": element["x"],
            "y_pos": element["y"],
            "width": element["width"],
            "height": element["height"],
            "center_x": element["center_x"],
            "center_y": element["center_y"],
            "aspect_ratio": element["width"] / max(element["height"], 1),

            # Relative Position Features
            "relative_x": element["x"] / img_w,
            "relative_y": element["y"] / img_h,
            "relative_x2": element["x2"] / img_w,
            "relative_y2": element["y2"] / img_h,

            "relative_center_x": element["center_x"] / img_w,
            "relative_center_y": element["center_y"] / img_h,

            "relative_width": element["width"] / img_w,
            "relative_height": element["height"] / img_h,

            "relative_dist_left": element["x"] / img_w,
            "relative_dist_right": (img_w - element["x2"]) / img_w,

            "relative_dist_top": element["y"] / img_h,
            "relative_dist_bottom": (img_h - element["y2"]) / img_h,

            # Context Features
            "position_in_line": (
                line_elements.index(element)
                if element in line_elements else 0
            ),
            "line_length": len(line_elements),

            # Distance to neighbors
            "dist_to_prev": (
                abs(element["x"] - prev_element["x2"])
                if prev_element else 0
            ),
            "relative_dist_prev": (
                abs(element["x"] - prev_element["x2"]) / img_w
                if prev_element else 0
            ),
            "dist_to_next": (
                abs(next_element["x"] - element["x2"])
                if next_element else 0
            ),
            "relative_dist_next": (
                abs(next_element["x"] - element["x2"]) / img_w
                if next_element else 0
            ),
            "y_diff_prev": (
                abs(element["center_y"] - prev_element["center_y"])
                if prev_element else 0
            ),
            "relative_y_diff_prev": (
                abs(element["center_y"] - prev_element["center_y"]) / img_h
                if prev_element else 0
            ),
            "y_diff_next": (
                abs(element["center_y"] - next_element["center_y"])
                if next_element else 0
            ),
            "relative_y_diff_next": (
                abs(element["center_y"] - next_element["center_y"]) / img_h
                if next_element else 0
            ),

            # OCR Confidence
            "ocr_score": element["score"],
        }

        return features


# Random Forest + Rule-Based
class FeatureExtractor:
    """
    Ekstraksi fitur dari raw OCR data untuk training Random Forest
    """
    
    def __init__(self):
        self.price_patterns = [
            r'^\d{3,}$',
            r'^\d{1,3}[,\.]\d{3}$',
            r'^\d{1,3}[,\.]\d{3}[,\.]\d{3}$'
        ]
        self.qty_patterns = [r'^\d{1,3}$', r'^\d+x$', r'^x\d+$']

        # Relative horizontal area boundaries
        LEFT_AREA = 0.33
        RIGHT_AREA = 0.66
        
        # Kata kunci untuk kategori
        self.product_keywords = [
            'tea', 'coffee', 'milk', 'juice', 'water', 'ice', 'fruit',
            'indomie', 'mie', 'goreng', 'soto', 'bakso', 'ayam', 'sapi',
            'steak', 'bone', 'sirloin', 'ribeye', 'salad', 'burger',
            'kg', 'gr', 'ml', 'ltr', 'pcs', 'box', 'pack', 'btl'
        ]
        
        self.ignore_keywords = [
            'total', 'subtotal', 'cash', 'tunai', 'kembali', 'bayar',
            'npwp', 'telp', 'jalan', 'jl.', 'alamat', 'terima kasih',
            'thank', 'diskon', 'discount', 'voucher', 'cancel',
            'dana', 'ovo', 'gopay', 'ovo', 'shopeepay',
        ]
    
    def extract_features(
            self, element: Dict,
            prev_element: Optional[Dict], 
            next_element: Optional[Dict],
            line_elements: List[Dict]
        ) -> Dict:
        """
        Ekstraksi fitur dari satu elemen OCR
        
        Features:
        1. Text-based features: panjang, numeric ratio, alpha ratio
        2. Position features: x, y, width, height, center positions
        3. Context features: jarak ke elemen sebelah, posisi dalam baris
        4. Pattern features: match dengan pattern harga/qty/produk
        5. Confidence: score OCR
        """
        text = element['text']
        text_lower = text.lower()
        img_w = max(element["image_width"], 1)
        img_h = max(element["image_height"], 1)
        
        features = {
            # Text Features
            'text_length': len(text),
            'num_words': len(text.split()),
            'num_digits': sum(c.isdigit() for c in text),
            'num_alpha': sum(c.isalpha() for c in text),
            'num_special': sum(not c.isalnum() and not c.isspace() for c in text),
            'digit_ratio': sum(c.isdigit() for c in text) / max(len(text), 1),
            'alpha_ratio': sum(c.isalpha() for c in text) / max(len(text), 1),
            'upper_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
            
            # Position Features
            'x_pos': element['x'],
            'y_pos': element['y'],
            'width': element['width'],
            'height': element['height'],
            'center_x': element['center_x'],
            'center_y': element['center_y'],
            'aspect_ratio': element['width'] / max(element['height'], 1),

            # Relative Position Features
            "relative_x": element["x"] / img_w,
            "relative_y": element["y"] / img_h,
            "relative_x2": element["x2"] / img_w,
            "relative_y2": element["y2"] / img_h,

            "relative_center_x": element["center_x"] / img_w,
            "relative_center_y": element["center_y"] / img_h,

            "relative_width": element["width"] / img_w,
            "relative_height": element["height"] / img_h,

            "relative_dist_left": element["x"] / img_w,
            "relative_dist_right": (img_w - element["x2"]) / img_w,

            "relative_dist_top": element["y"] / img_h,
            "relative_dist_bottom": (img_h - element["y2"]) / img_h,

            "is_left_area": int(element["center_x"] < img_w * self.LEFT_AREA),
            "is_middle_area": int(img_w * self.LEFT_AREA <= element["center_x"] <= img_w * self.RIGHT_AREA),
            "is_right_area": int(element["center_x"] > img_w * self.RIGHT_AREA),
            
            # Pattern matching features
            'is_price_pattern': int(any(re.match(p, text.replace(',','').replace('.','')) 
                                       for p in self.price_patterns)),
            'is_qty_pattern': int(any(re.match(p, text, re.IGNORECASE) 
                                     for p in self.qty_patterns)),
            'has_x_separator': int('x' in text_lower),
            'has_comma': int(',' in text),
            'has_dot': int('.' in text),
            'starts_with_digit': int(text[0].isdigit() if text else 0),
            'ends_with_digit': int(text[-1].isdigit() if text else 0),
            
            # Keyword features
            'has_product_keyword': int(any(kw in text_lower for kw in self.product_keywords)),
            'has_ignore_keyword': int(any(kw in text_lower for kw in self.ignore_keywords)),
            
            # Contextual features
            'position_in_line': line_elements.index(element) if element in line_elements else 0,
            'line_length': len(line_elements),
            'is_first_in_line': int(line_elements.index(element) == 0 if element in line_elements else False),
            'is_last_in_line': int(line_elements.index(element) == len(line_elements)-1 
                                  if element in line_elements and len(line_elements) > 0 else False),
            
            # Distance to neighbors
            "dist_to_prev": (
                abs(element["x"] - prev_element["x2"])
                if prev_element else 0
            ),
            "relative_dist_prev": (
                abs(element["x"] - prev_element["x2"]) / img_w
                if prev_element else 0
            ),
            "dist_to_next": (
                abs(next_element["x"] - element["x2"])
                if next_element else 0
            ),
            "relative_dist_next": (
                abs(next_element["x"] - element["x2"]) / img_w
                if next_element else 0
            ),
            "y_diff_prev": (
                abs(element["center_y"] - prev_element["center_y"])
                if prev_element else 0
            ),
            "relative_y_diff_prev": (
                abs(element["center_y"] - prev_element["center_y"]) / img_h
                if prev_element else 0
            ),
            "y_diff_next": (
                abs(element["center_y"] - next_element["center_y"])
                if next_element else 0
            ),
            "relative_y_diff_next": (
                abs(element["center_y"] - next_element["center_y"]) / img_h
                if next_element else 0
            ),
            
            # OCR confidence
            'ocr_score': element['score'],
        }
        
        return features


class MLReceiptParser:
    """
    Machine Learning-based Receipt Parser menggunakan Random Forest
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize parser
        
        Args:
            model_path: path ke saved model (jika sudah training)
        """
        self.feature_extractor = PureFeatureExtractor() # RF Only
        # self.feature_extractor = FeatureExtractor()     # RF + Rule-Based
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        # self.line_threshold = 15  # threshold untuk grouping baris
        self.line_threshold_min = 5    # floor px, ganti sesuai hasil kalibrasi kamu
        self.line_threshold_mult = 0.5
        self.verbose = False  # Set True untuk debugging
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _prepare_elements(
            self,
            texts: List[str],
            boxes: List[List[int]],
            scores: List[float],
            image_width: int,
            image_height: int
        ) -> List[Dict]:

        """Prepare OCR elements dengan metadata"""
        elements = []
        for i, text in enumerate(texts):
            text_clean = text.strip()
            if not text_clean:
                continue
            # box = boxes[i] if boxes and i < len(boxes) else [0,0,0,0]
            box = boxes[i] if len(boxes) > 0 and i < len(boxes) else [0, 0, 0, 0]
            score = scores[i] if scores and i < len(scores) else 1.0
            elements.append({
                'index': i,
                'text': text_clean,
                'text_lower': text_clean.lower(),
                'box': box,
                'score': score,
                'x': box[0],
                'y': box[1],
                'x2': box[2],
                'y2': box[3],
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'center_x': (box[0] + box[2]) / 2,
                'center_y': (box[1] + box[3]) / 2,
                'image_width': image_width,
                'image_height': image_height,
            })
        return elements
    
    # def _group_into_lines(self, elements: List[Dict]) -> List[List[Dict]]:
    #     """Group elements ke dalam baris berdasarkan posisi Y"""
    #     if not elements:
    #         return []
        
    #     elements_sorted = sorted(elements, key=lambda x: x['y'])
    #     lines = []
    #     current_line = [elements_sorted[0]]
        
    #     for elem in elements_sorted[1:]:
    #         prev_y = sum(e['center_y'] for e in current_line) / len(current_line)
    #         if abs(elem['center_y'] - prev_y) <= self.line_threshold:
    #             current_line.append(elem)
    #         else:
    #             current_line.sort(key=lambda x: x['x'])
    #             lines.append(current_line)
    #             current_line = [elem]
        
    #     if current_line:
    #         current_line.sort(key=lambda x: x['x'])
    #         lines.append(current_line)
        
    #     return lines

    def _group_into_lines(self, elements: List[Dict]) -> List[List[Dict]]:
        """Group elements ke dalam baris berdasarkan posisi Y (adaptif, sama seperti rule-based)"""
        if not elements:
            return []

        elements_sorted = sorted(elements, key=lambda x: x['y'])
        lines = []
        current_line = [elements_sorted[0]]

        for elem in elements_sorted[1:]:
            prev_y_center = sum(e['center_y'] for e in current_line) / len(current_line)
            avg_height = sum(e['height'] for e in current_line) / len(current_line)
            threshold = max(self.line_threshold_min, avg_height * self.line_threshold_mult)

            if abs(elem['center_y'] - prev_y_center) <= threshold:
                current_line.append(elem)
            else:
                current_line.sort(key=lambda x: x['x'])
                lines.append(current_line)
                current_line = [elem]

        if current_line:
            current_line.sort(key=lambda x: x['x'])
            lines.append(current_line)

        return lines
    
    def prepare_training_data(self, annotated_data: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data untuk training
        
        Args:
            annotated_data: List berisi dict dengan:
                - ocr_result: hasil OCR dari PaddleOCR
                - annotations: list label untuk setiap elemen
                  Format: [{'index': 0, 'label': 'item_name'}, ...]
                  Labels: 'item_name', 'qty', 'price', 'ignore'
        
        Returns:
            X (DataFrame): feature matrix
            y (array): labels
        """
        all_features = []
        all_labels = []
        
        for data in annotated_data:
            ocr_result = data['ocr_result']
            annotations = {ann['index']: ann['label'] for ann in data['annotations']}
            
            texts = ocr_result.get('rec_texts', [])
            boxes = ocr_result.get('rec_boxes', [])
            scores = ocr_result.get('rec_scores', [])
            image_width = ocr_result.get('image_width', [])
            image_height = ocr_result.get('image_height', [])
            
            elements = self._prepare_elements(texts, boxes, scores, image_width, image_height)
            elements.sort(key=lambda x: x['y'])
            lines = self._group_into_lines(elements)
            
            # Extract features untuk setiap element
            for line in lines:
                for i, elem in enumerate(line):
                    prev_elem = line[i-1] if i > 0 else None
                    next_elem = line[i+1] if i < len(line)-1 else None
                    
                    features = self.feature_extractor.extract_features(
                        elem, prev_elem, next_elem, line
                    )
                    
                    label = annotations.get(elem['index'], 'ignore')
                    
                    all_features.append(features)
                    all_labels.append(label)
        
        X = pd.DataFrame(all_features)
        y = np.array(all_labels)
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, 
              random_state: int = 42):
        """
        Train Random Forest model
        
        Args:
            X: feature matrix
            y: labels
            test_size: proporsi data test
            random_state: seed untuk reproducibility
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Train Random Forest
        print("Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Train accuracy: {self.model.score(X_train, y_train):.4f}")
        print(f"Test accuracy: {self.model.score(X_test, y_test):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
        print("="*60)
    
    def predict_elements(self, ocr_result: Dict) -> List[Dict]:
        """
        Prediksi label untuk setiap elemen dalam OCR result
        
        Returns:
            List of dict dengan format:
            {'element': Dict, 'label': str, 'probability': float}
        """
        if self.model is None:
            raise ValueError("Model belum di-train! Gunakan train() atau load_model()")
        
        texts = ocr_result.get('rec_texts', [])
        boxes = ocr_result.get('rec_boxes', [])
        scores = ocr_result.get('rec_scores', [])
        image_width = ocr_result.get('image_width', [])
        image_height = ocr_result.get('image_height', [])
        
        if self.verbose:
            print(f"[DEBUG] predict_elements called")
            print(f"[DEBUG] texts: {len(texts)}, boxes: {len(boxes)}, scores: {len(scores)}")

        elements = self._prepare_elements(texts, boxes, scores, image_width, image_height)
        if self.verbose:
            print(f"[DEBUG] prepared elements: {len(elements)}")
        
        elements.sort(key=lambda x: x['y'])
        lines = self._group_into_lines(elements)
        if self.verbose:
            print(f"[DEBUG] grouped into {len(lines)} lines")
        
        predictions = []
        
        for line_idx, line in enumerate(lines):
            for i, elem in enumerate(line):
                prev_elem = line[i-1] if i > 0 else None
                next_elem = line[i+1] if i < len(line)-1 else None
                
                try:
                    features = self.feature_extractor.extract_features(
                        elem, prev_elem, next_elem, line
                    )
                    
                    # Predict
                    X = pd.DataFrame([features])[self.feature_names]
                    label_encoded = self.model.predict(X)[0]
                    label = self.label_encoder.inverse_transform([label_encoded])[0]
                    proba = self.model.predict_proba(X)[0].max()
                    
                    predictions.append({
                        'element': elem,
                        'label': label,
                        'probability': proba
                    })
                    
                    if self.verbose:
                        print(f"[DEBUG] Line {line_idx}, Elem {i}: '{elem['text']}' -> {label} ({proba:.2f})")
                
                except Exception as e:
                    print("="*60)
                    print("Prediction Error")
                    print(elem["text"])
                    print(features)
                    print(e)
                    print("="*60)

                    raise
        
        if self.verbose:
            print(f"[DEBUG] Total predictions: {len(predictions)}")
        
        return predictions
    
    def parse(self, ocr_result: Dict) -> Dict[str, Any]:
        """
        Parse OCR result menjadi structured items
        
        Returns:
            Dictionary berisi:
            - items: List[Dict] dengan format {item_name, qty, price}
            - total_items: int
            - status: str
        """
        try:
            # Debug: print available keys
            if not isinstance(ocr_result, dict):
                return {
                    'items': [],
                    'total_items': 0,
                    'status': 'error',
                    'error': f'Invalid ocr_result type: {type(ocr_result)}'
                }
            
            # Check required keys
            if 'rec_texts' not in ocr_result:
                return {
                    'items': [],
                    'total_items': 0,
                    'status': 'error',
                    'error': f'Missing rec_texts. Available keys: {list(ocr_result.keys())}'
                }
            
            predictions = self.predict_elements(ocr_result)
            
            if not predictions:
                return {
                    'items': [],
                    'total_items': 0,
                    'status': 'no_predictions',
                    'error': 'No elements could be predicted'
                }
            
            # Group predictions menjadi items
            items = []
            current_item = {}
            
            for pred in predictions:
                label = pred['label']
                text = pred['element']['text']
                
                if label == 'item_name':
                    # Simpan item sebelumnya jika ada
                    if current_item and 'item_name' in current_item:
                        items.append(self._finalize_item(current_item))
                    
                    # Mulai item baru
                    current_item = {'item_name': text}
                
                elif label == 'qty':
                    current_item['qty'] = self._parse_qty(text)
                
                elif label == 'price':
                    current_item['price'] = self._parse_price(text)
            
            # Tambahkan item terakhir
            if current_item and 'item_name' in current_item:
                items.append(self._finalize_item(current_item))
            
            # Filter item yang valid
            valid_items = [item for item in items if self._validate_item(item)]
            
            return {
                'items': valid_items,
                'total_items': len(valid_items),
                'status': 'Success✅' if valid_items else 'no_items_found',
                'debug_info': {
                    'total_predictions': len(predictions),
                    'total_items_before_validation': len(items),
                    'label_distribution': self._get_label_distribution(predictions)
                }
            }
        
        except Exception as e:
            import traceback
            return {
                'items': [],
                'total_items': 0,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _finalize_item(self, item: Dict) -> Dict:
        """Finalisasi item dengan default values"""
        return {
            'item_name': item.get('item_name', ''),
            'qty': item.get('qty', 1),
            'price': item.get('price', 0)
        }
    
    def _parse_qty(self, text: str) -> int:
        """Parse quantity dari text"""
        text_clean = text.replace('x', '').strip()
        match = re.search(r'\d+', text_clean)
        return int(match.group()) if match else 1
    
    def _parse_price(self, text: str) -> int:
        """Parse price dari text"""
        text_clean = text.replace(',', '').replace('.', '').replace('Rp', '').strip()
        text_clean = re.sub(r'[^\d]', '', text_clean)
        return int(text_clean) if text_clean else 0
    
    def _validate_item(self, item: Dict) -> bool:
        """Validasi item"""
        if not item.get('item_name') or len(item['item_name']) < 2:
            return False
        if not (1 <= item.get('qty', 0) <= 999):
            return False
        if not (100 <= item.get('price', 0) <= 10000000):
            return False
        return True
    
    def _get_label_distribution(self, predictions: List[Dict]) -> Dict:
        """Get distribution of predicted labels"""
        distribution = {}
        for pred in predictions:
            label = pred['label']
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def save_model(self, model_path: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {model_path}")


# ============================================================================
# HELPER FUNCTIONS UNTUK PERSIAPAN DATA TRAINING
# ============================================================================

def create_annotation_template(ocr_result: Dict, output_file: str):
    """
    Buat template annotation untuk labeling manual
    
    Args:
        ocr_result: hasil OCR dari PaddleOCR
        output_file: path output JSON untuk annotation
    """
    template = {
        'ocr_result': ocr_result,
        'annotations': []
    }
    
    texts = ocr_result.get('rec_texts', [])
    for i, text in enumerate(texts):
        template['annotations'].append({
            'index': i,
            'text': text,
            'label': 'ignore'  # default label, harus di-edit manual
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"Annotation template created: {output_file}")
    print("Edit file tersebut dan ubah 'label' menjadi:")
    print("- 'item_name': nama produk")
    print("- 'qty': kuantitas")
    print("- 'price': harga")
    print("- 'ignore': elemen yang diabaikan")


def load_annotated_data(annotation_files: List[str]) -> List[Dict]:
    """
    Load multiple annotated files
    
    Args:
        annotation_files: list path ke file annotation
    
    Returns:
        List of annotated data
    """
    data = []
    for file_path in annotation_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONTOH 1: MEMBUAT ANNOTATION TEMPLATE
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: CREATE ANNOTATION TEMPLATE")
    print("="*60)
    
    # Load sample OCR result
    with open('output/nota9_res.json', 'r') as f:
        sample_ocr = json.load(f)
    
    # Create template untuk manual annotation
    create_annotation_template(sample_ocr, 'annotations/nota9_annotated.json')
    
    print("\nSetelah membuat template, edit file annotation secara manual:")
    print("1. Buka annotations/nota9_annotated.json")
    print("2. Untuk setiap element, ubah 'label' sesuai kategorinya")
    print("3. Ulangi untuk beberapa receipt lainnya (minimal 10-20 receipt)")
    
    # ========================================================================
    # CONTOH 2: TRAINING MODEL (setelah punya data annotated)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: TRAINING MODEL")
    print("="*60)
    
    # Load annotated data
    # annotation_files = [
    #     'annotations/nota9_annotated.json',
    #     'annotations/nota10_annotated.json',
    #     # ... tambahkan file lainnya
    # ]
    # annotated_data = load_annotated_data(annotation_files)
    
    # Untuk demo, kita skip training
    print("Untuk training, siapkan data annotated terlebih dahulu")
    print("Kemudian uncomment code di atas dan jalankan:")
    print("""
    parser = MLReceiptParser()
    X, y = parser.prepare_training_data(annotated_data)
    parser.train(X, y)
    parser.save_model('models/receipt_parser_rf.pkl')
    """)
    
    # ========================================================================
    # CONTOH 3: INFERENCE (menggunakan trained model)
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: INFERENCE")
    print("="*60)
    
    # Load trained model (jika sudah ada)
    # parser = MLReceiptParser(model_path='models/receipt_parser_rf.pkl')
    
    # Parse receipt
    # result = parser.parse(sample_ocr)
    
    # print(f"\nStatus: {result['status']}")
    # print(f"Total Items: {result['total_items']}\n")
    
    # if result['items']:
    #     print("{:<3} {:<30} {:<6} {:<10}".format("No.", "Name", "Qty", "Price"))
    #     print("-"*55)
    #     for i, item in enumerate(result['items'], 1):
    #         print("{:<3} {:<30} {:<6} {:>10,}".format(
    #             i, item['item_name'][:28], item['qty'], item['price']
    #         ))
    
    print("\nSetelah model trained, uncomment code di atas untuk inference")