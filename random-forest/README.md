# Machine Learning-based Receipt Parser

Post-processing OCR menggunakan **Random Forest Classifier** untuk Finance Tracker aplikasi.

## 📋 Deskripsi

Sistem ini menggunakan pendekatan **Machine Learning** untuk post-processing hasil OCR dari PaddleOCR. Berbeda dengan rule-based parsing, ML-based parser dapat:
- **Belajar dari data** - Model belajar pola dari data annotated
- **Adaptif** - Lebih fleksibel terhadap variasi format receipt
- **Scalable** - Performa meningkat seiring bertambahnya data training
- **Feature-rich** - Menggunakan 30+ features untuk klasifikasi

## 🏗️ Arsitektur Sistem

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Receipt   │────▶│  PaddleOCR  │────▶│   Feature   │────▶│   Random    │
│    Image    │     │   (OCR)     │     │ Extraction  │     │   Forest    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
                                                            ┌─────────────────┐
                                                            │  Structured     │
                                                            │  Items Output   │
                                                            └─────────────────┘
```

## 📦 Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
paddleocr>=2.7.0
paddlepaddle>=2.5.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
colorama>=0.4.4
```

## 📂 Struktur Project

```
receipt-parser-ml/
├── annotations/                  # Annotated data for training
│   ├── nota1_annotated.json
│   ├── nota2_annotated.json
│   └── ...
├── datasets/                     # Datasets
│   ├── test/
|   │   ├── nota1.jpg
│   │   ├── nota2.jpg
│   │   └── ...
│   └── train/
│       ├── nota1.jpg
│       ├── nota2.jpg
│       └── ...
├── models/                       # Trained models
│   ├── receipt_parser_rf.pkl
│   └── receipt_parser_rf_results.json
├── output/                       # OCR results
│   ├── nota1_res.json
│   ├── nota2_res.json
│   └── ...
├── plots/                        # Visualization plots
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── per_class_performance.png
├── results/                      # Final parsing results
│   ├── batch_summary.json
│   ├── nota1_parsed.json
│   ├── nota2_parsed.json
│   └── ...
├── tests/                        # Testing
│   ├── debug.py
│   └── test_app.py
├── annotation_tool.py            # Interactive annotation tool
├── ml_receipt_parser.py          # Core ML parser
├── paddleocr_generator.py        # OCR generator
├── train_pipeline.py             # Training pipeline
├── app.py                        # Application
└── README.md                     # Documentation
```

## 🚀 Quick Start

### 1. Generate OCR Data

```bash
# Run PaddleOCR Generator
python paddleocr_generator.py
```

or

```bash
# Run PaddleOCR from terminal
python -c "
from paddleocr import PaddleOCR
import os

model = PaddleOCR(
    text_detection_model_name='PP-OCRv5_mobile_det',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

for img_file in os.listdir('datasets/train/'):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join('datasets/train', img_file)
        predict = model.predict(img_path)
        for res in predict:
            res.save_to_json('output')
"
```

### 2. Annotate Data

Buat annotated data untuk training model:

```bash
# Single file annotation
python annotation_tool.py single output/nota1_res.json annotations/nota1_annotated.json

# Batch annotation (recommended)
python annotation_tool.py batch output/ annotations/
```

**Proses Annotation:**
1. Program akan menampilkan setiap elemen text dari OCR
2. Tentukan label untuk setiap elemen:
   - `1` = **item_name** (nama produk)
   - `2` = **qty** (kuantitas)
   - `3` = **price** (harga)
   - `4` = **ignore** (elemen yang diabaikan)
3. Commands:
   - `s` = save progress
   - `u` = undo last action
   - `q` = quit

**Tips Annotation:**
- Minimal annotate **10-20 receipts** untuk hasil yang baik
- Annotate **lebih banyak variasi** format receipt untuk model yang lebih robust
- Label yang konsisten sangat penting untuk akurasi model

### 3. Train Model

```bash
python train_pipeline.py
```

**Training Process:**
1. Load annotated data dari `annotations/`
2. Extract features (30+ features per element)
3. Train Random Forest model
4. Evaluate dengan cross-validation
5. Generate performance plots
6. Save model ke `models/receipt_parser_rf.pkl`

**Optional: Hyperparameter Tuning**
- Saat ditanya, pilih `y` untuk hyperparameter tuning
- Proses akan lebih lama tapi menghasilkan model optimal

### 4. Application

```bash
# Single image
python app.py single img/nota1.jpg

# Batch processing
python app.py batch img/
```

## 📊 Features yang Digunakan

Model menggunakan **30+ features** untuk klasifikasi setiap elemen:

| Metode | Features |
|--------|-----------|
| Random Forest | 22 |
| Random Forest + Rule-Based | 33 |

### 1. Text-based Features
- `text_length`: panjang teks
- `num_words`: jumlah kata
- `num_digits`, `num_alpha`, `num_special`: jumlah karakter
- `digit_ratio`, `alpha_ratio`, `upper_ratio`: rasio karakter

### 2. Position Features
- `x_pos`, `y_pos`: posisi koordinat
- `width`, `height`: dimensi bounding box
- `center_x`, `center_y`: center point
- `aspect_ratio`: rasio lebar/tinggi

### 3. Pattern Features (Rule-Based Features)
- `is_price_pattern`: match dengan pattern harga
- `is_qty_pattern`: match dengan pattern kuantitas
- `has_x_separator`: ada separator 'x'
- `has_comma`, `has_dot`: ada koma/titik
- `starts_with_digit`, `ends_with_digit`: dimulai/diakhiri dengan angka

### 4. Keyword Features (Rule-Based Features)
- `has_product_keyword`: mengandung keyword produk
- `has_ignore_keyword`: mengandung keyword yang harus diabaikan

### 5. Contextual Features
- `position_in_line`: posisi dalam baris
- `line_length`: panjang baris
- `is_first_in_line`, `is_last_in_line`: posisi edge (Rule-Based Features)
- `dist_to_prev`, `dist_to_next`: jarak ke elemen tetangga
- `y_diff_prev`, `y_diff_next`: perbedaan Y dengan tetangga

### 6. Confidence Score
- `ocr_score`: confidence score dari OCR

## 📈 Model Performance

Setelah training, model akan menghasilkan:

### Classification Report
Contoh output:
```
              precision    recall  f1-score   support

   item_name       0.95      0.93      0.94       120
         qty       0.92      0.90      0.91        80
       price       0.97      0.96      0.97       100
      ignore       0.99      0.99      0.99       300

    accuracy                           0.97       600
   macro avg       0.96      0.95      0.95       600
weighted avg       0.97      0.97      0.97       600
```

### Confusion Matrix
Menunjukkan distribusi prediksi vs actual labels.
Contoh output:
```
Confusion Matrix:
[[126   1   4   1]
 [  3  26   0   0]
 [  6   0  20   0]
 [  1   0   0  16]]
```

### Feature Importance
Top features yang paling berpengaruh dalam klasifikasi.

## 🔍 Comparison: ML vs Rule-based

| Aspect | Rule-based | ML-based (Random Forest) |
|--------|-----------|-------------------------|
| **Approach** | Hard-coded rules & patterns | Learn from annotated data |
| **Flexibility** | Low - needs manual rules update | High - adapts with more data |
| **Accuracy** | 60-80% (varies by format) | 85-95% (with enough training data) |
| **Development Time** | Fast initial | Slower (need annotation) |
| **Maintenance** | High - update rules manually | Low - retrain with new data |
| **Scalability** | Poor - new format = new rules | Good - add to training data |
| **Edge Cases** | Hard to handle | Learns from examples |

## 🎯 Use Cases

### Kapan Menggunakan ML-based:
✅ Punya cukup data annotated (10+ receipts)  
✅ Butuh akurasi tinggi  
✅ Banyak variasi format receipt  
✅ Planning untuk scale up  
✅ Ada resource untuk annotation  

### Kapan Menggunakan Rule-based:
✅ Data training terbatas  
✅ Format receipt sangat konsisten  
✅ Butuh hasil cepat tanpa training  
✅ Prototype/proof-of-concept  

## 🛠️ Advanced Usage

### Custom Configuration

```python
from ml_receipt_parser import MLReceiptParser

# Custom configuration
config = {
    'line_threshold': 15,      # threshold grouping baris
    'min_price': 100,          # harga minimum valid
    'max_price': 10000000,     # harga maksimum valid
    'max_qty': 999             # qty maksimum valid
}

parser = MLReceiptParser(model_path='models/receipt_parser_rf.pkl')
result = parser.parse(ocr_result)
```

### Retrain Model dengan Data Baru

```python
from train_pipeline import run_complete_pipeline

# Tambahkan annotated data baru ke folder annotations/
# Kemudian retrain
run_complete_pipeline(
    annotation_dir='annotations',
    model_output_path='models/receipt_parser_rf_v2.pkl',
    plot_output_dir='plots',
    tune_hyperparams=True  # Optional: hyperparameter tuning
)
```

### Ensemble dengan Rule-based

```python
from ml_receipt_parser import MLReceiptParser
from parser import ReceiptParser  # rule-based

# ML parser
ml_parser = MLReceiptParser(model_path='models/receipt_parser_rf.pkl')
ml_result = ml_parser.parse(ocr_result)

# Rule-based parser
rb_parser = ReceiptParser()
rb_result = rb_parser.parse(ocr_result)

# Ensemble: gunakan hasil dengan confidence lebih tinggi
# atau voting mechanism
```

## 📝 Tips

### Metodologi:
1. **Problem Definition**: Post-processing OCR untuk Finance Tracker
2. **Data Collection**: 20-30 receipt images dengan variasi format
3. **Data Annotation**: Manual labeling untuk training
4. **Feature Engineering**: 30+ features dari OCR data
5. **Method Selection**: Random Forest Classifier atau Rule-based
6. **Training & Validation**: Train-test split + cross-validation
7. **Evaluation**: Precision, Recall, F1-Score, Confusion Matrix
8. **Comparison**: ML vs Rule-based parsing
9. **Use Cases**: Kapan menggunakan ML-based, kapan menggunakan rule-based

### Metrics untuk Evaluasi:
- **Accuracy**: overall correctness
- **Precision**: ketepatan per class
- **Recall**: coverage per class
- **F1-Score**: harmonic mean precision & recall
- **Item-level accuracy**: berapa item terdeteksi dengan benar
- **Processing time**: waktu inference per receipt

## 🐛 Troubleshooting

### Model Accuracy Rendah
- **Penyebab**: Data training terlalu sedikit atau tidak diverse
- **Solusi**: Annotate lebih banyak receipt dengan variasi format

### Overfitting
- **Gejala**: Train accuracy tinggi (>95%) tapi test accuracy rendah (<80%)
- **Solusi**: Reduce model complexity, add more training data

### Missing Items
- **Penyebab**: Pattern tidak terecognize oleh model
- **Solusi**: Add more examples dari pattern tersebut ke training data

### Wrong Classification
- **Penyebab**: Ambiguitas dalam features atau labeling tidak konsisten
- **Solusi**: Review annotation consistency, add discriminative features

## 📚 References

- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- Scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
- OCR Post-processing: Relevant papers & approaches

## 📧 Contact

Untuk pertanyaan atau diskusi, silakan kontak:
- Email: [contact@hikmal-falah.com](mailto:contact@hikmal-falah.com)
- GitHub: [@Superior231 (Hikmal Falah)](https://github.com/Superior231)

---

NOTE:
Perlu diperhatikan pada data training ini karena bounding box OCR kurang akurat (qty menyatu dengan harga per item):
- nota3.jpg
- nota6.jpg
- nota7.jpg
- nota8.jpg
- nota11.jpg
- nota20.jpg
- nota21.jpg
- nota22.jpg

Datasets yang kurang:
- Nota Alfamart
- Nota caffee
- Nota elektronik
- Nota dengan format:
    | Nama          |          |
    |---------------|----------|
    | Qty   @price  | Subtotal |
- Nota dengan format:
    | Qty  Nama     | Subtotal |
    |---------------|----------|
