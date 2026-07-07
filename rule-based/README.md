# Receipt OCR Parser

Proyek ini melakukan ekstraksi informasi (nama item, qty, harga) dari gambar struk/nota belanja menggunakan **PaddleOCR** untuk deteksi & pengenalan teks, kemudian melakukan *post-processing* dengan **fuzzy matching** untuk mem-parsing hasil OCR menjadi data terstruktur. Proyek ini juga dilengkapi skrip evaluasi untuk mengukur akurasi hasil parsing terhadap data *ground truth*.

## Struktur Folder

```
Rule-Based Receipt Parser
├── app.py                     # Skrip utama: OCR + parsing semua gambar di folder img/
├── evaluate_rule_based.py     # Skrip evaluasi hasil parsing vs ground truth
├── parser.py                  # Modul ReceiptParser (fuzzy matching & parsing)
├── img/                       # Folder input: gambar nota/struk (.jpg, .jpeg, .png, .webp)
├── output/                    # Folder output: hasil mentah OCR PaddleOCR (JSON), dibuat otomatis
├── results/                   # Folder output: hasil parsing terstruktur (JSON), dibuat otomatis
└── ground_truth/              # Folder input: label ground truth (JSON) untuk evaluasi
```

## Requirements

- Python 3.8+
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- scikit-learn

Instalasi dependensi:

```bash
pip install paddleocr scikit-learn
```

> Pastikan juga PaddlePaddle (backend PaddleOCR) sudah terinstall sesuai environment kamu (CPU/GPU). Lihat dokumentasi resmi PaddleOCR untuk instruksi instalasi PaddlePaddle.

## Cara Pakai

### 1. Siapkan gambar nota

Letakkan semua gambar struk/nota yang ingin diproses ke dalam folder `img/`:

```
img/
├── nota01.jpg
├── nota02.jpg
└── nota26.jpg
```

### 2. Jalankan OCR + Parsing

```bash
python app.py
```

Skrip ini akan:
1. Memuat semua gambar di folder `img/`.
2. Menjalankan PaddleOCR (deteksi + pengenalan teks) pada setiap gambar.
3. Menyimpan hasil OCR mentah ke `output/<nama_file>_res.json`.
4. Melakukan parsing (fuzzy matching) untuk mengekstrak `item_name`, `qty`, dan `price`.
5. Menampilkan hasil parsing di terminal.
6. Menyimpan hasil parsing terstruktur ke `results/<nama_file>_res.json`.

### 3. Siapkan Ground Truth

Buat file JSON ground truth untuk setiap nota di folder `ground_truth/`, dengan nama file yang **sama** dengan nama gambar (tanpa akhiran `_res`). Contoh: untuk `img/nota26.jpg`, buat `ground_truth/nota26.json` berisi list item aktual:

```json
[
  { "item_name": "Indomie Goreng", "qty": 2, "price": 3500 },
  { "item_name": "Aqua 600ml", "qty": 1, "price": 4000 }
]
```

### 4. Jalankan Evaluasi

```bash
python evaluate_rule_based.py
```

Skrip ini akan mencocokkan setiap file di `ground_truth/` dengan hasil prediksi yang bersesuaian di `results/` (`<nama>_res.json`), lalu menampilkan:

- **Confusion Matrix** untuk kategori `item_name`, `qty`, `price`, `ignore`
- **Classification Report** (precision, recall, f1-score) menggunakan `scikit-learn`

### 5. Jalankan Compare Predictions

```bash
python compare_predictions.py
```

Skrip ini akan menampilkan perbandingan antara hasil parsing dan ground truth untuk setiap gambar.

1. ❌ di depan baris = nama item TIDAK cocok dengan ground truth
2. ✗  setelah angka  = qty atau price TIDAK cocok dengan ground truth
3. ✓  setelah angka  = qty atau price cocok dengan ground truth
4. —                 = item tidak ada (kosong) di sisi tersebut

## Konfigurasi Model

Parameter PaddleOCR yang digunakan di `app.py` (dapat disesuaikan):

| Parameter | Nilai | Keterangan |
|---|---|---|
| `text_detection_model_name` | `PP-OCRv5_mobile_det` | Model deteksi teks |
| `text_recognition_model_name` | `PP-OCRv5_mobile_rec` | Model pengenalan teks |
| `use_doc_orientation_classify` | `False` | Nonaktifkan klasifikasi orientasi dokumen |
| `use_doc_unwarping` | `False` | Nonaktifkan koreksi distorsi dokumen |
| `use_textline_orientation` | `False` | Nonaktifkan deteksi orientasi baris teks |
| `text_det_unclip_ratio` | `1.2` | Diturunkan agar bounding box tidak mudah menyambung antar baris |
| `enable_mkldnn` | `True` | Akselerasi komputasi di CPU Intel/AMD |

## Catatan

- Format ekstensi gambar yang didukung: `.jpg`, `.jpeg`, `.png`, `.webp`.
- Folder `output/` dan `results/` akan dibuat otomatis jika belum ada.
- Jika hasil parsing kosong untuk suatu gambar, status akan ditampilkan sebagai error dengan pesan "Tidak ada item yang terdeteksi."