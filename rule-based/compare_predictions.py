"""
compare_predictions.py

Membandingkan hasil parsing (results/) dengan ground truth (ground_truth/)
secara detail per-item, untuk membantu diagnosa apakah kesalahan berasal
dari OCR (teks salah dibaca) atau dari logic fuzzy matching di parser.py.

Jalankan dari folder yang sama dengan app.py / evaluate_rule_based.py:
    python compare_predictions.py
"""

import os
import json

GROUND_TRUTH_DIR = "ground_truth"
PREDICTION_DIR = "results"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_file(filename):
    gt_path = os.path.join(GROUND_TRUTH_DIR, filename)
    pred_filename = filename.replace(".json", "_res.json")
    pred_path = os.path.join(PREDICTION_DIR, pred_filename)

    if not os.path.exists(pred_path):
        print(f"⚠️  Prediksi tidak ditemukan untuk: {filename} (dicari: {pred_filename})")
        return 0, 0

    gt = load_json(gt_path)
    pred = load_json(pred_path)

    total = max(len(gt), len(pred))
    n_correct_row = 0

    print("="*90)
    print(f"| File: {filename}   (ground_truth: {len(gt)} item, prediksi: {len(pred)} item)")
    print("="*90)

    header = "{:<3} {:<28} {:<28} {:<8} {:<8} {:<10} {:<10}"
    print(header.format("No", "GT Item", "Pred Item", "GT Qty", "Pr Qty", "GT Price", "Pr Price"))
    print("-"*90)

    for i in range(total):
        gt_item = gt[i] if i < len(gt) else None
        pr_item = pred[i] if i < len(pred) else None

        gt_name = gt_item["item_name"] if gt_item else "—"
        pr_name = pr_item["item_name"] if pr_item else "—"
        gt_qty = gt_item["qty"] if gt_item else "—"
        pr_qty = pr_item["qty"] if pr_item else "—"
        gt_price = gt_item["price"] if gt_item else "—"
        pr_price = pr_item["price"] if pr_item else "—"

        name_ok = gt_item is not None and pr_item is not None and gt_item["item_name"] == pr_item["item_name"]
        qty_ok = gt_item is not None and pr_item is not None and gt_item["qty"] == pr_item["qty"]
        price_ok = gt_item is not None and pr_item is not None and gt_item["price"] == pr_item["price"]

        if name_ok and qty_ok and price_ok:
            n_correct_row += 1

        mark_name = "✓" if name_ok else "✗"
        mark_qty = "✓" if qty_ok else "✗"
        mark_price = "✓" if price_ok else "✗"

        row = header.format(
            i + 1,
            str(gt_name)[:28],
            str(pr_name)[:28],
            f"{gt_qty}{mark_qty}",
            f"{pr_qty}",
            f"{gt_price}{mark_price}",
            f"{pr_price}",
        )
        # Tandai nama item yang tidak cocok dengan simbol ✗ di depan baris
        prefix = "   " if name_ok else " ❌"
        print(f"{prefix}{row}")

    print("-"*90)
    print(f"| Baris cocok 100% (item+qty+price benar semua): {n_correct_row}/{total}")
    print("-"*90)
    print()

    return n_correct_row, total


def main():
    if not os.path.isdir(GROUND_TRUTH_DIR):
        print(f"Folder '{GROUND_TRUTH_DIR}/' tidak ditemukan.")
        return

    files = sorted(f for f in os.listdir(GROUND_TRUTH_DIR) if f.endswith(".json"))

    if not files:
        print(f"Tidak ada file .json di '{GROUND_TRUTH_DIR}/'")
        return

    total_correct = 0
    total_rows = 0

    for filename in files:
        c, t = compare_file(filename)
        total_correct += c
        total_rows += t

    print("="*90)
    print(f"| RINGKASAN: {total_correct}/{total_rows} baris cocok 100% di semua file "
          f"({(total_correct/total_rows*100 if total_rows else 0):.2f}%)")
    print("="*90)
    print()
    print("Keterangan:")
    print("1.  ❌ di depan baris = nama item TIDAK cocok dengan ground truth")
    print("2.  ✗  setelah angka  = qty atau price TIDAK cocok dengan ground truth")
    print("3.  ✓  setelah angka  = qty dan price cocok dengan ground truth")
    print("4.  —                 = item tidak ada (kosong) di sisi tersebut")


if __name__ == "__main__":
    main()
