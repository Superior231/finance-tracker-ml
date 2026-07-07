import os
import json
from sklearn.metrics import classification_report, confusion_matrix

GROUND_TRUTH_DIR = "ground_truth"
PREDICTION_DIR = "results"

y_true = []
y_pred = []

# Ambil semua file ground truth
files = sorted([
    f for f in os.listdir(GROUND_TRUTH_DIR)
    if f.endswith(".json")
])

for filename in files:

    gt_path = os.path.join(GROUND_TRUTH_DIR, filename)
    pred_filename = filename.replace(".json", "_res.json")
    pred_path = os.path.join(PREDICTION_DIR, pred_filename)

    if not os.path.exists(pred_path):
        print(f"Prediction tidak ditemukan: {pred_filename}")
        continue

    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)

    # Jumlah item berbeda
    total = max(len(gt), len(pred))

    for i in range(total):

        gt_item = gt[i] if i < len(gt) else None
        pr_item = pred[i] if i < len(pred) else None

        # ----------------------
        # ITEM
        # ----------------------
        y_true.append("item_name")

        if (
            gt_item is not None and
            pr_item is not None and
            gt_item["item_name"] == pr_item["item_name"]
        ):
            y_pred.append("item_name")
        else:
            y_pred.append("ignore")

        # ----------------------
        # QTY
        # ----------------------
        y_true.append("qty")

        if (
            gt_item is not None and
            pr_item is not None and
            gt_item["qty"] == pr_item["qty"]
        ):
            y_pred.append("qty")
        else:
            y_pred.append("ignore")

        # ----------------------
        # PRICE
        # ----------------------
        y_true.append("price")

        if (
            gt_item is not None and
            pr_item is not None and
            gt_item["price"] == pr_item["price"]
        ):
            y_pred.append("price")
        else:
            y_pred.append("ignore")

print("="*60)
print("Confusion Matrix")
print("="*60)

print(confusion_matrix(
    y_true,
    y_pred,
    labels=["item_name","qty","price","ignore"]
))

print("="*60)
print("Classification Report")
print("="*60)

print(classification_report(
    y_true,
    y_pred,
    labels=["item_name","qty","price","ignore"],
    digits=4
))
