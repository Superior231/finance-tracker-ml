# Generate JSON from PaddleOCR
from paddleocr import PaddleOCR
import os

model = PaddleOCR(
    text_detection_model_name='PP-OCRv5_mobile_det',
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_unclip_ratio=0.5,
)

# Single
# img_path = 'datasets/train/nota6.jpg'
# predict = model.predict(img_path)
# for res in predict:
#     res.save_to_json('output')
#     res.save_to_img("output")

# Batch
for img_file in os.listdir('datasets/train/'):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join('datasets/train', img_file)
        predict = model.predict(img_path)
        for res in predict:
            res.save_to_json('output')
