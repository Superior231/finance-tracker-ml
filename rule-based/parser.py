import re
from typing import List, Dict, Any, Optional
from fuzzywuzzy import fuzz
import json
import os

class ReceiptParser:
    """
    Parser struk belanja OCR multi-format:
    - Nota1: standard format
    - Nota2: name qty price format -> "Indomie Goreng 1x 5000"
    - Nota3: format dengan nomor urut (1. Item Name)
    - Nota4: format restaurant: qty name price -> "2x T-Bone 500gr  1,000,000"
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Inisialisasi ReceiptParser dengan konfigurasi.
        
        Args:
            config: Dictionary konfigurasi opsional berisi:
                - line_threshold: threshold untuk mengelompokkan elemen ke dalam baris
                - fuzzy_threshold: threshold untuk fuzzy matching
                - confidence_threshold: threshold confidence OCR
                - min_price: harga minimum item yang valid
                - max_price: harga maksimum item yang valid
                - max_qty: quantity maksimum yang valid
        """
        self.config = config or {}
        self.line_threshold = self.config.get('line_threshold', 5)
        self.fuzzy_threshold = self.config.get('fuzzy_threshold', 80)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.ignore_patterns = {
            'header': [
                'npwp', 'telp', 'telepon', 'phone', 'fax', 'email',
                'alamat', 'address', 'jl.', 'JL.', 'jln', 'jalan', 'rt', 'rw',
                'kelurahan', 'kecamatan', 'provinsi', 'kota',
                'website', 'www', 'instagram', 'facebook'
            ],
            'footer': [
                'cancel', 'voucher', 'diskon', 'discount', 'potongan',
                'harga', 'jual', 'total', 'subtotal',
                'bayar', 'tunai', 'cash', 'kembali', 'kembalian',
                'terima kasih', 'thank you', 'selamat', 'datang kembali',
                'payment', 'payment method', 'debit', 'credit', 'cash',
                'dana', 'gopay', 'ovo', 'transfer',
                'anda', 'qty'
            ]
        }
        # Pattern untuk mendeteksi tanggal/timestamp
        self.datetime_patterns = [
            r'\d{2}\.\d{2}-\d{2}:\d{2}',    # 06.18-17:08
            r'\d{1,2}\.\d{1,2}\.\d{2,4}',   # 2.1.27 atau 16.06.18
            r'\d{4}-\d{2}-\d{2}',           # 2023-08-02
            r'\d{2}:\d{2}:\d{2}',           # 08:46:36
            r'\d{6,}/[A-Z]+/\d+',           # 301135/RATIH/01 -> cashier
        ]
        self.product_indicators = [
            'kg','gr','gram','ml','ltr','pcs','box','pack',
            'btl','bottle','sachet','dus','karton','x', 'lusin',
            'abc','indomie','indofood','unilever','wings', 
            'nestle','coca','pepsi','aqua', 'sedap', 'tea', 
            'sari', 'susu', 'milk', 'ice', 'oat', 'mineral', 'choco',
            'fruit', 'belfood', 'sosis', 'goreng', 'steak', 'bone',
            'sirloin', 'ribeye', 'water', 'juice', 'salad', 'tonic',
            'milo', 'rinso', 'downy',
        ]
        self.min_price = self.config.get('min_price',100)
        self.max_price = self.config.get('max_price',10000000)
        self.max_qty = self.config.get('max_qty',999)

    def parse(self, ocr_result) -> Dict[str, Any]:
        """
        Parse hasil OCR struk belanja menjadi daftar item terstruktur.
        
        Args:
            ocr_result: Dictionary atau list hasil OCR berisi rec_texts, rec_boxes, rec_scores
            
        Returns:
            Dictionary berisi:
                - items: List[Dict] daftar item yang terparse
                - total_items: jumlah total item
                - status: status parsing ('Success✅', 'no_data', atau 'error')
                - error: pesan error jika ada
        """
        try:
            if isinstance(ocr_result, list) and ocr_result:
                ocr_result = ocr_result[0]
            
            texts  = ocr_result.get('rec_texts', [])
            boxes  = ocr_result.get('rec_boxes', [])
            scores = ocr_result.get('rec_scores', [])
            
            if not texts:
                return {'items': [], 'total_items':0, 'status':'no_data'}
            
            elements = self._prepare_elements(texts, boxes, scores)
            elements.sort(key=lambda x: x['y'])
            lines = self._group_into_lines(elements)
            items = self._multi_pass_extraction(lines)
            items = self._post_process_items(items)
            
            return {'items': items, 'total_items': len(items), 'status':'Success✅'}
        except Exception as e:
            return {'items':[], 'total_items':0, 'status':'error','error':str(e)}

    def _prepare_elements(self, texts: List[str], boxes: List[List[int]], scores: List[float]) -> List[Dict]:
        """
        Menyiapkan elemen-elemen OCR dengan metadata tambahan.
        
        Args:
            texts: List string teks hasil OCR
            boxes: List bounding boxes [x1, y1, x2, y2]
            scores: List confidence scores
            
        Returns:
            List[Dict]: List elemen dengan metadata lengkap (posisi, ukuran, center point)
        """
        elements = []
        for i, text in enumerate(texts):
            text_clean = text.strip()
            if not text_clean:
                continue
            box = boxes[i] if boxes and i < len(boxes) else [0,0,0,0]
            score = scores[i] if scores and i < len(scores) else 1.0
            elements.append({
                'index': i, 'text': text_clean, 'text_lower': text_clean.lower(),
                'box': box, 'score': score,
                'x': box[0], 'y': box[1], 'x2': box[2], 'y2': box[3],
                'width': box[2]-box[0], 'height': box[3]-box[1],
                'center_x': (box[0]+box[2])/2, 'center_y': (box[1]+box[3])/2
            })
        return elements

    def _group_into_lines(self, elements: List[Dict]) -> List[List[Dict]]:
        """
        Mengelompokkan elemen-elemen OCR menjadi baris-baris berdasarkan posisi vertikal.
        
        Args:
            elements: List[Dict] elemen OCR yang sudah diurutkan berdasarkan y
            
        Returns:
            List[List[Dict]]: List berisi list elemen per baris
        """
        if not elements: 
            return []
        lines = []
        current_line = [elements[0]]
        for elem in elements[1:]:
            prev_y_center = sum([e['center_y'] for e in current_line])/len(current_line)
            avg_height = sum([e['height'] for e in current_line])/len(current_line)
            threshold = max(self.line_threshold, avg_height*0.5)
            if abs(elem['center_y']-prev_y_center) <= threshold:
                current_line.append(elem)
            else:
                current_line.sort(key=lambda x: x['x'])
                lines.append(current_line)
                current_line = [elem]
        if current_line:
            current_line.sort(key=lambda x: x['x'])
            lines.append(current_line)
        return lines

    def _multi_pass_extraction(self, lines: List[List[Dict]]) -> List[Dict]:
        """
        Ekstraksi item dengan multiple parsing passes untuk berbagai format struk.
        Menangani format: numbered, restaurant, standard, compact, dan qty x price.
        
        Args:
            lines: List[List[Dict]] baris-baris teks OCR
            
        Returns:
            List[Dict]: List item yang berhasil diparse
        """
        items = []
        prev_line_name = None
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue
            
            # Format dengan nomor urut: "1. Indomie Goreng"
            item = self._parse_numbered_format(lines, i)
            if item and self._validate_item(item):
                items.append(item)
                prev_line_name = item['item_name']
                i += 1
                continue
            
            # Format restaurant: "2x T-Bone 500gr    1,000,000"
            item = self._parse_restaurant_format(line)
            if item and self._validate_item(item):
                items.append(item)
                prev_line_name = item['item_name']
                i += 1
                continue
            
            # Nota2 format
            item = self._parse_standard_format(line)
            if item and self._validate_item(item):
                items.append(item)
                prev_line_name = item['item_name']
                i += 1
                continue
                
            item = self._parse_compact_format(line)
            if item and self._validate_item(item):
                items.append(item)
                prev_line_name = item['item_name']
                i += 1
                continue
                
            # Nota3 qty x price format
            if any('x' in e['text'] for e in line):
                item = self._parse_qtyxprice_format(line, prev_line_name)
                if item and self._validate_item(item):
                    items.append(item)
                    prev_line_name = item['item_name']
                    i += 1
                    continue
            
            i += 1
            
        return items

    def _parse_restaurant_format(self, line: List[Dict]) -> Optional[Dict]:
        """
        Parse format restaurant dengan qty prefix.
        Format: "2x T-Bone 500gr  1,000,000" atau "1x Sirloin Steak  550,000"
        
        Args:
            line: List[Dict] elemen dalam satu baris
            
        Returns:
            Optional[Dict]: Dictionary item {item_name, qty, price} atau None jika tidak match
        """
        if self._should_ignore_line(line):
            return None
            
        line_text = ' '.join([e['text'] for e in line])
        
        # Pattern: qty x item_name price
        # Cari qty di awal (bisa "2x" atau "2 x")
        match = re.match(r'^(\d+)\s*x\s+(.+?)\s+([\d,\.]+)$', line_text, re.IGNORECASE)
        
        if match:
            qty = int(match.group(1))
            item_name = match.group(2).strip()
            price = self._parse_price(match.group(3))
            
            # Validasi nama produk
            if self._is_product_name(item_name):
                return {
                    'item_name': item_name,
                    'qty': qty,
                    'price': price
                }
        
        return None

    def _parse_numbered_format(self, lines: List[List[Dict]], current_idx: int) -> Optional[Dict]:
        """
        Parse format dengan nomor urut di awal.
        Line 1: "1. Indomie Goreng"
        Line 2: "1 lusin x 36,000" atau "1 x 27,000"
        
        Args:
            lines: List[List[Dict]] semua baris
            current_idx: index baris saat ini
            
        Returns:
            Optional[Dict]: Dictionary item {item_name, qty, price} atau None jika tidak match
        """
        if current_idx >= len(lines):
            return None
            
        current_line = lines[current_idx]
        line_text = ' '.join([e['text'] for e in current_line])
        
        # Cek apakah ada nomor urut di awal (1. , 2. , dst)
        numbered_match = re.match(r'^(\d+)\.\s*(.+)', line_text)
        if not numbered_match:
            return None
        
        item_name = numbered_match.group(2).strip()
        
        # Cek apakah nama produk valid
        if not self._is_product_name(item_name):
            return None
        
        # Cari baris berikutnya yang berisi qty x price atau harga
        if current_idx + 1 < len(lines):
            next_line = lines[current_idx + 1]
            next_text = ' '.join([e['text'] for e in next_line])
            
            # Format: "1 lusin x 36,000" atau "1 500 ml x 7,000"
            qty_price_match = re.search(r'(\d+)\s*(?:lusin|pcs|box|ml|gr|kg)?\s*x\s*([\d,\.]+)', next_text, re.IGNORECASE)
            
            if qty_price_match:
                qty = int(qty_price_match.group(1))
                unit_price = self._parse_price(qty_price_match.group(2))
                
                return {
                    'item_name': item_name,
                    'qty': qty,
                    'price': unit_price
                }
            
            # Format alternatif: qty dan price terpisah
            # "1 x 27,000" dengan harga di sebelah kanan
            alt_match = re.search(r'(\d+)\s*x\s*([\d,\.]+)', next_text)
            if alt_match:
                qty = int(alt_match.group(1))
                unit_price = self._parse_price(alt_match.group(2))
                return {
                    'item_name': item_name,
                    'qty': qty,
                    'price': unit_price
                }
            
            # Cari harga di baris berikutnya
            prices = [e for e in next_line if self._is_price(e['text'])]
            if prices:
                # Ambil harga pertama sebagai unit price
                unit_price = self._parse_price(prices[0]['text'])
                return {
                    'item_name': item_name,
                    'qty': 1,
                    'price': unit_price
                }
        
        return None

    def _parse_standard_format(self, line: List[Dict]) -> Optional[Dict]:
        """
        Parse format standard struk: nama produk, qty (opsional), dan harga dalam satu baris.
        
        Args:
            line: List[Dict] elemen dalam satu baris
            
        Returns:
            Optional[Dict]: Dictionary item {item_name, qty, price} atau None jika tidak match
        """
        if self._should_ignore_line(line): 
            return None
        names  = [e for e in line if self._is_product_name(e['text'])]
        qtys   = [e for e in line if re.match(r'^\d{1,3}$', e['text'])]
        prices = [e for e in line if self._is_price(e['text'])]
        if not names or not prices: 
            return None
        item_name = names[0]['text']
        qty = int(qtys[0]['text']) if qtys else 1
        unit_price = sorted([self._parse_price(p['text']) for p in prices])[0]
        return {'item_name': item_name, 'qty': qty, 'price': unit_price}

    def _parse_compact_format(self, line: List[Dict]) -> Optional[Dict]:
        """
        Parse format compact: nama produk dan harga saja tanpa qty eksplisit.
        
        Args:
            line: List[Dict] elemen dalam satu baris
            
        Returns:
            Optional[Dict]: Dictionary item {item_name, qty=1, price} atau None jika tidak match
        """
        if len(line)<2 or self._should_ignore_line(line): 
            return None
        name, price = None, None
        for e in line:
            if self._is_product_name(e['text']): 
                name = e['text']
            elif self._is_price(e['text']):
                val = self._parse_price(e['text'])
                if self.min_price<=val<=self.max_price: 
                    price = val
        if name and price: 
            return {'item_name': name, 'qty':1, 'price': price}
        return None

    def _parse_qtyxprice_format(self, line: List[Dict], prev_line_name: Optional[str]=None) -> Optional[Dict]:
        """
        Parse format qty x price, biasanya nama produk di baris sebelumnya.
        Format: "2 x 15,000" atau "3 pcs x 5,000"
        
        Args:
            line: List[Dict] elemen dalam satu baris
            prev_line_name: nama produk dari baris sebelumnya
            
        Returns:
            Optional[Dict]: Dictionary item {item_name, qty, price} atau None jika tidak match
        """
        text = ' '.join([e['text'] for e in line])
        match = re.search(r'(\d+)\s*(?:[a-zA-Z]*)\s*x\s*([\d,\.]+)', text)
        if match:
            qty = int(match.group(1))
            price = self._parse_price(match.group(2))
            item_name = prev_line_name if prev_line_name else line[0]['text']
            return {'item_name': item_name, 'qty': qty, 'price': price}
        return None

    def _is_product_name(self, text: str) -> bool:
        """
        Validasi apakah text adalah nama produk yang valid.
        
        Args:
            text: string teks untuk divalidasi
            
        Returns:
            bool: True jika valid sebagai nama produk
        """
        if not re.search(r'[a-zA-Z]{2,}', text): 
            return False
        if re.match(r'^[\d,\.\(\)]+$', text): 
            return False
        # Jangan anggap nomor urut sebagai nama produk
        if re.match(r'^\d+\.$', text): 
            return False
        # Jangan anggap pattern "2x" atau "3x" saja sebagai nama produk
        if re.match(r'^\d+x?$', text, re.IGNORECASE): 
            return False
        # Jangan anggap tanggal/timestamp sebagai nama produk
        if self._is_datetime(text): 
            return False
        return True

    def _is_price(self, text: str) -> bool:
        """
        Validasi apakah text adalah format harga yang valid.
        
        Args:
            text: string teks untuk divalidasi
            
        Returns:
            bool: True jika valid sebagai harga
        """
        text_clean = text.replace(' ','')
        patterns = [r'^\d{3,}$', r'^\d{1,3}[,\.]\d{3}$', r'^\d{1,3}[,\.]\d{3}[,\.]\d{3}$', r'^\(\d+\)$']
        return any(re.match(p,text_clean) for p in patterns)

    def _parse_price(self, text: str) -> int:
        """
        Konversi string harga menjadi integer.
        Menghilangkan karakter non-digit seperti koma, titik, tanda kurung.
        
        Args:
            text: string harga (contoh: "1,000,000" atau "Rp 50.000")
            
        Returns:
            int: nilai harga dalam integer
        """
        text_clean = text.replace('(','').replace(')','').replace(',','').replace('.','').replace('Rp','').strip()
        numbers = re.findall(r'\d+', text_clean)
        return int(numbers[0]) if numbers else 0

    def _is_datetime(self, text: str) -> bool:
        """
        Validasi apakah text adalah format tanggal/timestamp.
        
        Args:
            text: string teks untuk divalidasi
            
        Returns:
            bool: True jika match dengan pattern tanggal/timestamp
        """
        for pattern in self.datetime_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _should_ignore_line(self, line: List[Dict]) -> bool:
        """
        Cek apakah baris harus diabaikan (header/footer struk).
        
        Args:
            line: List[Dict] elemen dalam satu baris
            
        Returns:
            bool: True jika baris harus diabaikan
        """
        line_text = ' '.join([e['text'] for e in line]).lower()
        for patterns in self.ignore_patterns.values():
            for p in patterns:
                if p in line_text: 
                    return True
                if fuzz.partial_ratio(p, line_text)>self.fuzzy_threshold: 
                    return True
        return False

    def _validate_item(self, item: Dict) -> bool:
        """
        Validasi apakah item yang diparse memenuhi kriteria valid.
        Cek nama, qty, harga, dan kata-kata yang harus dihindari.
        
        Args:
            item: Dictionary item {item_name, qty, price}
            
        Returns:
            bool: True jika item valid
        """
        if not item: 
            return False
        name = item.get('item_name','')
        if not name or len(name)<2: 
            return False
        if not re.search(r'[a-zA-Z]{2,}', name): 
            return False
        # Jangan terima nama yang merupakan tanggal/timestamp
        if self._is_datetime(name): 
            return False
        
        qty = item.get('qty',0)
        if not (1<=qty<=self.max_qty): 
            return False
        
        price = item.get('price',0)
        if not (self.min_price<=price<=self.max_price): 
            return False
        for word in ['cancel','voucher','diskon','total','bayar','kembali']:
            if word in name.lower(): 
                return False
        return True

    def _post_process_items(self, items: List[Dict]) -> List[Dict]:
        """
        Post-processing untuk menghilangkan duplikat item.
        Menggunakan fuzzy matching untuk nama dan exact matching untuk harga.
        
        Args:
            items: List[Dict] item hasil parsing
            
        Returns:
            List[Dict]: List item tanpa duplikat
        """
        unique_items = []
        for item in items:
            # if any(fuzz.ratio(item['item_name'].lower(), u['item_name'].lower())>90 and item['price']==u['price'] for u in unique_items):
            if any(u['item_name'].lower()==item['item_name'].lower() and u['price']==item['price'] for u in unique_items):
                continue
            unique_items.append(item)
        return unique_items

    def save_parsed_to_json(self, items: list, output_file: str):
        """
        Simpan hasil parsing items ke file JSON.
        
        Args:
            items: List[Dict] hasil parse
            output_file: path file output JSON
        """
        if not items:
            print("Tidak ada item untuk disimpan.")
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = ReceiptParser()
    
    for filename in [
            'output/nota_res.json',     # ✅ success
            'output/nota2_res.json',    # ✅ success
            'output/nota3_res.json',    # ⚠️ not accurate -> qty not match
            'output/nota4_res.json',    # ✅ success
            'output/nota5_res.json',    # ✅ success
            'output/nota6_res.json',    # ❌ error
            'output/nota7_res.json',    # ⚠️ not accurate -> qty not match
            'output/nota8_res.json',    # ❌ error
            'output/nota9_res.json',    # ✅ success
            'output/nota10_res.json',   # ✅ success
            'output/nota11_res.json',   # ⚠️ not accurate -> qty not match
            'output/nota12_res.json',   # ⚠️ missing item because duplicate
            'output/nota13_res.json',   # ❌ error
            'output/nota14_res.json',   # ✅ success
            'output/nota15_res.json',   # ⚠️ not accurate
            'output/nota16_res.json',   # ⚠️ missing 1 item
            'output/nota17_res.json',   # ⚠️ not accurate -> item name, qty, price not match
            'output/nota18_res.json',   # ⚠️ not accurate -> qty not match (same img but different angle -> nota16)
            'output/nota19_res.json',   # ✅ success
            'output/nota20_res.json',   # ✅ success
            'output/nota21_res.json',   # ⚠️ not accurate
        ]:

        if not os.path.exists(filename):
            continue

        with open(filename) as f:
            data = json.load(f)
        print("="*50)
        print(f"| Test: {filename}")
        print("="*50)

        result = parser.parse(data)

        if result['items']:
            # for item in result['items']:
            #     print(f"✓ {item['item_name']}: {item['qty']}x @ Rp{item['price']:,}")

            print("{:<3} {:<30} {:<6} {:<6}".format(" No.", "Name", "Qty", "Price"))
            for i, item in enumerate(result['items'], 1):
                print(" {:<3} {:<30} {:<6} {:>6,}".format(
                    i, item['item_name'], item['qty'], item['price']
                ))
            print("-"*50)
            print(f"| Status: {result['status']}")
            print("-"*50)
            print("\n")

            parser.save_parsed_to_json(result['items'], f"results/{os.path.basename(filename)}")
        else:
            print(f"| Status: Error ❌")
            print(f"| Detail: Tidak ada item yang terdeteksi.")
            print("-"*50)
            print("\n")
