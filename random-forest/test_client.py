import requests
import json

# URL endpoint Flask API Anda
url = "http://localhost:5000/api/parse-receipt"

# Path ke gambar nota Anda
image_path = "img/nota2.jpg"

try:
    print(f"Mengirim berkas {image_path} ke API...")
    
    # Membuka file gambar dalam mode binary
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        
        # Mengirim POST request dengan multipart/form-data
        response = requests.post(url, files=files)
        
    # Memeriksa status response
    if response.status_code == 200:
        print("\n--- HASIL PARSING BERHASIL ---")
        # Mencetak JSON dengan format yang rapi
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    else:
        print(f"\nAPI mengembalikan error (Status Code: {response.status_code})")
        print(response.text)

except FileNotFoundError:
    print(f"Error: File tidak ditemukan di path '{image_path}'. Pastikan foldernya benar.")
except requests.exceptions.ConnectionError:
    print("Error: Tidak dapat terhubung ke server API. Apakah 'api.py' sudah dijalankan?")
except Exception as e:
    print(f"Terjadi error: {str(e)}")