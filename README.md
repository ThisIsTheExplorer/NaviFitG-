# NaviFitG-

✨ NaviFitG-

NaviFitG- adalah perangkat pintar berbasis Python yang dirancang untuk membantu tunanetra mendeteksi lubang jalan secara real-time. Dengan memanfaatkan model deteksi berbasis ultralytics dan pengolahan audio, alat ini memberi peringatan suara yang cepat dan akurat, sehingga pengguna dapat bergerak lebih aman dan percaya diri.

🚀 Fitur Utama

🔊 Peringatan Audio Instan – Memberi tahu pengguna tentang adanya lubang.

🧠 Model AI Cerdas – Menggunakan PyTorch/Ultralytics untuk deteksi lubang.

⚡ Ringan dan Cepat – Dibuat dengan Flask untuk server lokal dan OpenCV untuk pengolahan video.

🌍 Mendukung Blue Economy – Memberi solusi mobilitas yang ramah inklusi sosial.

🛠 Instalasi
📌 Cara Pertama (Disarankan – Menggunakan Virtual Environment)
# 1. Buat dan aktifkan virtual environment
python -m venv env

# Linux / Mac
source env/bin/activate

# Windows
env\Scripts\activate

# 2. Instal semua library yang dibutuhkan
pip install -r requirements.txt

📌 Cara Kedua (Langsung Menggunakan Pip)

Jika tidak ingin membuat virtual environment, jalankan:

pip install flask ultralytics opencv-python numpy statistics pynmea2 pyserial

📂 Catatan

📜 Semua dependensi utama sudah ada di requirements.txt, jadi cukup install sekali dengan cara pertama jika ingin lebih rapi dan terisolasi.
