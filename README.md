# CardioCheck ❤️🩺
Sistem Prediksi Risiko Serangan Jantung Berbasis Machine Learning & Generative AI (Google Gemini).

---

## 🌐 Live Demo
Aplikasi ini sudah di-deploy dan dapat diakses langsung melalui Vercel:  
🔗 **[cardio-checker.vercel.app](https://cardio-checker.vercel.app)**

---

## 📌 Tentang Aplikasi
**CardioCheck** adalah aplikasi web interaktif yang membantu pengguna memperkirakan risiko serangan jantung berdasarkan parameter gaya hidup, kondisi fisik, dan riwayat kesehatan.

### Fitur Utama:
1. **Machine Learning Risk Prediction**:
   - Model klasifikasi *Logistic Regression* yang telah dilatih menggunakan dataset kesehatan CDC BRFSS.
   - Menghitung persentase probabilitas risiko secara presisi (0% - 100%) dan membaginya ke dalam 3 kategori:
     - 🟢 **Risiko Rendah** (< 20%)
     - 🟡 **Waspada / Risiko Sedang** (20% - 49.9%)
     - 🔴 **Risiko Tinggi** (≥ 50%)
2. **Generative AI Clinical Analysis**:
   - Terintegrasi dengan **Google Gemini AI** (`gemini-2.5-flash`) untuk memberikan penjelasan klinis yang komprehensif, personal, dan mudah dipahami oleh orang awam.
   - Memberikan rekomendasi gaya hidup, pola makan, serta anjuran medis sesuai profil risiko masing-masing pasien.
   - Dilengkapi sistem **Auto-Retry** otomatis di sisi backend untuk menangani kendala server overload (HTTP 503) secara transparan.

---

## 🛠️ Tech Stack
- **Backend & ML**: Python 3.x, Flask, Scikit-Learn, Pandas, NumPy, Joblib
- **Generative AI**: Google Gemini API (`gemini-2.5-flash`)
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API, Vanilla JS)
- **Deployment**: Vercel Serverless Function

---

## 💻 Cara Menjalankan di Komputer Lokal (Localhost)

### 1. Clone Repository
```bash
git clone https://github.com/AllenGTx/Cardio-Check.git
cd Cardio-Check
```

### 2. Buat & Aktifkan Virtual Environment (Disarankan)
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment Variable (`.env`)
Buat file bernama `.env` di folder root project (sejajar dengan `secapp.py`), lalu isi dengan API Key Gemini Anda:
```env
GEMINI_API_KEY=masukkan_api_key_gemini_anda_disini
```
> 💡 *Catatan: Anda bisa mendapatkan API Key secara gratis di [Google AI Studio](https://aistudio.google.com/).*

### 5. Jalankan Server Flask
```bash
python secapp.py
```
Aplikasi akan berjalan di browser pada alamat:  
👉 **`http://127.0.0.1:5000`**

---

## 🚀 Cara Commit & Push Pembaruan ke GitHub / Vercel

Setiap kali Anda selesai melakukan perubahan kode dan ingin memperbarui website Vercel secara otomatis:

1. **Cek status perubahan file:**
   ```bash
   git status
   ```

2. **Tambahkan file yang ingin di-commit:**
   ```bash
   git add .
   ```

3. **Buat commit dengan pesan yang jelas:**
   ```bash
   git commit -m "Deskripsi singkat perubahan Anda"
   ```

4. **Kirim pembaruan ke GitHub:**
   ```bash
   git push origin main
   ```

> ⚡ **Auto-Deploy Vercel:**  
> Setelah perintah `git push` selesai, Vercel akan otomatis mendeteksi commit baru di branch `main` dan memperbarui deployment live Anda dalam waktu 1-2 menit tanpa perlu deploy manual.
