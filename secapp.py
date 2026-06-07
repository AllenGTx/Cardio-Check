import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
import traceback

# 1. Load Environment Variables
load_dotenv()

# 2. Konfigurasi Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel('gemini-2.5-flash')

# 3. Inisialisasi Flask dengan folder 'Asset' untuk file gambar
app = Flask(__name__, static_folder='Asset', static_url_path='/Asset')

# 4. Load Model Logistic Regression
MODEL_PATH = 'models/final_heart_attack_lr_model_calibrated.pkl'
try:
    lr_model = joblib.load(MODEL_PATH)
    print("✅ Model Logistic Regression berhasil dimuat!")
except Exception as e:
    print(f"❌ Error memuat model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        name = data.get('name', 'Pengguna')
        gender = data.get('gender')
        bmi = float(data.get('bmi'))
        age_str = data.get('age_category')
        alcohol = data.get('alcohol') 
        
        if bmi < 10 or bmi > 80:
            return jsonify({'status': 'error', 'message': 'Nilai BMI tidak wajar.'}), 400

        age_mapping = {
            "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, 
            "45-49": 5, "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9, 
            "70-74": 10, "75-79": 11, "80 or older": 12
        }
        
        input_dict = {
            'BMI': [bmi],
            'AgeCategory': [int(age_mapping.get(age_str, 0))],
            'SmokerStatus': [int(data.get('smoker', 0))],
            'PhysicalActivities': [1 if data.get('physicalActivity') == 'Yes' else 0],
            'HadDiabetes': [1.0 if data.get('hadDiabetes') == 'Yes' else 0.0],
            'HadStroke': [1 if data.get('hadStroke') == 'Yes' else 0],
            'HadAngina': [1 if data.get('hadAngina') == 'Yes' else 0],
            'DifficultyWalking': [1 if data.get('diffWalking') == 'Yes' else 0],
            'HadCOPD': [1 if data.get('hadCOPD') == 'Yes' else 0],
            'HadKidneyDisease': [1 if data.get('hadKidneyDisease') == 'Yes' else 0],
            'Sex': [1 if gender == 'Male' else 0]
        }

        input_data = pd.DataFrame(input_dict)
        expected_columns = ['BMI', 'AgeCategory', 'SmokerStatus', 'PhysicalActivities', 
                            'HadDiabetes', 'HadStroke', 'HadAngina', 'DifficultyWalking', 
                            'HadCOPD', 'HadKidneyDisease', 'Sex']
        input_data = input_data[expected_columns]

        risk_probability = float(lr_model.predict_proba(input_data)[0][1])
        risk_score = round(risk_probability * 100, 1)


        smoker_labels = {'0': 'Tidak Pernah Merokok', '1': 'Mantan Perokok', '2': 'Perokok Beberapa Hari', '3': 'Perokok Setiap Hari'}
        smoker_display = smoker_labels.get(str(data.get('smoker', '0')), 'Tidak Pernah Merokok')
        prompt = f"""
        Kamu adalah dokter spesialis jantung. Buat penjelasan hasil analisis untuk pasien bernama {name} dalam format HTML murni (tanpa markdown, tanpa kode blok, langsung output HTML).

        Data Pasien: Usia {age_str}, Gender {gender}, BMI {bmi}, Perokok: {smoker_display}, Alkohol: {alcohol}, Diabetes: {data.get('hadDiabetes')}, Stroke: {data.get('hadStroke')}, Angina: {data.get('hadAngina')}, Kesulitan Berjalan: {data.get('diffWalking')}, COPD: {data.get('hadCOPD')}, Ginjal: {data.get('hadKidneyDisease')}.
        Skor Risiko Jantung ML: {risk_score}%.

        ATURAN FORMAT HTML YANG WAJIB DIIKUTI:

        1. Untuk heading setiap bagian, gunakan: <h3>Judul Bagian</h3>
           Heading akan otomatis tampil dengan garis biru di kiri (sudah di-style).

        2. Untuk teks yang menunjukkan HAL POSITIF / BAIK (faktor pelindung jantung):
           Gunakan: <span class="ai-positive">teks positif</span>  → tampil HIJAU TEBAL

        3. Untuk teks yang menunjukkan HAL BERISIKO / BERBAHAYA:
           Gunakan: <span class="ai-risk">teks berisiko</span>  → tampil MERAH TEBAL

        4. Untuk angka skor atau nilai penting:
           Gunakan: <span class="ai-score">{risk_score}%</span>  → tampil merah besar

        5. Untuk istilah medis atau penekanan netral penting:
           Gunakan: <span class="ai-highlight">istilah</span>  → tampil cokelat/amber tebal

        6. Untuk daftar faktor: gunakan <ul><li>...</li></ul>
           Di awal setiap <li>, tulis label faktor dengan <strong>Nama Faktor:</strong> lalu penjelasannya.
           Contoh: <li><strong>BMI ({bmi}):</strong> Penjelasan... <span class="ai-positive">berat badan ideal</span> mengurangi...</li>

        7. Pisahkan 3 bagian dengan: <hr>

        STRUKTUR 3 BAGIAN YANG WAJIB:

        BAGIAN 1 — <h3>Analisis Skor Risiko</h3>
        Tulis 1-2 paragraf <p> yang menjelaskan arti skor <span class="ai-score">{risk_score}%</span>, apakah rendah/sedang/tinggi, dan gambaran umum kondisi pasien. Gunakan span warna sesuai aturan di atas.
        Lalu buat sub-heading dengan <h3>Faktor yang Mengurangi Risiko (Positif):</h3> dan daftar <ul> faktor positif pasien.
        Lalu buat sub-heading <h3>Faktor yang Meningkatkan Risiko:</h3> dan daftar <ul> faktor risiko pasien.

        BAGIAN 2 — Setelah <hr>, tulis <h3>Analisis Kebiasaan dan Riwayat Kesehatan</h3>
        Buat daftar <ul> yang menganalisis SETIAP faktor data pasien secara individual. Setiap <li> harus memiliki <strong>Nama Faktor:</strong> di awal, lalu penjelasan mekanisme medisnya. Gunakan span warna sesuai aturan.

        BAGIAN 3 — Setelah <hr>, tulis <h3>Solusi dan Rekomendasi Edukasi</h3>
        Buat daftar <ul> rekomendasi konkret dan spesifik untuk pasien ini. Setiap <li> diawali <strong>Nama Rekomendasi:</strong>. Sertakan kapan harus ke dokter. Gunakan span warna untuk hal positif yang dianjurkan.
        """

        ai_response = llm_model.generate_content(prompt)
        ai_analysis = ai_response.text.replace("```html", "").replace("```", "").strip()

        return jsonify({'status': 'success', 'risk_score': risk_score, 'ai_analysis': ai_analysis})

    except Exception:
        return jsonify({'status': 'error', 'message': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)