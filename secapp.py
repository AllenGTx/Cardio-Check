import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import traceback

# 1. Load Environment Variables
load_dotenv()

# 3. Inisialisasi Flask dengan folder 'Asset' untuk file gambar
app = Flask(__name__, static_folder='Asset', static_url_path='/Asset')

# 4. Load Model dan Scaler (BMI Only)
MODEL_PATH = 'models/heart_attack_model.pkl'
SCALER_PATH = 'models/heart_attack_scaler.pkl'
THRESHOLD = 0.4898

try:
    lr_model = joblib.load(MODEL_PATH)
    print("✅ Model Logistic Regression berhasil dimuat!")
except Exception as e:
    print(f"❌ Error memuat model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler BMI-Only berhasil dimuat!")
except Exception as e:
    print(f"❌ Error memuat scaler: {e}")

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

        # Susun X_input: 11 fitur dalam urutan yang benar
        X_input = [
            bmi,                                                              # index 0: BMI (akan di-scale)
            int(age_mapping.get(age_str, 0)),                                 # index 1: AgeCategory
            int(data.get('smoker', 0)),                                       # index 2: SmokerStatus
            1 if data.get('physicalActivity') == 'Yes' else 0,               # index 3: PhysicalActivities
            1.0 if data.get('hadDiabetes') == 'Yes' else 0.0,               # index 4: HadDiabetes
            1 if data.get('hadStroke') == 'Yes' else 0,                      # index 5: HadStroke
            1 if data.get('hadAngina') == 'Yes' else 0,                      # index 6: HadAngina
            1 if data.get('diffWalking') == 'Yes' else 0,                    # index 7: DifficultyWalking
            1 if data.get('hadCOPD') == 'Yes' else 0,                        # index 8: HadCOPD
            1 if data.get('hadKidneyDisease') == 'Yes' else 0,               # index 9: HadKidneyDisease
            1 if gender == 'Male' else 0                                      # index 10: Sex
        ]

        # Scale HANYA BMI (index 0), fitur lain dibiarkan mentah
        bmi_mentah = X_input[0]
        bmi_scaled = scaler.transform([[bmi_mentah]])[0][0]
        X_input[0] = bmi_scaled

        # Probabilitas murni kelas berisiko (kelas 1)
        probabilitas = lr_model.predict_proba([X_input])[0][1]

        # Terapkan custom threshold secara manual
        if probabilitas >= THRESHOLD:
            hasil_prediksi = 1  # Berisiko
        else:
            hasil_prediksi = 0  # Aman

        risk_score = round(probabilitas * 100, 1)

        smoker_labels = {'0': 'Tidak Pernah Merokok', '1': 'Mantan Perokok', '2': 'Perokok Beberapa Hari', '3': 'Perokok Setiap Hari'}
        smoker_display = smoker_labels.get(str(data.get('smoker', '0')), 'Tidak Pernah Merokok')

        if risk_score < 20:
            kategori_risiko = "Risiko Rendah"
            warna_class = "ai-positive"
            pesan = "Kondisi jantung Anda saat ini masuk dalam kategori aman. Tetap pertahankan gaya hidup sehat, rutin berolahraga, dan makan makanan bergizi."
        elif risk_score < 50:
            kategori_risiko = "Waspada"
            warna_class = "ai-highlight"
            pesan = "Anda memiliki beberapa faktor risiko yang perlu diperhatikan. Disarankan untuk mulai memperbaiki gaya hidup, mengurangi konsumsi tidak sehat, dan memantau kondisi secara berkala."
        else:
            kategori_risiko = "Risiko Tinggi"
            warna_class = "ai-risk"
            pesan = "Perhatian! Skor risiko Anda menunjukkan probabilitas yang cukup tinggi. Sangat disarankan untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan mengambil langkah penanganan yang tepat."

        ai_analysis = f"""
        <h3>Analisis Skor Risiko</h3>
        <p>Berdasarkan analisis sistem, skor risiko jantung Anda adalah <span class="ai-score">{risk_score}%</span>, yang termasuk dalam kategori <span class="{warna_class}"><strong>{kategori_risiko}</strong></span>.</p>
        <p>{pesan}</p>
        <hr>
        <h3>Faktor yang Mempengaruhi</h3>
        <ul>
            <li><strong>BMI:</strong> {bmi}</li>
            <li><strong>Perokok:</strong> {smoker_display}</li>
            <li><strong>Aktivitas Fisik:</strong> {data.get('physicalActivity', 'Tidak diketahui')}</li>
            <li><strong>Diabetes:</strong> {data.get('hadDiabetes', 'Tidak diketahui')}</li>
            <li><strong>Stroke:</strong> {data.get('hadStroke', 'Tidak diketahui')}</li>
        </ul>
        """

        return jsonify({
            'status': 'success',
            'risk_score': risk_score,
            'hasil_prediksi': hasil_prediksi,
            'ai_analysis': ai_analysis
        })

    except Exception:
        return jsonify({'status': 'error', 'message': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
