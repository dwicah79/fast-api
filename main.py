from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # izinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],  # izinkan semua method (GET, POST, dll)
    allow_headers=["*"],  # izinkan semua header
)

# Load model dan encoder
model = joblib.load("Model/model_decision_tree.pkl")
encoders = joblib.load("Model/label_encoders.pkl")
label_mapping = joblib.load("Model/label_mapping.pkl")  # {0: "Tidak Putus Studi", 1: "Putus Studi"}

# Fungsi untuk mengekstrak angkatan dari NIM (2 digit pertama)
def get_angkatan_from_nim(nim: str) -> str:
    return nim[:2]

class InputData(BaseModel):
    NIM: str  # Input NIM untuk menentukan angkatan
    JenisKelamin: str  # Contoh: "L" atau "P"
    SKS: int
    IPK: float
    PekerjaanAyah: str
    PekerjaanIbu: str

    @validator("NIM")
    def nim_min_length(cls, v):
        if len(v) < 2:
            raise ValueError("NIM harus minimal 2 karakter")
        return v

@app.post("/predict")
def predict(data: InputData):
    try:
        angkatan = get_angkatan_from_nim(data.NIM)
        print(f"Angkatan dari NIM: {angkatan}")

        # Mapping input ke nama kolom yang sesuai dengan encoder
        input_dict = {
            "angkatan": angkatan,
            "Jenis Kelamin": data.JenisKelamin,
            "Pekerjaan Ayah": data.PekerjaanAyah,
            "Pekerjaan Ibu": data.PekerjaanIbu
        }
        print("Input dict:", input_dict)

        encoded_features = []

        # Urutan fitur harus sama dengan saat training:
        # ['angkatan', 'Jenis Kelamin', 'SKS', 'IPK', 'Pekerjaan Ayah', 'Pekerjaan Ibu']
        
        # Encode 'angkatan' dan 'Jenis Kelamin'
        for col in ['angkatan', 'Jenis Kelamin']:
            encoder = encoders.get(col)
            if encoder is None:
                raise HTTPException(status_code=500, detail=f"Encoder untuk '{col}' tidak ditemukan.")
            try:
                encoded = encoder.transform([input_dict[col]])[0]
            except Exception as e:
                print(f"Error encode kolom '{col}' dengan nilai '{input_dict[col]}': {e}")
                raise HTTPException(status_code=400, detail=f"Nilai '{input_dict[col]}' tidak dikenali untuk kolom '{col}'.")
            print(f"Encoded {col}: {encoded}")
            encoded_features.append(encoded)

        # Tambahkan fitur numerik
        encoded_features.append(data.SKS)
        encoded_features.append(data.IPK)
        print("Fitur numerik:", [data.SKS, data.IPK])

        # Encode 'Pekerjaan Ayah' dan 'Pekerjaan Ibu'
        for col in ['Pekerjaan Ayah', 'Pekerjaan Ibu']:
            encoder = encoders.get(col)
            if encoder is None:
                raise HTTPException(status_code=500, detail=f"Encoder untuk '{col}' tidak ditemukan.")
            try:
                encoded = encoder.transform([input_dict[col]])[0]
            except Exception as e:
                print(f"Error encode kolom '{col}' dengan nilai '{input_dict[col]}': {e}")
                raise HTTPException(status_code=400, detail=f"Nilai '{input_dict[col]}' tidak dikenali untuk kolom '{col}'.")
            print(f"Encoded {col}: {encoded}")
            encoded_features.append(encoded)

        print("Fitur akhir (encoded + numerik):", encoded_features)

        X = np.array([encoded_features])

        prediction = model.predict(X)[0]
        pred_label = label_mapping.get(prediction, "Tidak Diketahui")
        print(f"Prediksi model: {prediction} ({pred_label})")

        # Predict probabilitas, cek apakah model support predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            print("Probabilitas prediksi:", proba)
        else:
            proba = [None, None]

        return {
            "prediction": pred_label,
            "label": int(prediction),
            "probability": {
                "Tidak Putus Studi": proba[0],
                "Putus Studi": proba[1]
            }
        }

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print("Error saat prediksi:", e)
        raise HTTPException(status_code=500, detail=str(e))
