from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from io import BytesIO
import pandas as pd
import math

app = FastAPI()

# Dummy model class
class InputData:
    def __init__(self, NIM, JenisKelamin, SKS, IPK, PekerjaanAyah, PekerjaanIbu):
        self.NIM = NIM
        self.JenisKelamin = JenisKelamin
        self.SKS = SKS
        self.IPK = IPK
        self.PekerjaanAyah = PekerjaanAyah
        self.PekerjaanIbu = PekerjaanIbu

def predict(data: InputData):
    return {
        "prediction": "LULUS" if data.IPK >= 2.5 else "TIDAK LULUS",
        "label": 1 if data.IPK >= 2.5 else 0,
        "probability": round(min(max(data.IPK / 4.0, 0), 1), 2)
    }

def sanitize_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(i) for i in obj]
    else:
        return obj

@app.get("/")
def home():
    return {"message": "FastAPI jalan"}

@app.post("/predict_excel")
async def predict_excel(
    file: UploadFile = File(...),
    page: int = Query(1, description="Page number (starts from 1)")
):
    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))

        required_columns = ["NIM", "Jenis Kelamin", "Pekerjaan Ayah", "Pekerjaan Ibu", "SKS", "IPK"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Kolom berikut tidak ditemukan di file Excel: {', '.join(missing_columns)}"
            )

        results = []

        for _, row in df.iterrows():
            try:
                data = InputData(
                    NIM=str(row["NIM"]),
                    JenisKelamin=row["Jenis Kelamin"],
                    SKS=int(row["SKS"]),
                    IPK=float(row["IPK"]),
                    PekerjaanAyah=row["Pekerjaan Ayah"],
                    PekerjaanIbu=row["Pekerjaan Ibu"]
                )
                result = predict(data)

                row_result = row.to_dict()
                row_result.update({
                    "prediction": result["prediction"],
                    "label": result["label"],
                    "probability": result["probability"]
                })
                results.append(row_result)

            except Exception as row_err:
                results.append({
                    "error": str(row_err),
                    "original_data": row.to_dict()
                })

        # Pagination
        page_size = 50
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_results = sanitize_json(results[start:end])

        return JSONResponse(content={
            "success": True,
            "page": page,
            "per_page": page_size,
            "total": total,
            "total_pages": math.ceil(total / page_size),
            "results": paginated_results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses file Excel: {str(e)}")
