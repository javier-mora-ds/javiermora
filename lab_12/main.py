from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Cargar el modelo
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = FastAPI()

class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
async def home():
    return {
        "modelo": "XGBoost Classifier",
        "problema": "Predicción de potabilidad del agua",
        "entrada": "Mediciones de 9 características del agua",
        "salida": "0 (no potable) o 1 (potable)"
    }

@app.post("/potabilidad/")
async def predict_potability(sample: WaterSample):
    try:
        # Convertir los datos de entrada en un array numpy
        features = np.array([[
            sample.ph,
            sample.Hardness,
            sample.Solids,
            sample.Chloramines,
            sample.Sulfate,
            sample.Conductivity,
            sample.Organic_carbon,
            sample.Trihalomethanes,
            sample.Turbidity
        ]])
        
        # Realizar la predicción
        prediction = model.predict(features)
        
        # Devolver el resultado
        return {"potabilidad": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)