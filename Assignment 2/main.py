import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model

app = FastAPI(title="Adult Income Prediction API")

model = load_model("best_pipeline")


class AdultInput(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
def home():
    return {"message": "Adult Income Prediction API is running."}


@app.post("/predict")
def predict(data: AdultInput):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = predict_model(model, data=input_df)

    return {
        "prediction": prediction["prediction_label"][0]
    }
