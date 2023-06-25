import joblib
import pandas as pd

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.ml.model import inference

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

class Prediction_data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
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


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def welcome():
    return  {"greeting": "Welcome to census model API"}

@app.post("/predict")
async def model_predict(data: Prediction_data):

    data = dict(data)

    data_df = pd.DataFrame(data, columns=data.keys(), index=[0])

    data_df.columns = data_df.columns.str.replace("_", "-")

    X_predict, _, _, _ = process_data(data_df, 
                              categorical_features=cat_features, 
                              training=False, 
                              encoder=encoder)
    
    predictions = inference(model, X_predict)

    response = {"prediction": int(predictions)}

    return response