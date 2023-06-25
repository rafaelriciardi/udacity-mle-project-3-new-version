import pytest
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model


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

data = pd.read_csv('data/cleaned_census.csv')

train, test = train_test_split(data, test_size=0.20)


#Check the processing functions
def test_process_data():
    train, test = train_test_split(data, test_size=0.20)
    
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    assert len(data) == len(train) + len(test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

#Check if training is returning the correct object type
def test_training_step():
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    trained_model = train_model(X_train, y_train)
    assert isinstance(trained_model, RandomForestClassifier)

#Check if saved model file is correct
def test_saved_model():
    # Load the model
    saved_model = joblib.load("model/model.pkl")
    assert isinstance(saved_model, RandomForestClassifier)