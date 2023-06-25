import sys
sys.path.append('../')

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

class_0_example = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

class_1_example = {
    "age": 38,
    "workclass": "Federal-gov",
    "fnlgt": 125933,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "Iran"
}

def test_welcome():
    response = client.get("/")
    #print(response.json())
    assert response.status_code == 200
    assert response.json()["greeting"] == "Welcome to census model API"

def test_class_0():
    response = client.post("/predict", json=class_0_example)
    #print(response.json())
    assert response.status_code == 200
    assert response.json()["prediction"] == 0

def test_class_1():
    response = client.post("/predict", json=class_1_example)
    #print(response.json())
    assert response.status_code == 200
    assert response.json()["prediction"] == 1

if __name__ == '__main__':
    test_welcome()
    test_class_0()
    test_class_1()

    