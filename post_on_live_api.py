import requests

live_url = 'https://mle-project3.onrender.com'

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

print("\nExecuting GET request:")
get_response = requests.get(live_url)
print(get_response)
print(get_response.json())

print("\nExecuting POST request on 0 class example:")
class_0_response = requests.post(live_url + '/predict', 
                                 json = class_0_example)
print(class_0_response)
print(class_0_response.json())

print("\nExecuting POST request on 1 class example:")
class_1_response = requests.post(live_url + '/predict', 
                                 json = class_1_example)
print(class_1_response)
print(class_1_response.json())
