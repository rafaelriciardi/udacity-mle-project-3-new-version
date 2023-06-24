# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import joblib
import logging
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, evaluate_model_slices

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add code to load in the data.
logging.info("Loading data")
data = pd.read_csv("../data/cleaned_census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logging.info("Spliting data")
train, test = train_test_split(data, test_size=0.20)

logging.info("Preprocessing data")
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
logging.info("Training model")
trained_model = train_model(X_train, y_train)

logging.info("Scoring model on test data")
y_pred = inference(trained_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Test Scores:\nPrecision: {precision: .2f} \nRecall: {recall: .2f} \nFbeta: {fbeta: .2f}")

logging.info("Evaluating model performance on sliced data and saving to file")
evaluate_model_slices(trained_model, test, cat_features, encoder, lb)

logging.info("Saving trained model")
joblib.dump(trained_model, '../model/model.pkl')

