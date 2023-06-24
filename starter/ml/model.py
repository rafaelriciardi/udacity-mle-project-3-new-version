from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred

def evaluate_model_slices(model, test_data, cat_features, encoder, lb):
    """ Evaluate the model on sliced data and save it to txt file.

    Inputs
    ------
    model : object
        Trained machine learning model.
    test_data : pd.DataFrame
        Test data.
    cat_features : list
        List of categorial features.
    encoder : object
        Enconder from process_data function.
    lb: object
        Fitted Label Binarizer from process_data function
    Returns
    -------
    None
    """

    metrics_json = {}

    for feature in cat_features:
        feature_json  = {}
        for category in test_data[feature].unique():
            subset = test_data.loc[test_data[feature] == category]

            X_test, y_test, _, _ = process_data(subset, 
                                                categorical_features=cat_features, 
                                                label="salary", 
                                                training=False, 
                                                encoder=encoder, 
                                                lb=lb
                                                )

            y_pred = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            
            feature_json[category] = {'precision': precision, 
                                        'recall': recall,
                                        'fbeta': fbeta}
            
        metrics_json[feature] = feature_json
    
    with open('../model/slice_output.txt', 'w') as f:
        f.write(str(metrics_json))