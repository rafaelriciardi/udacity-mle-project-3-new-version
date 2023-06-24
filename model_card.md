# Model Card

## Model Details
Model created by Rafael Riciardi. It is a Random Forest Classifier with default parameters from sklearn.

## Intended Use
The project has been developed for the Udacity's Machine Learning Engeneer nano degree program.
The task is to predict the salary of US citizens based on a list of attributes and automate it using a CI/CD pipeline with FastAPI and Heroku.


## Training Data
Census data are provided with the project under 'data/census.csv'.
It's originally obtained from https://archive.ics.uci.edu/ml/datasets/census+income.


## Evaluation Data
20% of the original data is used to evaluate the trained model, separated by a train_test_split method.


## Metrics
The metrics used to evaluate this task were precision, recall and fbeta. 

Precision:  0.73
Recall:     0.61
Fbeta:      0.66

## Ethical Considerations
As the model is trained on public census data, it is possible to suffer from any bias from the data. However, until the present moment, no ethical bias has been detected during the model's development.


## Caveats and Recommendations
The model was trained on old data, thus its recommended to update the training data to a more recent census due to data drift issues. Also, it could be good pratice to test another classsification tecniques and hyperparameter tunning to achieve better results. 
