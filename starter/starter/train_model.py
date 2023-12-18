# Script to train machine learning model.
import pickle
import pandas as pd
from ml.data import process_data
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics, inference

# Add the necessary imports for the starter code.
data = pd.read_csv('../data/census.csv')

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
)


# Train and save a model.
path = "../model"

model = train_model(X_train, y_train)
pickle.dump(lb, open(f'{path}/labelizer.pkl', "wb"))
pickle.dump(encoder, open(f'{path}/encoder.pkl', "wb"))
pickle.dump(model, open(f'{path}/trained_model.pkl'), "wb")

# assess the performance of the model
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
