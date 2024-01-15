# Script to train machine learning model.
import pickle
import pandas as pd
from ml.data import process_data
from sklearn.model_selection import train_test_split
from ml.model import train_model, compute_model_metrics, inference, compute_slices

# Add the necessary imports for the starter code.
data = pd.read_csv('starter/data/census.csv')
data.columns = data.columns.str.replace(' ', '')
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
X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features,
                                                     label="salary", training=False,
                                                     encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(lb, open(f'starter/model/labelizer.pkl', "wb"))
pickle.dump(encoder, open(f'starter/model/encoder.pkl', "wb"))
pickle.dump(model, open(f'starter/model/trained_model.pkl', "wb"))

# assess the performance of the model
predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, predictions)
print(f'precision, recall, fbeta : {precision, recall, fbeta}')

for feat in cat_features:
    compute_slices(test, feat, y_test, predictions)