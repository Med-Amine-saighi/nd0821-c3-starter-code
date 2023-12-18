"""
Module for testing ML model

"""
import os
import pytest
import joblib
import logging
import pandas as pd
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split
from model import compute_model_metrics, inference, compute_slices

PATH_DATA = "data/census.csv"
PATH_MODEL = "model/trained_model.pkl"

@pytest.fixture()
def check_data():
    """
    Loads dataset

    Returns:
        pd.DataFrame: dataset
    """
    dataset = pd.read_csv(PATH_DATA)
    return dataset

@pytest.fixture()
def check_model():
    """
    Loads the trained ML model

    Returns:
        object: Trained ML model
    """
    # load model
    return joblib.load(PATH_MODEL)
    
@pytest.fixture()
def get_features():
    """
    Provides categorical features.

    Returns:
        list: List of categorical features.
    """
    cat_features = ["workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country"]
    return cat_features

@pytest.fixture()
def get_train_test_split(data, features):
    """
    Provides train and test data.

    Args:
        - data (pd.DataFrame): Loaded dataset.
        - features (list): List of categorical features.

    Returns:
        tuple: Training and testing data.
    """
    train, test = train_test_split(data,
                                   test_size=0.20,
                                   random_state=0,
                                   )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=features,
        label="salary",
        training=True,
        encoder=encoder,
        lb=lb,
    )
    return X_train, X_test, y_train, y_test

def test_data():
    """
    Test import data
    
    """
    try:
        dataset = pd.read_csv(PATH_DATA)
    except FileNotFoundError :
        logging.error("Check the dataset path !!!")
        raise FileNotFoundError
    
    try:
        assert dataset.shape[0] != 0 and dataset.shape[1] != 0

    except AssertionError:
        logging.error("Empty Dataset !!!")
        raise AssertionError
    
def test_inference(model, get_train_test_split):
    """
    Verify that the model can do inference

    Args:
        - model (object): Trained ML model
        - get_train_test_split (tuple): train test data.

    Raises:
        - AssertionError
    """
    _, X_test, _, _ = get_train_test_split
    try:
        assert model.predict(X_test)
    except BaseException:
        logging.error("you didn't train the model !!")

def test_metrics(model, data_train_test):
    """
    Verify that we can compute metrics.

    Args:
        - model (object): Trained ML model.
        - get_train_test_split (tuple): train test data.

    Raises:
        - AssertionError
    """
    try:
        _, X_test, _, y_test = data_train_test
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)
    except AssertionError:
        logging.error("Can't calculate the metrics !!")
        raise AssertionError