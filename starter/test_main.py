"""
Tests for the FastAPI app.

"""
from main import app
from fastapi.testclient import TestClient
import os
import logging
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__)))

client = TestClient(app)


@pytest.fixture
def sample_input():
    """
    A Fixture that provides sample input data class 1

    Returns:
        dict: input data.
    """
    sample_input = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Separated",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    return sample_input


@pytest.fixture
def sample_input2():
    """
    A Fixture that provides sample input data class 0

    Returns:
        dict: input data.
    """
    sample_input2 = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    return sample_input2


def test_welcome_message():
    """
    Tests the greeting message ("/").

    Checks the greeting message 
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()[
        'message'] == "Hi, this is a FastAPI app for udacity training !!"


def test_model_inference_class_0(sample_input2):
    """
    Tests the inference endpoint ("/predict") for class 0
    """
    # endpoint for predict 
    r = client.post("/predict/", json=sample_input2)
    # success
    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_input2["age"]
    assert r.json()[0]["fnlgt"] == sample_input2["fnlgt"]
    assert r.json()[0]["prediction"] == ' >50K'

def test_model_inference_class1(sample_input):
    """
    Tests the inference endpoint ("/predict") for class 1

    """
    # endpoint for predict 
    r = client.post("/predict", json=sample_input)
    # success
    assert r.status_code == 200
    assert r.json()[0]["age"] == sample_input["age"]
    assert r.json()[0]["fnlgt"] == sample_input["fnlgt"]
    assert r.json()[0]["prediction"] == ' <=50K'



def test_incomplete():
    """
    Test for incomplete model inference query.
    """
    data = {
        "occupation": "Prof-specialty",
        "race": "Black",
        "capital_gain": 0,
        "education": "HS-grad"}
    r = client.post("/predict", json=data)
    assert r.status_code == 422
    assert 'prediction' not in r.json()["detail"][0].keys()

    logging.warning(
        f"Must be 14 features")