"""
Script for testing a deployed API by sending a sample request and checking the response.

The script sends a sample data input to a specified API endpoint and checks if the response status code is 200.
It logs information about the status code, response content, and the time taken for the request.

"""

import requests
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def api_test(url, data_input):
    """
    Tests the deployed Application

    Parameters:
        - url (str): URL of the API endpoint.
        - data_input (dict): The input data to be sent with the POST request.
    Returns:
        None
    Raises:
        - requests.exceptions.RequestException: If the request fails.
    """
    try:
        start_time = time.time()
        resp = requests.post(url, json=data_input)
        # check if the request was successful
        resp.raise_for_status()

        # elapsed time
        elapsed_time = time.time() - start_time

        # Log responses
        logging.info("Response Content:")
        logging.info(resp.text)
        logging.info("Prediction")
        logging.info(resp.json()[0]["prediction"])
        logging.info("Testing Render app")
        logging.info(f"Status code: {resp.status_code}")
        logging.info(f"Time taken: {elapsed_time:.4f} seconds")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")   


if __name__ == "__main__":
    # URL endpoint
    api_url = "https://udacity-app.onrender.com/predict"

    # input data sample
    data_input = {
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

    api_test(api_url, data_input)