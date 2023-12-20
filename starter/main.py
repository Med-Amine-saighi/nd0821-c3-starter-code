"""
FastAPI app to serve ML model
"""

import os 
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starter.ml.data import process_data
from typing import Literal
import pickle


PATH_SAVE = 'starter/model/'

app = FastAPI(
    title="RandomForest API",
    description = "This app takes input and returns predictions"
)

class ModelInput(BaseModel):
    """
    Input data schema for predictions endpoint

    Attributes:
        - Age: The age of the individual.
        - Workclass: The working class of the individual.
        - Fnlgt: Final weight.
        - Education: The education level of the individual.
        - Education_num: A numeric representation of the education level.
        - Marital_status: The marital status of the individual.
        - Occupation: The occupation of the individual.
        - Relationship: The relationship status of the individual.
        - Race: The race of the individual.
        - Sex: The gender of the individual.
        - Capital_gain: Capital gain.
        - Capital_loss: Capital loss.
        - Hours_per_week: The number of hours worked per week.
        - Native_country: The native country of the individual.
    """
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal['Bachelors',
                       'HS-grad',
                       '11th',
                       'Masters',
                       '9th',
                       'Some-college',
                       'Assoc-acdm',
                       '7th-8th',
                       'Doctorate',
                       'Assoc-voc',
                       'Prof-school',
                       '5th-6th',
                       '10th',
                       'Preschool',
                       '12th',
                       '1st-4th']
    education_num: int
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal['United-States',
                            'Cuba',
                            'Jamaica',
                            'India',
                            'Mexico',
                            'Puerto-Rico',
                            'Honduras',
                            'England',
                            'Canada',
                            'Germany',
                            'Iran',
                            'Philippines',
                            'Poland',
                            'Columbia',
                            'Cambodia',
                            'Thailand',
                            'Ecuador',
                            'Laos',
                            'Taiwan',
                            'Haiti',
                            'Portugal',
                            'Dominican-Republic',
                            'El-Salvador',
                            'France',
                            'Guatemala',
                            'Italy',
                            'China',
                            'South',
                            'Japan',
                            'Yugoslavia',
                            'Peru',
                            'Outlying-US(Guam-USVI-etc)',
                            'Scotland',
                            'Trinadad&Tobago',
                            'Greece',
                            'Nicaragua',
                            'Vietnam',
                            'Hong',
                            'Ireland',
                            'Hungary',
                            'Holand-Netherlands']
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 50,
                "workclass": 'Self-emp-not-inc',
                "fnlgt": 83311,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": " Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": 'United-States'
            }
        }

@app.on_event("startup")
async def startup_event():
    """
    This function loads the trained model, the encoder and the labilizer 
     
    """
    model, encoder, labilizer = load_objects()
    if model is not None:
        global loaded_model, loaded_encoder, loaded_labilizer
        loaded_model, loaded_encoder, loaded_labilizer = model, encoder, labilizer

        

def load_objects():
    """
    Loads the saved model, encoder, and labelizer.

    Returns:
    tuple: loaded_model, loaded_encoder, loaded_labilizer
    """
    loaded_model, loaded_encoder, loaded_labilizer = None, None, None
    try:
        with open(os.path.join(PATH_SAVE, "trained_model.pkl"), "rb") as model_file, \
            open(os.path.join(PATH_SAVE, "encoder.pkl"), "rb") as encoder_file, \
            open(os.path.join(PATH_SAVE, "labelizer.pkl"), "rb") as labelizer_file:
            
            loaded_model = pickle.load(model_file)
            loaded_encoder = pickle.load(encoder_file)
            loaded_labilizer = pickle.load(labelizer_file)

    except (FileNotFoundError, EOFError, pickle.PickleError) as e:

        print(f"Error loading saved objects: {e}")

    return loaded_model, loaded_encoder, loaded_labilizer


@app.get("/")
def read_root():
    """
    Returns greetings.

    Returns:
        dict: Welcome greetings.
    """
    return {
        "message": "Hi, this is a FastAPI app for udacity training !!"}

@app.post("/predict")
def predict(input: ModelInput):
    """
    Generate predictions.

    Args:
        input (ModelInput): input data

    Returns:
        dict: Predictions
    """
    Input_data = {'age': input.age,
                  'workclass': input.workclass,
                  'fnlgt': input.fnlgt,
                  'education': input.education,
                  'education-num': input.education_num,
                  'marital-status': input.marital_status,
                  'occupation': input.occupation,
                  'relationship': input.relationship,
                  'race': input.race,
                  'sex': input.sex,
                  'capital-gain': input.capital_gain,
                  'capital-loss': input.capital_loss,
                  'hours-per-week': input.hours_per_week,
                  'native-country': input.native_country,
                  }

    # Convert input data into a dataframe
    input_df = pd.DataFrame([Input_data])
    # set categorical features
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Load objects
    with open(os.path.join(PATH_SAVE, "trained_model.pkl"), "rb") as model_file, \
            open(os.path.join(PATH_SAVE, "encoder.pkl"), "rb") as encoder_file, \
            open(os.path.join(PATH_SAVE, "labelizer.pkl"), "rb") as lb_file:
        
        model = pickle.load(model_file)
        encoder = pickle.load(encoder_file)
        lb = pickle.load(lb_file)

    
    X, _, _, _ = process_data(input_df,
                              categorical_features=categorical_features,
                              training=False,
                              encoder=encoder,
                              lb=lb
                              )
    
    # calculate predictions
    pred = model.predict(X)
    # get the encoded prediction
    pred = lb.inverse_transform(pred)[0]
    input_df["prediction"] = pred
    return input_df.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)