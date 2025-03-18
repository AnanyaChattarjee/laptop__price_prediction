from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from pydantic import conint
from config import config
import pickle
import os
import numpy as np
import math
import uvicorn
from sklearn.preprocessing import PolynomialFeatures

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ENCODING_FILENAME),"rb") as file_handle :
    saved_features_encodings=pickle.load(file_handle)
with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NOMINAL_COLUMNS_IDX_FILENAME),"rb") as file_handle :
    nominal_columns_idx=pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.ORDINAL_COLUMNS_IDX_FILENAME),"rb") as file_handle :
    ordinal_columns_idx=pickle.load(file_handle)

with open(os.path.join(config.SAVED_ENCODINGS_PATH,config.NUMERIC_COLUMNS_IDX_FILENAME),"rb") as file_handle :
    numeric_columns_idx=pickle.load(file_handle)

def predict(X_transpose):

    nominal_columns_idx_copy = dict(zip(nominal_columns_idx.values(),nominal_columns_idx.keys()))
    nominal_features_ohe_values = list()

    for k,v in sorted(nominal_columns_idx_copy.items()):
        feature_unique_values_num = len(features_encodings[v])
        identity_mat = np.eye(feature_unique_values_num,feature_unique_values_num)
        feature_ohe_value = identity_mat[int(X_transpose[all_columns_idx[v]])]
        nominal_features_ohe_values.append(feature_ohe_value)

    X_transpose = np.concatenate((X_transpose[0:min(nominal_columns_idx.values())],
                                np.concatenate(nominal_features_ohe_values)))

    with open(os.path.join(config.SAVED_MODEL_PATH,config.SAVED_MODEL_FILE),"rb") as file_handle:
        thetas = pickle.load(file_handle)

    theta0_star = thetas[0]
    theta_star = thetas[1]
    theta_star = theta_star.reshape(theta_star.shape[0],)

    with open(os.path.join(config.SAVED_NORMALIZER_PATH,config.SAVED_NORMALIZER_FILE),"rb") as file_handle:
        normalizer = pickle.load(file_handle)

    X_transpose = X_transpose.reshape(1,-1)
    X_bar_transpose = normalizer.transform(X_transpose)
    y_hat = theta0_star + np.dot(X_bar_transpose,theta_star)

    return math.exp(y_hat.item())


app = FastAPI()

class InputFeatureVector(BaseModel):
    
    ScreenResolution: conint(ge = 0,le =39)
    Cpu: conint(ge = 0,le =117)
    Ram : conint(ge =2,le =64)
    Memory: conint(ge =0, le =39)
    Gpu: conint(ge=0 , le =109)
    TypeName: conint(ge=0, le =5)

@app.get("/")
def home_page():
    return "This Web API Predicts the price of used laptop in India"
    
@app.post("/predict_price")
def perform_prediction(x_bar_transpose:InputFeatureVector):
    feature1 = x_bar_transpose.ScreenResolution
    feature2 = x_bar_transpose.Cpu
    feature3 = x_bar_transpose.Ram
    feature4 = x_bar_transpose.Memory
    feature5 = x_bar_transpose.Gpu
    feature6 = x_bar_transpose.TypeName

    y_hat = predict(feature1,feature2,feature3,feature4,feature5,feature6)
    return "the price of the laptop is :{}".format(y_hat)

if __name__ == "__main__":
    uvicorn.run(app)
