import sys
import os
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV



def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
     try:
        with open(file_path, 'rb') as file_obj:
            model = dill.load(file_obj)
            return model
     
     except Exception as e:
          raise CustomException(e,sys)
    
    
def evaluate_model(model, x_train, y_train, x_test, y_test):
        # model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        return {
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "mae": mean_absolute_error(y_test, y_test_pred)
        }


def tune_model(model, params, x_train, y_train):
        try:
            gs = GridSearchCV(model, params, cv=3, scoring="r2", n_jobs=-1)
            gs.fit(x_train, y_train)
            return gs.best_estimator_
        except Exception:
            return model  