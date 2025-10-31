import sys
import os
import numpy as np
import json
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model,tune_model,save_object

import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class modelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts","model.pkl")
    model_report_path: str = os.path.join("artifacts", "model_report.json")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config= modelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("==== Model Training Started ====")
            
            x_train, y_train, x_test, y_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("train test spliting done")

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False, allow_writing_files=False)
            }

            params = {
                "LinearRegression": {},
                "Lasso": {"alpha": [0.1, 1, 10]},
                "Ridge": {"alpha": [0.1, 1, 10]},
                "ElasticNet": {"alpha": [0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]},
                "SVR": {"kernel": ["rbf", "poly"], "C": [1, 10]},
                "KNN": {"n_neighbors": [3, 5, 7]},
                "DecisionTree": {"max_depth": [None, 5, 10]},
                "RandomForest": {"n_estimators": [50, 100]},
                "GradientBoosting": {"learning_rate": [0.1, 0.01]},
                "AdaBoost": {"n_estimators": [50, 100]},
                "XGBoost": {"learning_rate": [0.1, 0.01], "n_estimators": [50, 100]},
                "CatBoost": {"depth": [4, 6, 8]}
            }
            
            report={}
            best_model = None
            best_model_name= None
            best_score= -np.inf

            for model_name,model in models.items():
                logging.info(f"Training model: {model_name}")

                tuned_model= tune_model(model,params[model_name],x_train,y_train)

                logging.info("model trained successfully")

                metrics = evaluate_model(tuned_model, x_train, y_train, x_test, y_test)
                report[model_name]= metrics

                if metrics["test_r2"] > best_score:
                    best_model = tuned_model
                    best_model_name= model_name
                    best_score= metrics["test_r2"]
                
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_score}")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            
            with open(self.model_trainer_config.model_report_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info(f"Best Model: {best_model_name} saved")

            return(
                best_model_name, best_score, self.model_trainer_config.trained_model_file_path
            )
    

        except Exception as e:
            raise CustomException(e,sys)
            