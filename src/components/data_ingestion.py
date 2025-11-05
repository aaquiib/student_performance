import sys
import os
from src.exception import CustomException
from src.logger import logging
import dill

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import dataTransformation
from src.components.model_trainer import modelTrainer

@dataclass
class dataIngestionConfig:
    train_data_path: str= os.path.join("artifacts","train.csv")
    test_data_path: str= os.path.join("artifacts","test.csv")
    raw_data_path: str= os.path.join("artifacts","data.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config= dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("entered data ingestion initiation")
        try:
            df = pd.read_csv("notebook\\data\\stud.csv")
            logging.info("data read as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initaited")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("ingestion of data completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj=dataIngestion()
    train_path, test_path= obj.initiate_data_ingestion()

    data_transformation_obj= dataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path= data_transformation_obj.initiate_data_transformation(train_path, test_path)

    model_trainer_obj= modelTrainer()
    model_trainer_obj.initiate_model_trainer(train_arr,test_arr,preprocessor_obj_file_path)
    
    #testing prediction
    # new_data = pd.DataFrame([{
    # "gender": "female",
    # "race_ethnicity": "group C",
    # "parental_level_of_education": "bachelor's degree",
    # "lunch": "standard",
    # "test_preparation_course": "completed",
    # "reading_score": 100,
    # "writing_score": 100
    # }])

    # # Load model
    # with open("artifacts/model.pkl", "rb") as f:
    #     model = dill.load(f)
    # # Load preprocessor
    # with open("artifacts/preprocessor.pkl", "rb") as f:
    #     preprocessor = dill.load(f)
    # processed_input = preprocessor.transform(new_data)
    # # Predict
    # prediction = model.predict(processed_input)
    # print("Predicted Math Score:", prediction[0])
