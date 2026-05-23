import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformer
from model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        try:
            df=pd.read_csv(r'C:\ML\EtoEMLProject\notebook\stud.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train Test Split Initiated')
            train_data,test_data = train_test_split(df,test_size=0.3,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of Data is Completed')
            return(
                self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    data_ingestion=DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
    data_transformer_obj=DataTransformer()
    transformed_train_data,transformed_test_data,_ = data_transformer_obj.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)
    model_trainer_obj=ModelTrainer()
    model_trainer_obj.initiate_model_trainer(transformed_train_data,transformed_test_data)


