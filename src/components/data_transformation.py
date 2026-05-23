from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import os
import pandas as pd
import numpy as np
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class DataTransformerConfig():
    preprocessor_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformer():
    def __init__(self):
        self.data_transformer_config=DataTransformerConfig()
    
    def get_data_transformer(self):
        logging.info('Entered Data Transformation Module')
        try:

            numerical_features=['writing_score','reading_score']
            categorical_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder())])
            preprocessor=ColumnTransformer([('numerical_pipeline',num_pipeline,numerical_features),('categorical_pipeline',cat_pipeline,categorical_features)])
            return preprocessor
        except Exception as e:

            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            preprocessing_obj=self.get_data_transformer()
            target_column='math_score'
            input_feature_train_df = train_data.drop([target_column],axis=1)
            output_feature_train_df = train_data[target_column]
            input_feature_test_df = test_data.drop([target_column],axis=1)
            output_feature_test_df = test_data[target_column]
            logging.info(f'Training and Test Data divided into Independent & Dependent Features with {target_column} selected as dependent Feature')
            processed_train_data = preprocessing_obj.fit_transform(input_feature_train_df)
            processed_test_data = preprocessing_obj.transform(input_feature_test_df)
            logging.info('Data Transformation Completed for Test and Training Independent Data')
            train_arr = np.c_[processed_train_data,np.array(output_feature_train_df)]
            test_arr = np.c_[processed_test_data,np.array(output_feature_test_df)]
            save_object(
                file_path=self.data_transformer_config.preprocessor_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,test_arr,self.data_transformer_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


