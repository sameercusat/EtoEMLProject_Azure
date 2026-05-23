from dataclasses import dataclass
import os
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model
import sys
from src.utils import save_object


@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()
    
    def initiate_model_trainer(self,tranformed_train_data,transformed_test_data):
        try:
     
            logging.info('Model Training Process Begins')
            models = {
                'LinearRegressor':LinearRegression(),
                'SupportVectorRegressor':SVR(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'KNNRegressor':KNeighborsRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'CatBRegressor':CatBoostRegressor(),
                'XGBRegressor':XGBRegressor(),
                'Ridge':Ridge(),
                'Lasso':Lasso()
            }
            param_grids = {

                    'LinearRegressor': {
                        'fit_intercept': [True, False],
                        'positive': [True, False]
                    },

                    'SupportVectorRegressor': {
                        'kernel': ['linear', 'rbf', 'poly'],
                        'C': [0.1, 1, 10, 100],
                        'epsilon': [0.01, 0.1, 0.5],
                        'gamma': ['scale', 'auto']
                    },

                    'DecisionTreeRegressor': {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                        'max_depth': [None, 5, 10, 20],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },

                    'KNNRegressor': {
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                        'p': [1, 2]   # Manhattan (1), Euclidean (2)
                    },

                    'RandomForestRegressor': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                        'max_features': ['sqrt', 'log2']
                    },

                    'CatBRegressor': {
                        'iterations': [100, 300],
                        'learning_rate': [0.01, 0.1],
                        'depth': [4, 6, 10],
                        'l2_leaf_reg': [1, 3, 5],
                        'verbose': [0]
                    },

                    'XGBRegressor': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 6, 10],
                        'subsample': [0.7, 1.0],
                        'colsample_bytree': [0.7, 1.0]
                    },

                    'Ridge': {
                        'alpha': [0.01, 0.1, 1, 10, 100],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                    },

                    'Lasso': {
                        'alpha': [0.001, 0.01, 0.1, 1, 10],
                        'selection': ['cyclic', 'random']
                    }
                }
            x_train = tranformed_train_data[:,:-1]
            y_train = tranformed_train_data[:,-1]
            x_test =  transformed_test_data[:,:-1]
            y_test = transformed_test_data[:,-1]
            report = evaluate_model(x_train,y_train,x_test,y_test,models,param_grids)
            logging.info('Final R2 Score Report Generated')
            best_model_score = max(report.values())
            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]
            logging.info(f'Best Model is {best_model_name} & Score is {best_model_score}')
            best_model = models[best_model_name]
            save_object(file_path=self.modelTrainerConfig.model_path,obj=best_model)
            logging.info('Model saved in pickle file')
        except Exception as e:
            raise CustomException(e,sys)


        




            

