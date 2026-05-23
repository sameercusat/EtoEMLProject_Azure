import os
import dill
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
     report ={}
     try:
          for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param_grid = params[model_name]
            rcv=RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=10,n_jobs=-1,random_state=42)
            print("Model Name: ",model_name)
            logging.info(f'{model_name} training starts')
            rcv.fit(x_train,y_train)
            model.set_params(**rcv.best_params_)
            logging.info(f'Best Params for {model_name} is {rcv.best_params_}')
            model.fit(x_train,y_train)
            logging.info('Training Completed')
            logging.info('Training Data Analysis') 
            y_train_pred = model.predict(x_train)
            r2_score_train = r2_score(y_train,y_train_pred)
            logging.info(f" Training R2 Score for {model_name} is {r2_score_train}")
            logging.info('Training Data Analysis Completed')
            logging.info(f"Test Data Analysis Started for {model_name}")
            y_test_pred=model.predict(x_test)
            r2_score_test=r2_score(y_test,y_test_pred)
            logging.info(f" Test R2 Score for {model_name} is {r2_score_test}")
            logging.info('Test Data Analysis Completed')
            report[model_name]=r2_score_test
          return report
     except Exception as e:
         raise CustomException(e,sys)
     
def load_object(file_path):
    try:
        with open(file_path,'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)
