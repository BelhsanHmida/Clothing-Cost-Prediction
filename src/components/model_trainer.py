import sys
import os 
from  dataclasses import dataclass

import pandas as pd
import numpy as np
import dill

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score



from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_objects
from src.components.HyperparamTuning import Modeltuning


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                
    "AdaBoostRegressor":AdaBoostRegressor,
    "GradientBoostingRegressor":GradientBoostingRegressor,
    "RandomForestRegressor":RandomForestRegressor,
    "LinearRegression":LinearRegression,
    "KNeighborsRegressor": KNeighborsRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    
        }
            

            print('Calling Evaluate mode')
            model_tuning=Modeltuning()
            model_report:dict = model_tuning.initiate_tuning(X_train, X_test, y_train, y_test)
            print('@$$$@out of the model_rep')
            best_model = max(model_report, key=model_report.get)
            best_model_score=model_report[best_model]

            if best_model_score<0.6:
                raise CustomException('No Sufficient Model was found ')
            logging.info(f'Best found model on both training and testing dataset')
            
            save_objects(
                file_path= self.model_trainer_config.trained_model_file_path , 
                obj=best_model
            )  
            
            #X_test=pd.DataFrame(X_test)
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
                               
            return best_model
        except Exception as e:
            raise CustomException(e,sys)


        

