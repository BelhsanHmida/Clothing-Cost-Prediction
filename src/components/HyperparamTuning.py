import sys
import os 
from  dataclasses import dataclass
#import pickle

import pandas as pd
import numpy as np
import dill
import optuna

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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



from src.logger import logging
from src.exceptions import CustomException



Trials=20
class Modeltuning:
    #def __init__(self,models):
    #    self.models=models
    def initiate_tuning(self,X_train, X_test, y_train, y_test):
        Tuned_models={}
        report={}
        try:
            def objective(trial):
                # Define hyperparameters to optimize
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
                
                # Create and train AdaBoostRegressor with the suggested hyperparameters
                model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                model.fit(X_train, y_train)
                
                # Predict and calculate R^2 score on the test set
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                
                return r2

            # Create study object and optimize the objective function
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=Trials)
            p=study.best_trial.params
            model=AdaBoostRegressor(**p)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            Tuned_models["AdaBoostRegressor"]=model
            report[model]=r2
            logging.info('AdaBoostRegressor has been Tuned')

            def objective(trial):
                # Define hyperparameters to optimize
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'verbosity': 0,
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
                }
                
                # Initialize XGBRegressor with the suggested hyperparameters
                xgb_reg = XGBRegressor(**params)
                
                # Train XGBRegressor
                xgb_reg.fit(X_train, y_train)
                
                # Predict and calculate R^2 score on the test set
                predictions = xgb_reg.predict(X_test)
                r2 = r2_score(y_test, predictions)
                
                return r2


            # Create study object and optimize the objective function
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=Trials)
            p=study.best_trial.params
            model=XGBRegressor(**p)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            Tuned_models["XGBRegressor"]=model
            report[model]=r2
            logging.info('XGB has been Tuned')

            
            def objective_(trial):
                # Define hyperparameters to optimize
                max_depth = trial.suggest_int('max_depth', 1, 32)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                max_features = trial.suggest_uniform('max_features',0.01,1.0)
                
                # Create and train DecisionTreeRegressor with the suggested hyperparameters
                model = DecisionTreeRegressor(max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=42)
                model.fit(X_train, y_train)
                
                # Predict and calculate R^2 score on the test set
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                
                return r2

        # Create study object and optimize the objective function
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_,n_trials=Trials)
            p=study.best_trial.params
            model=DecisionTreeRegressor(**p)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            Tuned_models["DecisionTreeRegressor"]=model
            report[model]=r2
            logging.info('DecisionTreeRegressor has been Tuned')   

            def objective(trial):
                # Define hyperparameters to optimize
                n_neighbors = trial.suggest_int('n_neighbors', 1, 10)
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                p = trial.suggest_int('p', 1, 2)  # p = 1 for Manhattan distance, p = 2 for Euclidean distance
                
                # Create and train KNeighborsRegressor with the suggested hyperparameters
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
                model.fit(X_train, y_train)
                
                # Predict and calculate R^2 score on the test set
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                
                return r2

            # Create study object and optimize the objective function
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=Trials)
            p=study.best_trial.params
            model=KNeighborsRegressor(**p)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            Tuned_models["KNeighborsRegressor"]=model
            report[model]=r2
            logging.info('KNeighborsRegressor has been Tuned') 



        

            def objective(trial):
                # Define hyperparameters to optimize
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 10)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                if max_features == 'auto':
                     max_features = None
                # Create and train RandomForestRegressor with the suggested hyperparameters
                model = RandomForestRegressor(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=42)
                
                model.fit(X_train, y_train)
                
                # Predict and calculate R^2 score on the test set
                predictions = model.predict(X_test)
                r2 = r2_score(y_test, predictions)
                
                return r2

            # Create study object and optimize the objective function
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=Trials)
            p=study.best_trial.params
            model=RandomForestRegressor(**p)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            Tuned_models["RandomForestRegressor"]=model
            report[model]=r2
            logging.info('RandomForestRegressor has been Tuned') 
            return report

        except Exception as e:
            raise CustomException(e,sys)    















