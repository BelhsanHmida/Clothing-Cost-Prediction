import sys
import os 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from src.utils  import save_object
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformer_object(self):
        try:
            numerical_features=[]
            high_cat_features=[]
            low_cat_features=[]
            cat_features=low_cat_features+high_cat_features

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
                )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info("Numerical Columns Standard scaling completed")
            logging.info("Categoricla Columns encoding completed")

            preprocessor=ColumnTransformer(
            [('num_pipeline',num_pipeline,numerical_features),
             ('cat_pipeline',cat_pipeline,cat_features)]
            )
            return preprocessor    

        except Exception as e:
            raise CustomException(e,sys)

    def iniate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Train and test data Read completed')
            logging.info('obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name=''
            numerical_features=[]
            high_cat_features=[]
            low_cat_features=[]
            cat_features=low_cat_features+high_cat_features

            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            
            target_feature_train_df=train_df[target_column_name]
            target_feature_test_df=test_df[target_column_name]

            logging.info(f'applying preprocessing object on training dataframe and tesing dataframe.')

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.fit_transform(input_features_test_df)

            train_arr=np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_features_test_arr,np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object. ")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e :
            raise CustomException(e,sys)