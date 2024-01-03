import os,sys
import pandas as pd
import numpy as np

from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler,OneHotEncoder,LabelEncoder 
from sklearn.compose import ColumnTransformer
from heart.exception import HeartException
from heart.logger import logging
from heart.constant.training_pipeline import TARGET_COLUMN,SCHEMA_FILE_PATH
from heart.entity.artifact_entity import (DataValidationArtifact,
                                          DataTransformationArtifact)
from heart.entity.config_entity import DataTransformationConfig

from heart.ml.model.estimator import TargetValueMapping
from heart.utils import save_numpy_array_data, save_object
from heart.utils import read_yaml_file


class DataTransformation:
    def __init__(self,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise HeartException(e,sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HeartException(e,sys)
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_cols = self._schema_config["numerical_columns"]
            categorical_cols = self._schema_config["categorical_columns"]
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop = 'if_binary'))
            ])
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            return preprocessor
        except Exception as e:
            raise HeartException(e,sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating data transformation")
            # print (len(self._schema_config["numerical_columns"]))
            # logging.info("Data transformation object created")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)  
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
           
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # print (input_feature_train_df.shape, target_feature_train_df.shape)
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            # print(target_feature_train_df_res)
            logging.info("Got train features and test features of Training dataset")
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            #transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info("Got train features and test features of Testing dataset")

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )
            # print (input_feature_train_df.shape, target_feature_train_arr.shape)
            preprocessor_obj = self.get_data_transformer_object()
            # print (preprocessor_obj)
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            logging.info(
                "Used the preprocessor object to fit transform the train features"
            )
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transform the test features")
            # print (input_feature_train_arr.shape, target_feature_train_df.shape)
            logging.info("Applying SMOTETomek on Training dataset")
            smt = SMOTETomek(sampling_strategy="minority",random_state=42)
            # print (input_feature_train_arr.shape, target_feature_train_df.shape)
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)   
            # print (input_feature_train_arr.shape)
            logging.info("Applied SMOTETomek on testing dataset")

            logging.info("Created train array and test array")
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            logging.info("Data transformation completed")

            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_obj
            )

            save_object(
                file_path=self.data_transformation_config.target_encoder_path,
                obj=preprocessor_obj
            )

            logging.info("Saving train array and test array")
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info("Saved train array and test array")
            logging.info("Creating DataTransformationArtifact")

            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            HeartException(e,sys)