import os,sys
from heart.constant.training_pipeline import SCHEMA_FILE_PATH

from heart.entity.config_entity import DataValidationConfig
from heart.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact

from heart.logger import logging
from heart.exception import HeartException

from heart.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise HeartException(e,sys)
        
    def validate_number_of_columns(self,dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Number of columns in schema: {number_of_columns}")
            logging.info(f"Number of columns in dataframe: {dataframe.shape[1]}")  
            if number_of_columns == dataframe.shape[1]:
                return True
            return False 
        except Exception as e:
            HeartException(e,sys)

    def is_numerical_column_present(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns.tolist()

            numerical_columns_present = True
            missing_numerical_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    numerical_columns_present = False
                    missing_numerical_columns.append(column)
            logging.info(f"Missing_numerical_columns: {missing_numerical_columns}")
        except Exception as e:
            raise HeartException(e,sys)

    @staticmethod    
    def read_data(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HeartException(e,sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            drift_detected = True
            report = {}
            for column in base_df.columns:
                statistic, pvalue = ks_2samp(base_df[column], current_df[column])
                if threshold <= pvalue:
                    drift_detected = False
                report[column] = {"pvalue": float(pvalue), 'drift_detected': drift_detected}

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #create directory if not exists
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path = drift_report_file_path, content = report)
            return drift_detected
            
        except Exception as e:
            raise HeartException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ""
            logging.info("Entered initiate_data_validation method of DataValidation class")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            #reading train and test data
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            #validating number of columns
            is_number_of_columns_valid = self.validate_number_of_columns(dataframe=train_df)
            if not is_number_of_columns_valid:
                error_message += "Number of columns in schema and train data are not same.\n"
            is_number_of_columns_valid = self.validate_number_of_columns(dataframe=test_df)
            if not is_number_of_columns_valid:
                error_message += "Number of columns in schema and test data are not same.\n"

            #validating numerical columns
            is_numerical_column_present = self.is_numerical_column_present(dataframe=train_df)
            if not is_numerical_column_present:
                error_message += "Numerical columns in schema and train data are not same.\n"
            is_numerical_column_present = self.is_numerical_column_present(dataframe=test_df)
            if not is_numerical_column_present:
                error_message += "Numerical columns in schema and test data are not same.\n"

            #detecting drift
            is_drift_detected = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            #creating artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status = is_drift_detected,
                valid_train_file_path = self.data_validation_config.valid_train_file_path,
                valid_test_file_path = self.data_validation_config.valid_test_file_path,
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HeartException(e,sys)