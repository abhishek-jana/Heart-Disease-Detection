import os,sys
from dataclasses import dataclass
from datetime import datetime
from heart.exception import HeartException

from heart.constant.training_pipeline import *

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME

    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

    timestamp: str = TIMESTAMP

@dataclass
class DataIngestionConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config

            self.data_ingestion_dir: str = os.path.join(self.training_pipeline_config.artifact_dir,DATA_INGESTION_DIR)
            
            self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
            
            self.dataset_name: str = DATA_INGESTION_DATA_SOURCE
            
            self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)

            self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)

            self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

        except Exception as e:
            raise HeartException(e,sys)


@dataclass
class DataValidationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config    
            self.data_validation_dir: str = os.path.join(self.training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
            self.valid_data_dir: str = os.path.join(self.training_pipeline_config.artifact_dir,DATA_VALIDATION_VALID_DIR)
            self.invalid_data_dir: str = os.path.join(self.training_pipeline_config.artifact_dir,DATA_VALIDATION_INVALID_DIR)
            self.valid_train_file_path: str = os.path.join(self.valid_data_dir,TRAIN_FILE_NAME)
            self.valid_test_file_path: str = os.path.join(self.valid_data_dir,TEST_FILE_NAME)

            self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir,TRAIN_FILE_NAME)
            self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir,TEST_FILE_NAME)

            self.drift_report_file_path: str = os.path.join(self.data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        
        except Exception as e:
            HeartException(e,sys)

