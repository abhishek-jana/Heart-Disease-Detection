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

@dataclass
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
            self.data_transformation_dir: str = os.path.join(self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
            self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace("csv","npy"))
            self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME.replace("csv","npy"))
            self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                                  PREPROCSSING_OBJECT_FILE_NAME)
            self.target_encoder_path: str = os.path.join(self.data_transformation_dir, TARGET_ENCODER_DIR,
                                                                  TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise HeartException(e, sys)

    
@dataclass
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, 
            MODEL_FILE_NAME
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        
@dataclass
class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME
            )
        self.report_file_path = os.path.join(self.model_evaluation_dir, MODEL_EVALUATION_REPORT_NAME)
        self.change_threshold =  MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

@dataclass
class ModelPusherConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path = os.path.join(self.model_evaluation_dir, MODEL_FILE_NAME)
        timestamp = round(datetime.now().timestamp())
        self.saved_model_path=os.path.join(
            SAVED_MODEL_DIR,
            f"{timestamp}",
            MODEL_FILE_NAME)