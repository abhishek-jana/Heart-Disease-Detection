from heart.exception import HeartException
from heart.logger import logging
from heart.components.data_ingesion import DataIngestion
from heart.components.data_validation import DataValidation
from heart.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from heart.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig
import sys


class TrainingPipeline:

    def __init__(self,training_config: TrainingPipelineConfig):
        self.training_config = training_config
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_config)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.data_ingestion_config
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise HeartException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )
            return data_validation_artifact
        except Exception as e:
            raise HeartException(e, sys)


    def start(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            # data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            # model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            # model_eval_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)
        except Exception as e:
            raise HeartException(e, sys)