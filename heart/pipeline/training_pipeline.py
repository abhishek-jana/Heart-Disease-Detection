from heart.exception import HeartException
from heart.logger import logging
from heart.components.data_ingesion import DataIngestion
from heart.entity.artifact_entity import DataIngestionArtifact
from heart.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
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


    def start(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
        except Exception as e:
            raise HeartException(e, sys)