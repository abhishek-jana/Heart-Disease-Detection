from heart.exception import HeartException
from heart.logging import logger
from heart.config.pipeline.training import HeartConfig
from heart.component.training.data_ingesion import DataIngestion
from heart.entity.artifact_entity import DataIngestionArtifact
import sys


class TrainingPipeline:

    def __init__(self, heart_config: HeartConfig):
        self.heart_config: HeartConfig = heart_config

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.heart_config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact

        except Exception as e:
            raise HeartException(e, sys)


    def start(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
        except Exception as e:
            raise HeartException(e, sys)