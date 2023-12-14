from time import strftime
from heart.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from heart.constant.training_pipeline_config import *
from heart.constant import TIMESTAMP
from heart.logging import logger
from heart.exception import HeartException
import os, sys
import requests
import json
from datetime import datetime
from heart.entity.metadata_entity import DataIngestionMetadata



class HeartConfig:

    def __init__(self, pipeline_name=PIPELINE_NAME, timestamp=TIMESTAMP):
        """
        Configuration Manager

        """
        self.timestamp = timestamp
        self.pipeline_name = pipeline_name
        self.pipeline_config = self.get_pipeline_config()

    def get_pipeline_config(self) -> TrainingPipelineConfig:
        """
        This function will provide pipeline config information


        returns > PipelineConfig = namedtuple("PipelineConfig", ["pipeline_name", "artifact_dir"])
        """
        try:
            artifact_dir = PIPELINE_ARTIFACT_DIR
            pipeline_config = TrainingPipelineConfig(pipeline_name=self.pipeline_name,
                                                     artifact_dir=artifact_dir)

            logger.info(f"Pipeline configuration: {pipeline_config}")

            return pipeline_config
        except Exception as e:
            raise HeartException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:

        """
        master directory for data ingestion
        we will store metadata information and ingested file to avoid redundant download
        """
        data_ingestion_master_dir = os.path.join(self.pipeline_config.artifact_dir,
                                                 DATA_INGESTION_DIR)

        # time based directory for each run
        # data_ingesion_dir = data_ingesion_master_dir
        data_ingestion_dir = os.path.join(data_ingestion_master_dir,
                                          DATA_INGESTION_FEATURE_STORE_DIR)

        metadata_file_path = os.path.join(data_ingestion_master_dir, DATA_INGESTION_METADATA_FILE_NAME)

        data_ingestion_metadata = DataIngestionMetadata(metadata_file_path=metadata_file_path)

        if data_ingestion_metadata.is_metadata_file_present:
            metadata_info = data_ingestion_metadata.get_metadata_info()
            from_date = metadata_info.to_date

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir=data_ingestion_dir, # change it to data_ingestion_dir to get timestamp based dir
            download_dir=os.path.join(data_ingestion_dir,  DATA_INGESTION_DOWNLOADED_DATA_DIR),
            file_name=DATA_INGESTION_FILE_NAME,
            feature_store_dir=os.path.join(data_ingestion_master_dir, DATA_INGESTION_FEATURE_STORE_DIR),
            failed_dir=os.path.join(data_ingestion_dir, DATA_INGESTION_FAILED_DIR),
            metadata_file_path=metadata_file_path,
            datasource=DATA_INGESTION_DATA_SOURCE

        )
        logger.info(f"Data ingestion config: {data_ingestion_config}")
        return data_ingestion_config