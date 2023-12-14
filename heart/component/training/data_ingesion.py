from dotenv import load_dotenv
load_dotenv()
import os
import sys
import time
from collections import namedtuple
import pandas as pd

from heart.entity.artifact_entity import DataIngestionArtifact
from heart.entity.config_entity import DataIngestionConfig
from heart.entity.metadata_entity import DataIngestionMetadata
from heart.exception import HeartException
from heart.logging import logger
from datetime import datetime
# from heart.constant.environment.variable_key import KAGGLE_DATA_USERNAME
from kaggle.api.kaggle_api_extended import KaggleApi


class DataIngestion:
    # Used to download data in chunks.
    def __init__(self, data_ingestion_config: DataIngestionConfig ):
        """
        data_ingestion_config: Data Ingestion config
        """
        try:
            logger.info(f"{'>>' * 20}Starting data ingestion.{'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HeartException(e, sys)
    

    def download_data(self):
        try:
            download_dir = self.data_ingestion_config.download_dir#os.path.dirname(self.data_ingestion_config.download_dir)
            # creating download directory
            os.makedirs(download_dir, exist_ok=True)         

            logger.info(f"Started writing downloaded data into csv file: { download_dir}")
            # Set your Kaggle API key
            api = KaggleApi()
            api.authenticate()
            # username = os.getenv(KAGGLE_DATA_USERNAME) #download_data.username
            dataset_name = self.data_ingestion_config.datasource #DATA_INGESTION_DATA_SOURCE
            # downloading data
            api.dataset_download_files(f'{dataset_name}', path=download_dir, unzip=True)
            logger.info(f"Downloaded data has been written into file: {download_dir}")

        except Exception as e:
            logger.info(e)
            raise HeartException(e, sys)

    # def write_metadata(self,file_path: str) -> None:
    #     """
    #     This function help us to update metadata information 
    #     so that we can avoid redundant download and merging.

    #     """
    #     try:
    #         logger.info(f"Writing metadata info into metadata file.")
    #         metadata_info = DataIngestionMetadata(metadata_file_path=self.data_ingestion_config.metadata_file_path)

    #         metadata_info.write_metadata_info(date=datetime.now().strftime("%Y-%m-%d"),
    #                                           data_file_path=file_path
    #                                           )
    #         logger.info(f"Metadata has been written.")
    #     except Exception as e:
    #         raise HeartException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(f"Started downloading csv file")
            self.download_data()

            feature_store_file_path = os.path.join(self.data_ingestion_config.feature_store_dir,
                                                   self.data_ingestion_config.file_name)
            artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
                download_dir=self.data_ingestion_config.download_dir,
                metadata_file_path=self.data_ingestion_config.metadata_file_path,

            )

            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact
        except Exception as e:
            raise HeartException(e, sys)