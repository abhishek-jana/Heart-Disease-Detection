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
from heart.logger import logging
from datetime import datetime
# from heart.constant.environment.variable_key import KAGGLE_DATA_USERNAME
from kaggle.api.kaggle_api_extended import KaggleApi


class DataIngestion:
    # Used to download data in chunks.
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        data_ingestion_config: Data Ingestion config
        """
        try:
            logging.info(f"{'>>' * 20}Starting data ingestion.{'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HeartException(e, sys)
    

    def download_data(self):
        try:
            download_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            # creating download directory
            os.makedirs(download_dir, exist_ok=True)         

            logging.info(f"Started writing downloaded data into csv file: { download_dir}")
            # Set your Kaggle API key
            api = KaggleApi()
            api.authenticate()
            dataset_name = self.data_ingestion_config.dataset_name
            # downloading data
            api.dataset_download_files(f'{dataset_name}', path=download_dir, unzip=True)
            logging.info(f"Downloaded data has been written into file: {download_dir}")

        except Exception as e:
            logging.info(e)
            raise HeartException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(f"Started downloading csv file")
            self.download_data()

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path
            )

            logging.info(f"Data ingestion artifact: {artifact}")
            return artifact
        except Exception as e:
            raise HeartException(e, sys)