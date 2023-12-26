from dotenv import load_dotenv
load_dotenv()
import os
import sys
import time
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from heart.entity.artifact_entity import DataIngestionArtifact
from heart.entity.config_entity import DataIngestionConfig
from heart.exception import HeartException
from heart.logger import logging
from datetime import datetime


from heart.constant.training_pipeline import SCHEMA_FILE_PATH,SCHEMA_TARGET_COL,FILE_NAME
from heart.utils import read_yaml_file, write_yaml_file

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
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

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
        
    def read_data_from_feature_store(self) -> DataFrame:
        
        try:
            dataframe = pd.read_csv(self.data_ingestion_config.feature_store_file_path)
            # print(dataframe.columns())
            return dataframe
        except Exception as e:
            HeartException(e,sys)

    def split_data_as_train_test(self, dataframe: DataFrame):

        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            split = StratifiedShuffleSplit(n_splits=1, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)

            logging.info("Performed train test split on the dataframe")

            for train_index,test_index in split.split(dataframe, dataframe[self._schema_config[SCHEMA_TARGET_COL]]):
                strat_train_set = dataframe.loc[train_index].drop([self._schema_config[SCHEMA_TARGET_COL]],axis=1)
                strat_test_set = dataframe.loc[test_index].drop([self._schema_config[SCHEMA_TARGET_COL]],axis=1)

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            strat_train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            strat_test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")

        except Exception as e:
            raise HeartException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(f"Started downloading csv file")
            self.download_data()
            dataframe = self.read_data_from_feature_store()
            # dataframe = pd.read_csv(self.data_ingestion_config.feature_store_file)
            self.split_data_as_train_test(dataframe=dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise HeartException(e, sys)