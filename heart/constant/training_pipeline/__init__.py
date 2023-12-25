import os
from heart.constant.training_pipeline.data_ingesion import *

TARGET_COLUMN = ""

PIPELINE_NAME = "heart"
ARTIFACT_DIR = "artifact"

# Common filename

FILE_NAME: str = "heart.csv"

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

MODEL_FILE_NAME = "model.pkl"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

SCHEMA_DROP_COLS = "drop_columns"