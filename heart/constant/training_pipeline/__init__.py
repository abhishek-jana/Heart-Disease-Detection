import os
from heart.constant.training_pipeline.data_ingesion import *
from heart.constant.training_pipeline.data_validation import *
from heart.constant.training_pipeline.data_transformation import *
from heart.constant.training_pipeline.model_evaluation import *
from heart.constant.training_pipeline.model_trainer import *
from heart.constant.training_pipeline.model_pusher import *

PIPELINE_NAME = "heart"
ARTIFACT_DIR = "artifact"

# SAVED_MODEL_DIR = "saved_models"
# TRANSFORMER_DIR_NAME = "transformer"
# TARGET_ENCODER_DIR_NAME = "target_encoder"
# MODEL_DIR_NAME = "model"

# Common filename

FILE_NAME: str = "Heart Attack.csv"

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"

MODEL_FILE_NAME = "model.pkl"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

SCHEMA_DROP_COLS = "drop_columns"

TARGET_COLUMN = "class"