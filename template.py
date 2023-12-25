# Create folders and files autometically

import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "heart"  # name specific name to your project

list_of_files = [
    # since it's an empty folder we need a file to commit to github
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/training/__init__.py",
    f"{project_name}/components/training/data_ingesion.py",
    f"{project_name}/components/training/data_validation.py",
    f"{project_name}/components/training/data_transformation.py",
    f"{project_name}/components/training/model_evaluation.py",
    f"{project_name}/components/training/model_pusher.py",
    f"{project_name}/components/training/model_trainer.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/mongo_db_connection.py",
    f"{project_name}/config/pipeline/training.py",
    f"{project_name}/config/pipeline/__init__.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/constant/application.py",
    f"{project_name}/constant/database.py",
    f"{project_name}/constant/env_variable.py",
    f"{project_name}/constant/s3_bucket.py",
    f"{project_name}/constant/prediction_pipeline/__init__.py",
    f"{project_name}/constant/training_pipeline/__init__.py",
    f"{project_name}/constant/training_pipeline/data_ingesion.py",
    f"{project_name}/constant/training_pipeline/data_validation.py",
    f"{project_name}/constant/training_pipeline/data_transformation.py",
    f"{project_name}/constant/training_pipeline/model_evaluation.py",
    f"{project_name}/constant/training_pipeline/model_pusher.py",
    f"{project_name}/constant/training_pipeline/model_trainer.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/ml/__init__.py",
    "config/model.yaml"
    "config/schema.yaml"
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"

]

for filepath in list_of_files:
    filepath = Path(filepath)  # to avoid confusion between linux and windows
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already present.")