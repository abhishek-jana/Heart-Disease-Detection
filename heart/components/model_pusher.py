from heart.exception import HeartException
from heart.logger import logging
from heart.entity.artifact_entity import (DataTransformationArtifact,ModelPusherArtifact,
                                          ModelEvaluationArtifact)
from heart.entity.config_entity import ModelPusherConfig
from heart.ml.model.estimator import ModelResolver
import os,sys
from heart.ml.metric.classification_metric import get_classification_score
from heart.utils import save_object,load_object,write_yaml_file

import shutil

class ModelPusher:

    def __init__(self,
                model_pusher_config:ModelPusherConfig,
                data_transformation_artifact:DataTransformationArtifact,
                model_eval_artifact:ModelEvaluationArtifact):

        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise HeartException(e, sys)
    

    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path
            
            #Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            transformer_path = self.model_pusher_config.pusher_transformer_path
            target_encoder_path = self.model_pusher_config.pusher_target_encoder_path

            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

            os.makedirs(os.path.dirname(transformer_path),exist_ok=True)
            shutil.copy(src=self.data_transformation_artifact.transformed_object_file_path, dst=transformer_path)

            os.makedirs(os.path.dirname(target_encoder_path),exist_ok=True)
            shutil.copy(src=self.data_transformation_artifact.target_encoder_path, dst=target_encoder_path)

            

            #saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            saved_transformer_path = self.model_pusher_config.saved_transformer_path
            saved_encoder_path = self.model_pusher_config.saved_encoder_path

            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)

            os.makedirs(os.path.dirname(saved_transformer_path),exist_ok=True)
            shutil.copy(src=self.data_transformation_artifact.transformed_object_file_path, dst=saved_transformer_path)

            os.makedirs(os.path.dirname(saved_encoder_path),exist_ok=True)
            shutil.copy(src=self.data_transformation_artifact.target_encoder_path, dst=saved_encoder_path)

            #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            return model_pusher_artifact
        except  Exception as e:
            raise HeartException(e, sys)
    