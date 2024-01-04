from heart.exception import HeartException
from heart.logger import logging
from heart.components.data_ingesion import DataIngestion
from heart.components.data_validation import DataValidation
from heart.components.data_transformation import DataTransformation
from heart.components.model_trainer import ModelTrainer
from heart.components.model_evaluation import ModelEvaluation
from heart.components.model_pusher import ModelPusher
from heart.entity.artifact_entity import (DataIngestionArtifact,
                                          DataValidationArtifact,
                                          DataTransformationArtifact,
                                          ModelTrainerArtifact,
                                          ModelEvaluationArtifact,
                                          ModelPusherArtifact)
from heart.entity.config_entity import (TrainingPipelineConfig,
                                        DataIngestionConfig,
                                        DataValidationConfig,
                                        DataTransformationConfig,
                                        ModelTrainerConfig,
                                        ModelEvaluationConfig,
                                        ModelPusherConfig)
import sys


class TrainingPipeline:
    is_pipeline_running=False
    def __init__(self,training_config: TrainingPipelineConfig):
        self.training_config = training_config
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_config)
        self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_config)
        self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_config)
        self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_config)

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

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = self.data_validation_config
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )
            return data_validation_artifact
        except Exception as e:
            raise HeartException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = self.data_transformation_config
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                    data_validation_artifact=data_validation_artifact) 
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            HeartException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = self.model_trainer_config
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, 
                                         data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise HeartException(e, sys)
        
    def start_model_evaluation(self,data_validation_artifact: DataValidationArtifact,
                            data_transformation_artifact: DataTransformationArtifact,
                            model_trainer_artifact: ModelTrainerArtifact):
        try:
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=self.training_config)
            model_eval = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                         data_validation_artifact=data_validation_artifact,
                                         data_transformation_artifact = data_transformation_artifact,
                                         model_trainer_artifact=model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except Exception as e:
            raise HeartException(e, sys)
        
    def start_model_pusher(self,data_transformation_artifact: DataTransformationArtifact,model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_config)
            model_pusher = ModelPusher(model_pusher_config = model_pusher_config, 
                                       data_transformation_artifact = data_transformation_artifact,
                                       model_eval_artifact = model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except  Exception as e:
            raise  HeartException(e,sys)


    def start(self):
        try:
            TrainingPipeline.is_pipeline_running=True
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact = data_validation_artifact,
                                                            data_transformation_artifact = data_transformation_artifact,
                                                            model_trainer_artifact = model_trainer_artifact)
            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            model_pusher_artifact = self.start_model_pusher(data_transformation_artifact = data_transformation_artifact,
                                                            model_eval_artifact = model_eval_artifact)
            TrainingPipeline.is_pipeline_running=False
        except Exception as e:
            raise HeartException(e, sys)