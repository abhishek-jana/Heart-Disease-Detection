from heart.exception import HeartException
from heart.logger import logging
from heart.entity.artifact_entity import (DataValidationArtifact,DataTransformationArtifact,
                                        ModelTrainerArtifact,ModelEvaluationArtifact)
from heart.entity.config_entity import ModelEvaluationConfig
import os,sys
from heart.ml.metric.classification_metric import get_classification_score
from heart.ml.model.estimator import HeartModel
from heart.utils import save_object,load_object,write_yaml_file
from heart.ml.model.estimator import ModelResolver
from heart.constant.training_pipeline import TARGET_COLUMN
import pandas  as  pd
class ModelEvaluation:


    def __init__(self,model_evaluation_config:ModelEvaluationConfig,
                data_validation_artifact:DataValidationArtifact,
                data_transformation_artifact: DataTransformationArtifact,                
                model_trainer_artifact:ModelTrainerArtifact):
        
        try:
            self.model_eval_config=model_evaluation_config
            self.data_transformation_artifact=data_transformation_artifact
            self.data_validation_artifact=data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise HeartException(e,sys)
    


    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            #if saved model folder has model the we will compare 
            #which model is best trained or the model from saved model folder

            logging.info("if saved model folder has model the we will compare "
            "which model is best trained or the model from saved model folder")
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            if not self.model_resolver.is_model_exists():
                logging.info("No model is available in saved model folder")
                is_model_accepted=True
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            #Finding location of transformer and target encoder
            # latest_transformer_path = self.model_resolver.get_latest_transformer_path()
            # latest_dir_path = self.model_resolver.get_latest_dir_path()
            logging.info(f"Finding location of transformer and target encoder")
            latest_model_path = self.model_resolver.get_latest_model_path()
            latest_target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info(f"Previous trained objects of transformer, model and target encoder")

            #previous trained objects of transformer, model and target encoder
            # latest_transformer = load_object(file_path=latest_transformer_path)
            latest_model = load_object(file_path=latest_model_path)
            latest_target_encoder = load_object(file_path=latest_target_encoder_path)

            logging.info(f"Current trained objects of transformer, model and target encoder")
            # trained_transformer = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path) 
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path) 
            trained_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            #valid train and test file dataframe
            test_df = pd.read_csv(valid_test_file_path)

            target_df = test_df[TARGET_COLUMN]
            y_true = latest_target_encoder.transform(target_df)

            # input_feature_name = list(latest_transformer.feature_names_in_)
            # df = latest_transformer.transform(test_df[input_feature_name])
            y_pred = latest_model.predict(test_df)
            logging.info(f"Prediction using latest model: {latest_target_encoder.inverse_transform(y_pred[:5])}")
            latest_metric = get_classification_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using latest trained model: {latest_metric}")
           
            # accuracy using trained trained model
            # input_feature_name = list(trained_transformer.feature_names_in_)
            # df = trained_transformer.transform(test_df[input_feature_name])
            y_pred = trained_model.predict(test_df)
            y_true = trained_target_encoder.transform(target_df)
            logging.info(f"Prediction using trained model: {trained_target_encoder.inverse_transform(y_pred[:5])}") 
            trained_metric = get_classification_score(y_true=y_true, y_pred=y_pred)
        

            improved_accuracy = trained_metric.f1_score-latest_metric.f1_score
            if self.model_eval_config.change_threshold < improved_accuracy:
                #0.02 < 0.03
                is_model_accepted=True
            else:
                is_model_accepted=False

            
            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)
            
            model_eval_report = model_evaluation_artifact.__dict__

            #save the report
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise HeartException(e,sys)

    
    