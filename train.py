from heart.pipeline.training_pipeline import TrainingPipeline
from heart.entity.config_entity import TrainingPipelineConfig

if __name__=="__main__":
    training_config= TrainingPipelineConfig()
    training_pipeline = TrainingPipeline(training_config=training_config)
    training_pipeline.start()
