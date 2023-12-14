from heart.pipeline.training import TrainingPipeline
from heart.config.pipeline.training import HeartConfig

if __name__=="__main__":
    training_pipeline_config= HeartConfig()
    training_pipeline = TrainingPipeline(heart_config=training_pipeline_config)
    training_pipeline.start()
