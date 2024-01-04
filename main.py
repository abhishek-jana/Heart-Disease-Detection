import os,sys
import io
# from heart.exception import HeartException
# from heart.logger import logging
# from heart.pipeline import training_pipeline
from heart.pipeline.training_pipeline import TrainingPipeline
from heart.utils import read_yaml_file
from heart.constant.training_pipeline import SAVED_MODEL_DIR
from heart.constant.application import APP_HOST, APP_PORT
from heart.ml.model.estimator import ModelResolver
from heart.entity.config_entity import TrainingPipelineConfig
from heart.utils import load_object
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse
from fastapi import FastAPI, File, UploadFile,Request
import pandas as pd



app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_config= TrainingPipelineConfig()
        train_pipeline = TrainingPipeline(training_config=training_config)
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.start()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(request:Request,file: UploadFile = File(...)):
    try:
        # Get data from user CSV file
        content = await file.read()  # Read the content of the uploaded file

        #conver csv file to dataframe
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        model_resolver = ModelResolver()
        if not model_resolver.is_model_exists():
            return Response("Model is not available")
        
        latest_model_path = model_resolver.get_latest_model_path()
        latest_target_encoder_path = model_resolver.get_latest_target_encoder_path()
        model = load_object(file_path=latest_model_path)
        encoder = load_object(file_path=latest_target_encoder_path)
        y_pred = model.predict(df)
        df['predicted_column'] = encoder.inverse_transform(y_pred)
        # df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        # return df.to_html()
        # Extract 'predicted_column' from DataFrame
        predicted_values = df['predicted_column'].tolist()

        # Return 'predicted_column' as JSON response
        return JSONResponse(content=predicted_values, media_type="application/json")
        
    except Exception as e:
        raise Response(f"Error processing file: {str(e)}", status_code=500)

# def main():
#     try:
#         set_env_variable(env_file_path)
#         training_pipeline = TrainPipeline()
#         training_pipeline.run_pipeline()
#     except Exception as e:
#         print(e)
#         logging.exception(e)


if __name__=="__main__":
    #main()
    # set_env_variable(env_file_path)
    app_run(app, host=APP_HOST, port=APP_PORT)