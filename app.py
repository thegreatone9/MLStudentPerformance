from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

application = Flask(__name__)
app = application
template = 'index.html'

try:
    from src.utils import ARTIFACTS_DIR
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found! Initiating Data Ingestion and Training...")
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
        print("Dynamic Model Training Complete!")
        
except Exception as e:
    print(f"Error during dynamic training setup: {e}")

@app.route('/', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template(template)
    
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('writing_score')),
            writing_score = float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template(template, results = results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")       