from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.prediction_pipeline import customData, predictPipeline

application= Flask(__name__)
app= application

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods= ["GET","POST"])
def predict_datapoint():

    if request.method== "GET" :
        return render_template('home.html')
    
    else:
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = float(request.form.get("reading_score"))
        writing_score = float(request.form.get("writing_score"))

        custom_data_obj = customData(gender=gender,race_ethnicity=race_ethnicity,parental_level_of_education= parental_level_of_education, lunch=lunch, test_preparation_course=test_preparation_course,reading_score=reading_score, writing_score=writing_score)

        input_df = custom_data_obj.get_data_as_data_frame()
        
        predict_pipeline = predictPipeline()
        predictions = predict_pipeline.predict(input_df)
        
        return render_template("home.html", prediction=round(predictions, 2))
    
if __name__ == "__main__":
    app.run(host='0.0.0.0')