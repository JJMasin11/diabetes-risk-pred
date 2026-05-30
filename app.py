from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            systolic_bp=int(request.form.get('systolic_bp')),
            diastolic_bp=int(request.form.get('diastolic_bp')),
            high_chol=request.form.get('high_chol'),
            chol_check=int(request.form.get('chol_check')),
            height=int(request.form.get('height')),
            weight=int(request.form.get('weight')),
            smoker=int(request.form.get('smoker')),
            stroke=int(request.form.get('stroke')),
            heart_disease_or_attack=int(request.form.get('heart_disease_or_attack')),
            physical_activity=int(request.form.get('physical_activity')),
            fruits=int(request.form.get('fruits')),
            veggies=int(request.form.get('veggies')),
            heavy_alc_consumption=int(request.form.get('heavy_alc_consumption')),
            healthcare=int(request.form.get('healthcare')),
            doctor_no_care=int(request.form.get('doctor_no_care')),
            general_health=int(request.form.get('general_health')),
            mental_health=int(request.form.get('mental_health')),
            physical_health=int(request.form.get('physical_health')),
            diff_walk=int(request.form.get('diff_walk')),
            sex=int(request.form.get('sex')),
            age=int(request.form.get('age')),
            education=int(request.form.get('education')),
            income=int(request.form.get('income'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        if results[0][0] == 0:
            display="Our model predicts that you are at risk of having diabetes. Please consult a doctor for an official diagnosis."
        else:
            display="Our model predicts that you are not at risk of having diabetes."

        return render_template('home.html', results=display)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)