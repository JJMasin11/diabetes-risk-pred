import os
import requests
from flask import Flask, request, render_template
from dotenv import load_dotenv

application = Flask(__name__)
app = application

SHAP_FEATURE_LABELS = {
    "high_bp":                 "High Blood Pressure",
    "high_chol":               "High Cholesterol",
    "chol_check":              "Cholesterol Check",
    "bmi":                     "BMI",
    "smoker":                  "Smoker",
    "stroke":                  "Stroke",
    "heart_disease_or_attack": "Heart Disease or Heart Attack",
    "phys_activity":           "Physical Activity",
    "fruits":                  "Fruit Consumption",
    "veggies":                 "Vegetable Consumption",
    "hvy_alcohol_consump":     "Heavy Alcohol Consumption",
    "any_healthcare":          "Has Healthcare Coverage",
    "no_doc_bc_cost":          "Could Not See Doctor Due to Cost",
    "gen_hlth":                "General Health",
    "ment_hlth":               "Mental Health Days (Past 30)",
    "phys_hlth":               "Physical Health Days (Past 30)",
    "diff_walk":               "Difficulty Walking or Climbing Stairs",
    "sex":                     "Sex",
    "age":                     "Age",
    "education":               "Education Level",
    "income":                  "Income",
    "bmi_age":                 "BMI × Age Interaction",
}

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/diabetes-risk-prediction', methods=['POST'])
def predict_datapoint():
    payload = {
        "systolic_bp": int(request.form.get('systolic_bp')),
        "diastolic_bp": int(request.form.get('diastolic_bp')),
        "high_chol": request.form.get('high_chol'),
        "chol_check": int(request.form.get('chol_check')),
        "height": int(request.form.get('height')),
        "weight": int(request.form.get('weight')),
        "smoker": int(request.form.get('smoker')),
        "stroke": int(request.form.get('stroke')),
        "heart_disease_or_attack": int(request.form.get('heart_disease_or_attack')),
        "physical_activity": int(request.form.get('physical_activity')),
        "fruits": int(request.form.get('fruits')),
        "veggies": int(request.form.get('veggies')),
        "heavy_alc_consumption": int(request.form.get('heavy_alc_consumption')),
        "healthcare": int(request.form.get('healthcare')),
        "doctor_no_care": int(request.form.get('doctor_no_care')),
        "general_health": int(request.form.get('general_health')),
        "mental_health": int(request.form.get('mental_health')),
        "physical_health": int(request.form.get('physical_health')),
        "diff_walk": int(request.form.get('diff_walk')),
        "sex": int(request.form.get('sex')),
        "age": int(request.form.get('age')),
        "education": int(request.form.get('education')),
        "income": int(request.form.get('income'))
    }
        
    load_dotenv()
    FASTAPI_URL = os.getenv("FASTAPI_URL")

    response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
    result = response.json()

    shap_display = [
        (SHAP_FEATURE_LABELS.get(k, k), v)
        for k, v in result.get('shap_values', {}).items()
    ]

    if result['diabetes_risk'] == 1:
        display="Our model predicts that you are at risk of having diabetes. Please consult a doctor for an official diagnosis."
    else:
        display="Our model predicts that you are not at risk of having diabetes."

    return render_template(
        'home.html',
        results=display,
        diabetes_probability=result['diabetes_probability'],
        shap_values=shap_display,
    )
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)