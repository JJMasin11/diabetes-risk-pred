import sys
import pandas as pd
import shap
from src.exception import CustomException
from src.logger import logging
from src.utils import load_from_blob, get_vault_secret
from src.api.schemas import PredictRequest, PredictResponse
from sqlalchemy import create_engine

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, request: PredictRequest):
        try:
            # Preprocess
            high_bp, high_chol, high_chol_prob, chol_imputed, bmi, age_cat, income_cat, bmi_age = self.preprocess(request)

            # Make dataframe
            input_dict = {
                "high_bp": [high_bp],
                "high_chol": [high_chol],
                "chol_check": [request.chol_check],
                "bmi": [bmi],
                "smoker": [request.smoker],
                "stroke": [request.stroke],
                "heart_disease_or_attack": [request.heart_disease_or_attack],
                "phys_activity": [request.physical_activity],
                "fruits": [request.fruits],
                "veggies": [request.veggies],
                "hvy_alcohol_consump": [request.heavy_alc_consumption],
                "any_healthcare": [request.healthcare],
                "no_doc_bc_cost": [request.doctor_no_care],
                "gen_hlth": [request.general_health],
                "ment_hlth": [request.mental_health],
                "phys_hlth": [request.physical_health],
                "diff_walk": [request.diff_walk],
                "sex": [request.sex],
                "age": [age_cat],
                "education": [request.education],
                "income": [income_cat],
                "bmi_age": [bmi_age]
            }

            input_data = pd.DataFrame(input_dict)
            logging.info("Input data:")
            logging.info(input_data)

            # Load model
            model = load_from_blob('main_model.pkl')
            preprocessor = load_from_blob('main_preprocessor.pkl')
            threshold = load_from_blob('threshold.pkl')

            data_scaled = preprocessor.transform(input_data)

            # Make prediction
            proba = model.predict_proba(data_scaled)
            pred = (proba >= threshold).astype(int)

            # Translate response
            if pred[0][0] == 0:
                diabetes_risk = 1
            else:
                diabetes_risk = 0

            diabetes_probability = float(proba[0][1])

            # Calculate SHAP probabilities
            top_shap = self.get_top_shap_values(model, data_scaled, input_data)

            # Create dataframe to log to database
            log_data = {
                "systolic_bp": [request.systolic_bp],
                "diastolic_bp": [request.diastolic_bp],
                "high_chol_input": [request.high_chol],
                "chol_check": [request.chol_check],
                "height": [request.height],
                "weight": [request.weight],
                "smoker": [request.smoker],
                "stroke": [request.stroke],
                "heart_disease_or_attack": [request.heart_disease_or_attack],
                "phys_activity": [request.physical_activity],
                "fruits": [request.fruits],
                "veggies": [request.veggies],
                "hvy_alcohol_consump": [request.heavy_alc_consumption],
                "any_healthcare": [request.healthcare],
                "no_doc_bc_cost": [request.doctor_no_care],
                "gen_hlth": [request.general_health],
                "ment_hlth": [request.mental_health],
                "phys_hlth": [request.physical_health],
                "diff_walk": [request.diff_walk],
                "sex": [request.sex],
                "age": [request.age],
                "education": [request.education],
                "income": [request.income],
                "high_bp": [high_bp],
                "bmi": [bmi],
                "age_category": [age_cat],
                "income_category": [income_cat],
                "bmi_age": [bmi_age],
                "high_chol": [high_chol],
                "chol_imputed": [chol_imputed],
                "chol_imputed_probability": [high_chol_prob],
                "diabetes_prediction": [diabetes_risk],
                "diabetes_probability": [diabetes_probability]
            }

            log_df = pd.DataFrame(log_data)
            self.log_to_database(log_df)

            return PredictResponse(diabetes_risk=diabetes_risk, diabetes_probability=diabetes_probability, shap_values=top_shap)
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def preprocess(self, request: PredictRequest):
        # Check if user has high blood pressure
        high_bp = self.get_high_bp(request)

        # Calculate BMI
        bmi = (request.weight * 703) / (request.height ** 2)

        # Categorize age
        age_cat = self.get_age_category(request)

        # Categorize income
        income_cat = self.get_income_category(request)

        # Get BMI Age interaction
        bmi_age = bmi * request.age

        # Impute unsure cholesterol data
        high_chol_prob = None
        chol_imputed = False

        if request.high_chol == 'Yes':
            high_chol = 1
        elif request.high_chol == 'No':
            high_chol = 0
        else:
            chol_imputed = True
            high_chol, high_chol_prob = self.impute_chol(request, high_bp, bmi, age_cat, income_cat, bmi_age)

        return high_bp, high_chol, high_chol_prob, chol_imputed, bmi, age_cat, income_cat, bmi_age
    
    def get_high_bp(self, request: PredictRequest):
        if request.systolic_bp >= 130 or request.diastolic_bp > 80:
            return 1
        return 0
    
    def get_age_category(self, request: PredictRequest):
        if request.age <= 24:
            return 1
        elif request.age <= 29:
            return 2
        elif request.age <= 34:
            return 3
        elif request.age <= 39:
            return 4
        elif request.age <= 44:
            return 5
        elif request.age <= 49:
            return 6
        elif request.age <= 54:
            return 7
        elif request.age <= 59:
            return 8
        elif request.age <= 64:
            return 9
        elif request.age <= 69:
            return 10
        elif request.age <= 74:
            return 11
        elif request.age <= 79:
            return 12
        else:
            return 13
        
    def get_income_category(self, request: PredictRequest):
        if request.income < 10000:
            return 1
        elif request.income < 15000:
            return 2
        elif request.income < 20000:
            return 3
        elif request.income < 25000:
            return 4
        elif request.income < 35000:
            return 5
        elif request.income < 50000:
            return 6
        elif request.income < 75000:
            return 7
        else:
            return 8
        
    def impute_chol(self, request: PredictRequest, bp, bmi, age, income, bmi_age):
        try:
            model = load_from_blob('chol_model.pkl')
            preprocessor = load_from_blob('chol_preprocessor.pkl')

        
            data_dict = {
                    "high_bp": [bp],
                    "chol_check": [request.chol_check],
                    "bmi": [bmi],
                    "smoker": [request.smoker],
                    "stroke": [request.stroke],
                    "heart_disease_or_attack": [request.heart_disease_or_attack],
                    "phys_activity": [request.physical_activity],
                    "fruits": [request.fruits],
                    "veggies": [request.veggies],
                    "hvy_alcohol_consump": [request.heavy_alc_consumption],
                    "any_healthcare": [request.healthcare],
                    "no_doc_bc_cost": [request.doctor_no_care],
                    "gen_hlth": [request.general_health],
                    "ment_hlth": [request.mental_health],
                    "phys_hlth": [request.physical_health],
                    "diff_walk": [request.diff_walk],
                    "sex": [request.sex],
                    "age": [age],
                    "education": [request.education],
                    "income": [income],
                    "bmi_age": [bmi_age]
                }
            
            data = pd.DataFrame(data_dict)

            data_scaled = preprocessor.transform(data)

            pred = model.predict(data_scaled)
            parsed_pred = int(pred[0])

            # Get high cholesterol probability
            proba = model.predict_proba(data_scaled)
            chol_proba = float(proba[0][1])

            return parsed_pred, chol_proba
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_top_shap_values(self, model, data, input_data):
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(data)
        shap_vals = dict(zip(input_data.columns, raw_shap[0].tolist()))
        sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict([(k, v) for k, v in sorted_shap if abs(v) > 0.01][:5])
    
    def log_to_database(self, df):
        # Connect to the PostgreSQL database
        engine = create_engine(f"postgresql+psycopg2://{get_vault_secret('DB-USER')}:{get_vault_secret('DB-PASSWORD')}@{get_vault_secret('DB-HOST')}:{get_vault_secret('DB-PORT')}/{get_vault_secret('DB-NAME')}")

        # Append prediction data
        try:
            df.to_sql('predictions', engine, if_exists='append', index=False)

        except Exception as e:
            raise CustomException(e, sys)
