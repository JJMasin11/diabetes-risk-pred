import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_from_blob

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model = load_from_blob('main_model.pkl')
            preprocessor = load_from_blob('main_preprocessor.pkl')
            threshold = load_from_blob('threshold.pkl')

            data_scaled = preprocessor.transform(features)

            proba = model.predict_proba(data_scaled)
            pred = (proba >= threshold).astype(int)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 systolic_bp: int,
                 diastolic_bp: int,
                 high_chol: str,
                 chol_check: int,
                 height: int,
                 weight: int,
                 smoker: int,
                 stroke: int,
                 heart_disease_or_attack: int,
                 physical_activity: int,
                 fruits: int,
                 veggies: int,
                 heavy_alc_consumption: int,
                 healthcare: int,
                 doctor_no_care: int,
                 general_health: int,
                 mental_health: int,
                 physical_health: int,
                 diff_walk: int,
                 sex: int,
                 age: int,
                 education: int,
                 income: int):
        self.systolic_bp = systolic_bp
        self.diastolic_bp = diastolic_bp
        self.high_chol_str = high_chol
        self.chol_check = chol_check
        self.height = height
        self.weight = weight
        self.smoker = smoker
        self.stroke = stroke
        self.heart_disease_or_attack = heart_disease_or_attack
        self.physical_activity = physical_activity
        self.fruits = fruits
        self.veggies = veggies
        self.heavy_alc_consumption = heavy_alc_consumption
        self.healthcare = healthcare
        self.doctor_no_care = doctor_no_care
        self.general_health = general_health
        self.mental_health = mental_health
        self.physical_health = physical_health
        self.diff_walk = diff_walk
        self.sex = sex
        self.age = age
        self.education = education
        self.income = income

    def impute_chol(self, bp, bmi, age, income, bmi_age):
        try:
            model = load_from_blob('chol_model.pkl')
            preprocessor = load_from_blob('chol_preprocessor.pkl')

        
            data_dict = {
                    "high_bp": [bp],
                    "chol_check": [self.chol_check],
                    "bmi": [bmi],
                    "smoker": [self.smoker],
                    "stroke": [self.stroke],
                    "heart_disease_or_attack": [self.heart_disease_or_attack],
                    "phys_activity": [self.physical_activity],
                    "fruits": [self.fruits],
                    "veggies": [self.veggies],
                    "hvy_alcohol_consump": [self.heavy_alc_consumption],
                    "any_healthcare": [self.healthcare],
                    "no_doc_bc_cost": [self.doctor_no_care],
                    "gen_hlth": [self.general_health],
                    "ment_hlth": [self.mental_health],
                    "phys_hlth": [self.physical_health],
                    "diff_walk": [self.diff_walk],
                    "sex": [self.sex],
                    "age": [age],
                    "education": [self.education],
                    "income": [income],
                    "bmi_age": [bmi_age]
                }
            
            data = pd.DataFrame(data_dict)

            data_scaled = preprocessor.transform(data)

            pred = model.predict(data_scaled)
            parsed_pred = int(pred[0])

            return parsed_pred
        
        except Exception as e:
            raise CustomException(e, sys)


    def get_age_category(self):
        if self.age <= 24:
            return 1
        elif self.age <= 29:
            return 2
        elif self.age <= 34:
            return 3
        elif self.age <= 39:
            return 4
        elif self.age <= 44:
            return 5
        elif self.age <= 49:
            return 6
        elif self.age <= 54:
            return 7
        elif self.age <= 59:
            return 8
        elif self.age <= 64:
            return 9
        elif self.age <= 69:
            return 10
        elif self.age <= 74:
            return 11
        elif self.age <= 79:
            return 12
        else:
            return 13
        
    def get_income_category(self):
        if self.income < 10000:
            return 1
        elif self.income < 15000:
            return 2
        elif self.income < 20000:
            return 3
        elif self.income < 25000:
            return 4
        elif self.income < 35000:
            return 5
        elif self.income < 50000:
            return 6
        elif self.income < 75000:
            return 7
        else:
            return 8

    def preprocess(self):
        # Check if user has high blood pressure
        high_bp = 0

        if self.systolic_bp >= 130 or self.diastolic_bp > 80:
            high_bp = 1

        # Calculate BMI
        bmi = (self.weight * 703) / (self.height ** 2)

        # Categorize age
        age_cat = self.get_age_category()

        # Categorize income
        income_cat = self.get_income_category()

        # Get BMI Age interaction
        bmi_age = bmi * self.age

        # Impute unsure cholesterol data
        if self.high_chol_str == 'Yes':
            high_chol = 1
        elif self.high_chol_str == 'No':
            high_chol = 0
        else:
            high_chol = self.impute_chol(high_bp, bmi, age_cat, income_cat, bmi_age)

        return high_bp, high_chol, bmi, age_cat, income_cat, bmi_age

    def get_data_as_dataframe(self):
        try:
            high_bp, high_chol, bmi, age_cat, income_cat, bmi_age = self.preprocess()

            custom_data_input_dict = {
                "high_bp": [high_bp],
                "high_chol": [high_chol],
                "chol_check": [self.chol_check],
                "bmi": [bmi],
                "smoker": [self.smoker],
                "stroke": [self.stroke],
                "heart_disease_or_attack": [self.heart_disease_or_attack],
                "phys_activity": [self.physical_activity],
                "fruits": [self.fruits],
                "veggies": [self.veggies],
                "hvy_alcohol_consump": [self.heavy_alc_consumption],
                "any_healthcare": [self.healthcare],
                "no_doc_bc_cost": [self.doctor_no_care],
                "gen_hlth": [self.general_health],
                "ment_hlth": [self.mental_health],
                "phys_hlth": [self.physical_health],
                "diff_walk": [self.diff_walk],
                "sex": [self.sex],
                "age": [age_cat],
                "education": [self.education],
                "income": [income_cat],
                "bmi_age": [bmi_age]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)