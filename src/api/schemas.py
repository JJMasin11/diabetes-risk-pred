from pydantic import BaseModel

class PredictRequest(BaseModel):
    systolic_bp: int
    diastolic_bp: int
    high_chol: str
    chol_check: int
    height: int
    weight: int
    smoker: int
    stroke: int
    heart_disease_or_attack: int
    physical_activity: int
    fruits: int
    veggies: int
    heavy_alc_consumption: int
    healthcare: int
    doctor_no_care: int
    general_health: int
    mental_health: int
    physical_health: int
    diff_walk: int
    sex: int
    age: int
    education: int
    income: int

class PredictResponse(BaseModel):
    diabetes_risk: int
    diabetes_probability: float
    shap_values: dict