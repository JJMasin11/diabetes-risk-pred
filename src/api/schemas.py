from pydantic import BaseModel, Field, field_validator

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
    mental_health: int = Field(ge=0, le=30)
    physical_health: int = Field(ge=0, le=30)
    diff_walk: int
    sex: int
    age: int
    education: int
    income: int

    @field_validator('height')
    def height_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Height must be greater than zero.")
        return v
    
    @field_validator('weight')
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Weight must be greater than zero.")
        return v
    
    @field_validator('age')
    def age_over_eighteen(cls, v):
        if v < 18:
            raise ValueError("Age must be 18 or older. This model was trained on adults only.")
        return v

class PredictResponse(BaseModel):
    diabetes_risk: int
    diabetes_probability: float
    shap_values: dict