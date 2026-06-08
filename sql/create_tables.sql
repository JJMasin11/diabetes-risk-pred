-- Create health indicators table
CREATE TABLE health_indicators (
    id SERIAL PRIMARY KEY,
    diabetes_012 SMALLINT NOT NULL,
    high_bp SMALLINT NOT NULL,
    high_chol SMALLINT NOT NULL,
    chol_check SMALLINT NOT NULL,
    bmi INT NOT NULL,
    smoker SMALLINT NOT NULL,
    stroke SMALLINT NOT NULL,
    heart_disease_or_attack SMALLINT NOT NULL,
    phys_activity SMALLINT NOT NULL,
    fruits SMALLINT NOT NULL,
    veggies SMALLINT NOT NULL,
    hvy_alcohol_consump SMALLINT NOT NULL,
    any_healthcare SMALLINT NOT NULL,
    no_doc_bc_cost SMALLINT NOT NULL,
    gen_hlth INT NOT NULL,
    ment_hlth INT NOT NULL,
    phys_hlth INT NOT NULL,
    diff_walk SMALLINT NOT NULL,
    sex SMALLINT NOT NULL,
    age INT NOT NULL,
    education INT NOT NULL,
    income INT NOT NULL
);

-- Create predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    systolic_bp INT NOT NULL,
    diastolic_bp INT NOT NULL,
    high_chol_input VARCHAR(6) NOT NULL, --'Yes', 'No', or 'Unsure'
    chol_check SMALLINT NOT NULL,
    height INT NOT NULL,
    "weight" INT NOT NULL,
    smoker SMALLINT NOT NULL,
    stroke SMALLINT NOT NULL,
    heart_disease_or_attack SMALLINT NOT NULL,
    phys_activity SMALLINT NOT NULL,
    fruits SMALLINT NOT NULL,
    veggies SMALLINT NOT NULL,
    hvy_alcohol_consump SMALLINT NOT NULL,
    any_healthcare SMALLINT NOT NULL,
    no_doc_bc_cost SMALLINT NOT NULL,
    gen_hlth INT NOT NULL,
    ment_hlth INT NOT NULL,
    phys_hlth INT NOT NULL,
    diff_walk SMALLINT NOT NULL,
    sex SMALLINT NOT NULL,
    age INT NOT NULL,
    education INT NOT NULL,
    income INT NOT NULL,

    high_bp SMALLINT NOT NULL,
    bmi FLOAT NOT NULL,
    age_category INT NOT NULL,
    income_category INT NOT NULL,
    bmi_age FLOAT NOT NULL,
    high_chol SMALLINT NOT NULL,

    chol_imputed BOOLEAN NOT NULL,
    chol_imputed_probability FLOAT NOT NULL,

    diabetes_prediction SMALLINT NOT NULL,
    diabetes_probability FLOAT NOT NULL
);