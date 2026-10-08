"""Clinical vocabularies used by the synthetic EHR generator.

The first entries of each list are clinically meaningful concepts that the
simulator ties to the latent health state; the remainder are filler codes with
no signal, mimicking the long tail of real EHR vocabularies.
"""
from __future__ import annotations

# (name, mean, sd, loading on latent frailty in SD units, lower/upper plausible)
INFORMATIVE_LABS = [
    ("albumin", 3.9, 0.5, -0.45),
    ("hemoglobin", 12.8, 1.7, -0.40),
    ("c_reactive_protein", 12.0, 9.0, 0.40),
    ("white_blood_cells", 7.5, 2.4, 0.25),
    ("neutrophil_lymph_ratio", 3.5, 2.0, 0.35),
    ("creatinine", 0.95, 0.30, 0.20),
    ("alkaline_phosphatase", 95.0, 40.0, 0.25),
    ("lactate_dehydrogenase", 210.0, 70.0, 0.30),
    ("sodium", 139.0, 3.0, -0.20),
    ("platelets", 260.0, 80.0, 0.15),
    ("total_protein", 7.0, 0.6, -0.25),
    ("glucose", 105.0, 25.0, 0.15),
]
FILLER_LABS = [
    "potassium", "chloride", "bicarbonate", "bun", "calcium", "magnesium",
    "phosphorus", "ast", "alt", "total_bilirubin", "direct_bilirubin", "ggt",
    "inr", "ptt", "mcv", "mch", "mchc", "rdw", "hematocrit", "rbc",
    "monocytes", "eosinophils", "basophils", "lymphocytes", "tsh", "free_t4",
    "ferritin", "iron", "tibc", "b12", "folate", "uric_acid", "hba1c",
    "triglycerides", "ldl", "hdl", "cea",
]

INFORMATIVE_DX = [
    ("depression", 0.12, 0.60), ("anxiety", 0.14, 0.45),
    ("chronic_pain", 0.10, 0.65), ("malnutrition", 0.04, 0.90),
    ("copd", 0.08, 0.60), ("heart_failure", 0.05, 0.70),
    ("chronic_kidney_disease", 0.07, 0.55), ("type2_diabetes", 0.16, 0.35),
    ("peripheral_neuropathy", 0.06, 0.55), ("anemia", 0.18, 0.55),
    ("fatigue", 0.20, 0.60), ("weight_loss", 0.09, 0.75),
    ("insomnia", 0.08, 0.40), ("obesity", 0.20, 0.15),
    ("atrial_fibrillation", 0.06, 0.40), ("falls", 0.03, 0.70),
    ("hypertension", 0.35, 0.25), ("cachexia", 0.02, 0.95),
]

INFORMATIVE_MEDS = [
    ("opioid_analgesics", 0.25, 0.65), ("antidepressants", 0.18, 0.50),
    ("corticosteroids", 0.20, 0.40), ("benzodiazepines", 0.12, 0.45),
    ("antiemetics", 0.30, 0.35), ("loop_diuretics", 0.08, 0.60),
    ("insulin", 0.07, 0.40), ("anticoagulants", 0.12, 0.40),
    ("appetite_stimulants", 0.04, 0.80), ("gabapentinoids", 0.12, 0.50),
]

INFORMATIVE_PROCS = [
    ("ct_chest", 0.55, 0.15), ("pet_ct", 0.40, 0.10),
    ("port_placement", 0.50, 0.05), ("blood_transfusion", 0.06, 0.80),
    ("echocardiogram", 0.15, 0.40), ("paracentesis", 0.03, 0.90),
    ("thoracentesis", 0.04, 0.80), ("physical_therapy_eval", 0.10, 0.55),
    ("nutrition_consult", 0.08, 0.65), ("emergency_imaging", 0.12, 0.55),
]

# (name, mean, sd, loading on frailty, (min, max) physiologically possible)
VITALS = [
    ("systolic_bp", 128.0, 16.0, -0.10, (50, 260)),
    ("diastolic_bp", 77.0, 10.0, -0.10, (25, 160)),
    ("heart_rate", 78.0, 12.0, 0.35, (25, 220)),
    ("respiratory_rate", 16.5, 2.2, 0.30, (5, 60)),
    ("temperature_c", 36.8, 0.35, 0.10, (33, 42.5)),
    ("spo2", 97.2, 1.8, -0.35, (60, 100)),
    ("bmi", 27.0, 5.5, -0.25, (12, 70)),
]

# (name, relative effect on on-treatment decline; positive = larger decline)
CANCER_TYPES = [
    ("breast", -0.6), ("colorectal", 0.1), ("lung", 0.7), ("pancreatic", 1.2),
    ("lymphoma", 0.0), ("ovarian", 0.4), ("gastric", 0.8), ("prostate", -0.3),
    ("head_neck", 0.6), ("leukemia", 0.5), ("sarcoma", 0.3), ("other", 0.1),
]
CANCER_TYPE_PROBS = [0.22, 0.12, 0.13, 0.06, 0.09, 0.06, 0.04, 0.05,
                     0.06, 0.05, 0.03, 0.09]

RACES = ["white", "asian", "black", "pacific_islander", "native_american",
         "other", "unknown"]
RACE_PROBS = [0.52, 0.24, 0.05, 0.02, 0.01, 0.12, 0.04]
ETHNICITIES = ["non_hispanic", "hispanic", "unknown"]
ETHNICITY_PROBS = [0.80, 0.15, 0.05]
INSURANCE = ["private", "medicare", "medicaid", "self_pay", "other"]
LANGUAGES = ["english", "spanish", "mandarin", "cantonese", "vietnamese",
             "russian", "other"]
LANGUAGE_PROBS = [0.82, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01]
MARITAL = ["married", "single", "divorced", "widowed", "unknown"]
MARITAL_PROBS = [0.58, 0.20, 0.10, 0.08, 0.04]

ENCOUNTER_TYPES = ["emergency", "inpatient", "outpatient", "telehealth", "urgent_care"]

CHEMO_MED_CLASS = "antineoplastic_chemotherapy"  # defines the index date; never a feature
