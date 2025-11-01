# Smoking-Related-Fatalities-Prediction
This project builds a hybrid machine learning and statistical model to predict smoking-related fatalities using public health data. It integrates a Poisson GLM for disease categories strongly linked to smoking (e.g., cancers, circulatory and respiratory diseases) and an XGBoost Regressor for all other diagnoses.

## Project Overview

**Goal:**  
To model and predict the **number of smoking-related fatalities** based on multiple health and lifestyle factors, and to identify the **key drivers** contributing to mortality rates.

**Datasets Used:**
- Admissions data (`df_adm`)
- Smoking-related fatalities (`df_fat`)
- Tobacco metrics and household expenditure (`df_met`)
- Smoking prevalence rates (`df_smo`)
- Prescription data for stop-smoking aids (`df_pre`)

**Data Sources:**  
NHS Digital, Office for National Statistics (ONS), UK Health Survey data.

---

## Technologies & Libraries Used

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.10 |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Modeling | XGBoost |
| Model Explainability | SHAP, sklearn.inspection |
| Environment | Jupyter Notebook |

---
## Project Structure
Smoking-Related-Fatalities-Prediction/
- │
- ├── data/                         # datasets
- ├── notebooks/
- │   └── Smoking_Fatalities_Model.ipynb
- ├── requirements.txt
- ├── README.md
- └── outputs/                      # Model results, charts, and visualizations

---
## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/ShravaniB18/Smoking-Related-Fatalities-Prediction.git
cd Smoking-Related-Fatalities-Prediction
pip install -r requirements.txt
```

