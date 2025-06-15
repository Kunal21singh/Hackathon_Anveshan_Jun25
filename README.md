
# 🕵️‍♂️ Job Fraud Detection System

Detect and visualize potentially fraudulent job listings using an XGBoost classifier and an interactive Dash dashboard.

---

## 📦 Project Components

```
├── job_fraud_xgb.py          # Model training & prediction script
├── job_fraud_dashboard.py    # Dash dashboard to visualize predictions
├── predict_from_model.py     # Script to use a saved model on new data
├── Train.csv                 # Training dataset
├── Test.csv                  # New dataset to test predictions
├── model.pkl                 # Trained XGBoost model (output)
├── predictions.csv           # Model prediction output on training data
├── predicted_jobs.csv        # Model prediction output on test data
```

---

## 📊 Dashboard Overview

The dashboard displays:
- Histogram of fraud probabilities
- Pie chart of real vs fake listings
- Top 10 most suspicious job postings
- Full searchable/filterable dataset

### 🔧 Run the Dashboard

```bash
pip install dash dash-bootstrap-components pandas plotly joblib
python job_fraud_dashboard.py --data predictions.csv --model model.pkl
```

Visit: `http://127.0.0.1:8050` in your browser

---

## 🧠 Train the Model

Use the XGBoost-based training script to build your model:

```bash
pip install pandas scikit-learn xgboost joblib
python job_fraud_xgb.py --train Train.csv --output predictions.csv --model-out model.pkl
```

The script outputs:
- `predictions.csv` – original data + `fake_percentage` column
- `model.pkl` – trained pipeline saved for reuse

---

## 🧪 Run Predictions on New Data

To predict fraud probability on new job listings:

```bash
python predict_from_model.py
```

This uses:
- `Test.csv` as input
- `model.pkl` for prediction
- Outputs `predicted_jobs.csv` with `fake_percentage`

---

## 📌 Required Columns in Data

Make sure both training and test CSVs include:

- `description`
- `telecommuting`
- `has_company_logo`
- `has_questions`

The training data must also include:
- `fraudulent` (0 = real, 1 = fake)

---

## 🛠️ Future Enhancements

- Live job posting scraping integration
- Confidence thresholds & alert system
- Model explainability (SHAP/feature importance)

---

