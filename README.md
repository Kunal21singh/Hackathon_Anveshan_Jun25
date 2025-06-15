
# 🔍 Job Fraud Detection System

This project helps detect potentially fraudulent job postings using an XGBoost model and visualize the predictions in a browser dashboard.

---

## 📁 Project Structure

```
├── job_fraud_xgb.py            # Script to train an XGBoost model and generate fraud predictions
├── job_fraud_dashboard.py      # Dash web app to explore the predictions visually
├── Train.csv                   # Sample dataset of job postings with labels
├── predictions.csv             # Output predictions with fraud probability
├── model.pkl                   # (Optional) Serialized trained model
└── README.md                   # You're reading it!
```

---

## 📊 Dashboard Preview

![Dashboard Preview](https://user-images.githubusercontent.com/example/dashboard-preview.png)

---

## 🧠 1. Model Training (`job_fraud_xgb.py`)

This script trains an XGBoost classifier on job posting data and outputs:
- A `predictions.csv` file with fraud probability (`fake_percentage`)
- An optional serialized model (`model.pkl`)

### ▶️ Usage

```bash
pip install -U pandas scikit-learn xgboost joblib

python job_fraud_xgb.py --train Train.csv --output predictions.csv --model-out model.pkl
```

### 🔢 Input Columns Required in `Train.csv`

- `description` (text)
- `telecommuting` (0/1)
- `has_company_logo` (0/1)
- `has_questions` (0/1)
- `fraudulent` (target: 0 for real, 1 for fake)

---

## 📈 2. Dashboard (`job_fraud_dashboard.py`)

This Dash web app loads precomputed predictions and displays:
- Fraud probability distribution histogram
- Real vs Fake pie chart
- Top 10 most suspicious job listings
- Filterable full predictions table

### ▶️ Usage

```bash
pip install -U dash dash-bootstrap-components pandas plotly joblib

python job_fraud_dashboard.py --data predictions.csv --model model.pkl
```

Then open your browser at:

```
http://127.0.0.1:8050
```

---

## 📂 Files Explained

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `Train.csv`           | Dataset of labeled job postings                  |
| `predictions.csv`     | Output from model containing fraud probabilities |
| `model.pkl`           | Trained XGBoost model (optional but recommended) |
| `job_fraud_xgb.py`    | Model training and prediction generation script  |
| `job_fraud_dashboard.py` | Web dashboard using Dash to visualize results |

---

## 🧪 Sample Output (`predictions.csv`)

| text              | telecommuting | has_company_logo | has_questions | fake_percentage |
|------------------|----------------|------------------|----------------|------------------|
| ...              | 0              | 1                | 0              | 74.32            |
| ...              | 1              | 1                | 0              | 18.67            |

---

## ⚙️ Notes

- The `fake_percentage` is derived from the XGBoost predicted probability.
- Listings with `fake_percentage >= 50%` are classified as "Fake".

---

## 📌 Future Improvements

- Add live Naukri.com scraping support (currently removed from dashboard).
- Auto-refresh predictions for new job posts.
- Export flagged listings to email or Excel.

---

