#!/usr/bin/env python3
"""Job Fraud Detection Dashboard
===========================================================
A Dash web-app to explore pre-computed job-fraud predictions with pre-trained XGBoost model.

Run
---
```bash
pip install -U dash dash-bootstrap-components pandas plotly beautifulsoup4 requests joblib requests-html
python job_fraud_dashboard.py --data predictions.csv --model model.pkl
```
Browse to **http://127.0.0.1:8050** to view the dashboard.
"""

import argparse
import os
import joblib
import pandas as pd

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_predictions(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "fake_percentage" not in df.columns:
            raise ValueError("CSV must have 'fake_percentage'.")
        return df
    return pd.DataFrame(columns=[
        "text", "telecommuting", "has_company_logo", "has_questions", "fake_percentage"
    ])


def label(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["is_fake"] = (d["fake_percentage"] >= 50).map({True: "Fake", False: "Real"})
    return d


def figures(df: pd.DataFrame):
    hist = px.histogram(df, x="fake_percentage", nbins=40,
                        title="Fraud-Probability Distribution",
                        labels={"fake_percentage": "Fraud Probability (%)"})
    pie = px.pie(df, names="is_fake", title="Fake vs Real")
    return hist, pie


# ─────────────────────────────────────────────────────────────────────────────
# Dash factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(df0: pd.DataFrame, model_path: str):
    model = joblib.load(model_path)
    df0 = label(df0)
    hist0, pie0 = figures(df0)
    top0 = df0.sort_values("fake_percentage", ascending=False).head(10)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        html.H1("Job Fraud Detection Dashboard", className="text-center my-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="hist", figure=hist0), md=6),
            dbc.Col(dcc.Graph(id="pie", figure=pie0), md=6),
        ]),
        html.H3("Explainability Overview"),
        html.Img(src="/assets/shap_summary.png", style={"width": "100%"}),
        html.Br(),
        html.Img(src="/assets/important_keywords.png", style={"width": "100%"}),


        html.H3("Top 10 Suspicious Listings", className="mt-4"),
        dash_table.DataTable(id="tbl-top", data=top0.to_dict("records"),
                             columns=[{"name": i, "id": i} for i in top0.columns],
                             style_table={"overflowX": "auto"},
                             page_size=10),
        html.H3("All Predictions", className="mt-4"),
        dash_table.DataTable(id="tbl-all", data=df0.to_dict("records"),
                             columns=[{"name": i, "id": i} for i in df0.columns],
                             filter_action="native", sort_action="native",
                             style_table={"overflowX": "auto"}, page_size=15),
        dcc.Store(id="store", data=df0.to_dict("records")),
    ], fluid=True)

    
    

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Dash fraud dashboard")
    parser.add_argument("--data", default="predictions.csv", help="CSV with pre‑computed predictions")
    parser.add_argument("--model", default="model.pkl", help="Trained XGBoost pipeline (.pkl)")
    parser.add_argument("--port", type=int, default=8050, help="Port")
    args = parser.parse_args()

    df_init = load_predictions(args.data)
    app = create_app(df_init, args.model)
    app.run(debug=True, port=args.port)
