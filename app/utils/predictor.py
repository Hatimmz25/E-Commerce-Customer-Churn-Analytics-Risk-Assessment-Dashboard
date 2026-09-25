import pandas as pd
import numpy as np

def get_risk_tier(p, threshold):
    """Returns standardized human-readable risk tier based on probability and threshold."""
    if p >= threshold:
        return "HIGH RISK"
    elif p >= (threshold / 2.0):
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def predict_single(model_pipeline, raw_input_df, threshold):
    """Executes single customer prediction and returns structured risk dict."""
    proba = float(model_pipeline.predict_proba(raw_input_df)[:, 1][0])
    pred_class = 1 if proba >= threshold else 0
    risk_level = get_risk_tier(proba, threshold)
    
    if risk_level == "HIGH RISK":
        risk_style, risk_color = "risk-card-high", "#dc2626"
        explanation = f"Estimated churn probability ({proba:.1%}) meets or exceeds decision threshold ({threshold:.1%}). Proactive retention action recommended."
    elif risk_level == "MEDIUM RISK":
        risk_style, risk_color = "risk-card-medium", "#d97706"
        explanation = f"Estimated churn probability ({proba:.1%}) is approaching threshold ({threshold:.1%}). Monitor customer engagement."
    else:
        risk_style, risk_color = "risk-card-low", "#16a34a"
        explanation = f"Estimated churn probability ({proba:.1%}) is well below decision threshold ({threshold:.1%}). Standard retention holds."

    return {
        "probability": proba,
        "predicted_class": pred_class,
        "risk_level": risk_level,
        "risk_style": risk_style,
        "risk_color": risk_color,
        "explanation": explanation
    }