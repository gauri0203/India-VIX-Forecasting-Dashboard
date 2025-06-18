import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from processor import *

st.set_page_config(page_title="India VIX Forecasting", layout="wide")

# Title
st.title("📈 India VIX Forecasting Dashboard")

# Sidebar
st.sidebar.header("Model Configuration")

# Load or run the pipeline
models, feature_names = load_models()

if not models:
    st.warning("Models not found. Running training pipeline...")
    results = run_complete_pipeline()
    models = results['models']
    metrics = results['metrics']
    predictions = results['predictions']
    y_test = results['y_test']
    X_test = results['X_test']
    df = results['data']
else:
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("vix_models.pkl", "rb") as f:
        models = pickle.load(f)

    # Load full pipeline to get metrics
    results = run_complete_pipeline()
    metrics = results['metrics']
    predictions = results['predictions']
    y_test = results['y_test']
    X_test = results['X_test']
    df = results['data']

# Sidebar: model selection
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Show model performance
st.subheader(f"📊 Performance Metrics for {model_name}")
metric_vals = metrics[model_name]
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{metric_vals['RMSE']:.4f}")
col2.metric("MAE", f"{metric_vals['MAE']:.4f}")
col3.metric("R² Score", f"{metric_vals['R2']:.4f}")

# Plot actual vs predicted
st.subheader("📉 Actual vs Predicted VIX Close")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_test.reset_index(drop=True), label="Actual", color='blue')
ax.plot(predictions[model_name], label="Predicted", color='red')
ax.set_xlabel("Test Data Index")
ax.set_ylabel("VIX Close")
ax.legend()
st.pyplot(fig)

# Feature importance if applicable
if model_name in ["Random Forest", "Gradient Boosting"]:
    st.subheader(f"🔍 Feature Importance: {model_name}")
    model = models[model_name]
    importances = model.feature_importances_
    top_features = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax, palette="viridis")
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

# Optional: Prediction for new input
st.subheader("🎯 Try Custom Prediction")

# Let user input values for top 5 features only
example_input = {}
top_5 = feature_names[:5]

with st.form("prediction_form"):
    st.write("Enter values for features:")
    for feature in top_5:
        val = st.number_input(f"{feature}", value=float(X_test.iloc[-1][feature]))
        example_input[feature] = val
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([example_input])
    
    # Fill remaining columns with last known test values
    for feat in feature_names:
        if feat not in input_df.columns:
            input_df[feat] = X_test.iloc[-1][feat]

    if isinstance(models[model_name], tuple):
        model, scaler = models[model_name]
        input_scaled = scaler.transform(input_df[feature_names])
        pred = model.predict(input_scaled)
    else:
        pred = models[model_name].predict(input_df[feature_names])
    
    st.success(f"Predicted VIX Close: {pred[0]:.4f}")

# Show raw data toggle
with st.expander("📄 Show Raw Data"):
    st.dataframe(df.tail(10))

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit for India VIX Forecasting.")
