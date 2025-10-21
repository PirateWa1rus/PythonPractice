# Save this as DsJobs_app.py and run: streamlit run DSJobs_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

df_filtered = pd.read_csv("C:\RepoClones\PythonPractice\DSJobs\df_filtereed.csv")
title_list = df_filtered['job_title'].dropna().unique()
industries = df_filtered['industry'].dropna().unique()

def skill_tokenizer(text):
    return [t for t in text.split('|') if t.strip()]

skill_list = []
for skills in df_filtered['skills_parsed'].dropna():
    tokens = skill_tokenizer(skills)
    for t in tokens:
        skill_list.append(t.strip(" []'\"").lower())
    

skill_list = list(set(skill_list))
    


pipeline = joblib.load(r"C:\RepoClones\PythonPractice\DSJobs\models\baseline_ridge_pipeline.joblib")
# Load or wrap quantile models (q10,q50,q90) - save and load similarly if trained separately

st.title("Salary range predictor (demo)")

seniority = st.selectbox("Experience level", ["junior", "senior"])
skills = st.multiselect("Skills", skill_list, default=["python", "sql", "machine learning"])
job_title = st.selectbox("Job title", title_list)
industry = st.selectbox("Industry", industries)
work_type = st.selectbox("Work type", ["onsite", "hybrid", "remote"])


if st.button("Predict"):
    # Preprocess single input into DataFrame row
    row = pd.DataFrame([{
        'Seniority_Binary': seniority,
        'skills_parsed': " | ".join(skills),
        'job_title': job_title,
        'industry': industry,
        'work_type': work_type
    }])
    pred_log = pipeline.predict(row)[0]
    median = np.expm1(pred_log)
    # For intervals, use quantile models; here we show a simple +/- heuristic
    lb = max(0, median * 0.8)
    ub = median * 1.25
    st.write(f"Predicted salary median: €{median:,.0f}")
    st.write(f"Estimated range: €{lb:,.0f} — €{ub:,.0f}")
    st.info("This demo uses a baseline model; train quantile LightGBM models for calibrated intervals.")

