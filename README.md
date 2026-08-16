# 🎓 Student Academic Performance & Digital Well-being Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-satvik89076--gif-181717?logo=github)](https://github.com/satvik89076-gif/Student-performance-predicter)

An end-to-end Machine Learning pipeline and interactive web dashboard designed to predict student final examination scores ($G3$) using both **traditional academic records** and **modern digital well-being metrics** (sleep duration, recreational screen time, and perceived stress levels).

---

## 📌 Project Overview

Educational predictive systems frequently focus solely on historical grade transcripts and attendance. This project augments classic academic performance indicators with digital lifestyle features to evaluate how modern habits (screen time, sleep deprivation, stress) influence final examination outcomes.

### 🌟 Key Highlights
* **Feature Engineering:** Synthesized realistic digital well-being variables (`sleep_hours`, `screen_time_hours`, `stress_level`) correlated with health, freetime, study habits, and past failures.
* **Model Benchmarking:** Evaluated Linear Regression, Random Forest, and Gradient Boosting Regressors across standard regression metrics ($MAE$, $RMSE$, $R^2$).
* **Feature Importance Analysis:** Discovered that lifestyle factors (`sleep_hours`, `screen_time_hours`) rank within the top 10 most influential predictors alongside earlier exam marks ($G1$, $G2$).
* **Live Web Deployment:** Built with Streamlit and deployed on Streamlit Community Cloud with dynamic risk-scoring and actionable recommendations.

---

## 🏗️ System Architecture

```text
├── data/
│   ├── student-mat.csv                 # Raw UCI student performance dataset
│   └── student_lifestyle_data.csv      # Augmented dataset with lifestyle features
├── models/
│   ├── best_student_model.pkl          # Serialized Gradient Boosting Regressor
│   └── model_features.pkl              # Feature schema for inference alignment
├── notebooks/
│   └── eda.ipynb                       # Data exploration, feature engineering & model training
├── app.py                              # Streamlit interactive web application
├── requirements.txt                    # Project dependencies for local & cloud runtime
└── README.md                           # Project documentation
