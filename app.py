import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Student Performance & Well-being Predictor",
    page_icon="🎓",
    layout="wide",
)

# Title and introduction
st.title("🎓 Student Academic Performance & Lifestyle Impact Predictor")
st.markdown(
    """
Predict a student's final examination score (*G3) based on both **academic history* and *modern lifestyle habits* (screen time, sleep, and stress).
"""
)

# Load trained model and feature list
try:
    model = joblib.load("models/best_student_model.pkl")
    model_features = joblib.load("models/model_features.pkl")
except Exception as e:
    st.error(
        f"Error loading model files. Please ensure you ran the training notebook first. Details: {e}"
    )
    st.stop()

# Layout: 2 Columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Academic Performance")
    g1 = st.slider(
        "First Period Grade - G1 (0 to 20)",
        min_value=0,
        max_value=20,
        value=12,
        step=1,
    )
    g2 = st.slider(
        "Second Period Grade - G2 (0 to 20)",
        min_value=0,
        max_value=20,
        value=12,
        step=1,
    )
    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "< 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "> 10 hours",
        }[x],
        index=1,
    )
    failures = st.number_input(
        "Number of Past Class Failures", min_value=0, max_value=4, value=0
    )
    absences = st.slider(
        "School Absences (Days)", min_value=0, max_value=50, value=4, step=1
    )
    higher = st.selectbox("Wants to take Higher Education?", ["yes", "no"])

with col2:
    st.subheader("🧘 Lifestyle & Digital Well-being")
    sleep_hours = st.slider(
        "Average Sleep (Hours/Night)",
        min_value=4.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
    )
    screen_time_hours = st.slider(
        "Daily Screen Time (Recreational Hours)",
        min_value=1.0,
        max_value=9.0,
        value=3.5,
        step=0.5,
    )
    stress_level = st.slider(
        "Perceived Stress Level (1 to 10)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )
    age = st.slider("Student Age", min_value=15, max_value=22, value=17, step=1)
    health = st.slider(
        "Current Health Status (1: very bad to 5: very good)",
        min_value=1,
        max_value=5,
        value=4,
        step=1,
    )

st.markdown("---")

# Predict button
if st.button("🚀 Predict Final Score (G3)", use_container_width=True):
    # Construct base dictionary matching training features
    input_data = {col: 0 for col in model_features}

    # Populate numerical and categorical values
    input_data["age"] = age
    input_data["studytime"] = studytime
    input_data["failures"] = failures
    input_data["absences"] = absences
    input_data["health"] = health
    input_data["sleep_hours"] = sleep_hours
    input_data["screen_time_hours"] = screen_time_hours
    input_data["stress_level"] = stress_level
    input_data["G1"] = g1
    input_data["G2"] = g2

    if higher == "yes" and "higher_yes" in input_data:
        input_data["higher_yes"] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_features]

    # Predict
    predicted_grade = model.predict(input_df)[0]
    predicted_grade = np.clip(predicted_grade, 0, 20)

    # Display results
    st.subheader("🎯 Prediction Result")
    score_col1, score_col2 = st.columns([1, 2])

    with score_col1:
        st.metric(
            label="Predicted Final Grade (G3)",
            value=f"{predicted_grade:.1f} / 20",
        )

    with score_col2:
        if predicted_grade >= 14:
            st.success(
                "🌟 *Distinction / High Performance:* Strong academic track record and balanced lifestyle habits."
            )
        elif predicted_grade >= 10:
            st.info(
                "👍 *Passing / Satisfactory:* Consistent, with potential for score improvement by optimizing study routine and sleep balance."
            )
        else:
            st.warning(
                "⚠️ *At Risk of Failing / Needs Intervention:* Consider reducing recreational screen time, improving sleep schedule, and focusing on foundational study hours."
            )
