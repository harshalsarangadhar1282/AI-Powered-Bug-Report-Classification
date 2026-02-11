import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Bug Report Classifier",
    page_icon="🐞",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #1f4037, #99f2c8);
}
.title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
}
.result-card {
    padding: 20px;
    border-radius: 10px;
    background-color: rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚀 AI-Powered Bug Report Classification System</div>", unsafe_allow_html=True)
st.write("")

# ================= LOAD MODEL =================
model = joblib.load("xgboost_bug_report_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ================= TABS =================
tab1, tab2 = st.tabs(["🔍 Live Prediction", "📊 Model Evaluation"])

# ======================================================
# TAB 1 — LIVE PREDICTION
# ======================================================

with tab1:

    st.subheader("Paste Bug Report Below")

    user_input = st.text_area("Bug Report Text:", height=200)

    if st.button("Predict Bug Category"):

        if user_input.strip() == "":
            st.warning("Please enter bug report text.")
        else:

            vector = tfidf.transform([user_input])
            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]
            probability = probabilities[1]

            st.markdown("## 🧾 Prediction Result")

            # ---------------- Classification ----------------
            if prediction == 1:
                st.success("⚡ Performance Related Bug Detected")
            else:
                st.error("🐞 Non-Performance Bug")

            # ---------------- Confidence ----------------
            st.write(f"### Confidence Score: {probability:.2f}")
            st.progress(float(probability))

            # ---------------- Severity ----------------
            if probability > 0.90:
                severity = "🔴 P1 - Critical"
            elif probability > 0.70:
                severity = "🟠 P2 - High"
            else:
                severity = "🟡 P3 - Moderate"

            st.write(f"### Severity Level: {severity}")

            # ---------------- Probability Breakdown ----------------
            st.write("### 📊 Probability Breakdown")
            st.write(f"Non-Performance: {probabilities[0]:.2f}")
            st.write(f"Performance: {probabilities[1]:.2f}")

            # ---------------- Keyword Detection ----------------
            performance_terms = ["slow", "memory", "cpu", "gpu", "latency", "throughput", "optimization"]
            detected = [term for term in performance_terms if term in user_input.lower()]

            if detected:
                st.write("### 🔎 Detected Performance Indicators")
                st.write(detected)

            # ---------------- Root Cause Suggestion ----------------
            if "memory" in user_input.lower():
                suggestion = "Possible memory leak or inefficient memory handling."
            elif "cpu" in user_input.lower():
                suggestion = "High CPU utilization due to heavy processing."
            elif "latency" in user_input.lower():
                suggestion = "Network or backend processing delay."
            elif "slow" in user_input.lower():
                suggestion = "Inefficient algorithm or heavy resource usage."
            else:
                suggestion = "Further log and system profiling required."

            st.info(f"🧠 Suggested Root Cause: {suggestion}")

            # ---------------- Suggested Solution ----------------
            if prediction == 1:
                solution = """
                ### ✅ Recommended Optimization Steps:
                - Optimize algorithm complexity
                - Check for memory leaks
                - Profile CPU/GPU usage
                - Implement caching or batching
                - Improve database indexing
                """
            else:
                solution = """
                ### ✅ Recommended Debugging Steps:
                - Validate input handling
                - Check exception logs
                - Verify API responses
                - Inspect UI event triggers
                """

            st.success(solution)


# ======================================================
# TAB 2 — MODEL EVALUATION
# ======================================================

with tab2:

    st.subheader("Model Performance Metrics")

    try:
        data = pd.read_csv("data/Title+Body.csv").fillna("")
    except:
        st.error("Dataset not found in data folder.")
        st.stop()

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        f1_score,
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score
    )

    le = LabelEncoder()
    data["sentiment"] = le.fit_transform(data["sentiment"])

    X = tfidf.transform(data["text"])
    y = data["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_probs)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    }

    st.write("### 📈 Metrics")
    st.write(metrics)

    # Bar Chart
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    st.pyplot(fig)

    # Confusion Matrix
    st.write("### 🔢 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    st.info("""
    Confusion Matrix Explanation:
    - Top Left  → True Negatives
    - Top Right → False Positives
    - Bottom Left → False Negatives
    - Bottom Right → True Positives
    """)
