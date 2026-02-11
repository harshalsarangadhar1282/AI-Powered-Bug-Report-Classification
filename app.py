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
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🚀 AI-Powered Bug Report Classification System</div>", unsafe_allow_html=True)
st.write("")

# ================= LOAD MODEL (CORRECTED PATH) =================
model = joblib.load("xgboost_bug_report_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ================= TABS =================
tab1, tab2 = st.tabs(["🔍 Live Prediction", "📊 Model Evaluation"])

# ======================================================
# TAB 1 — LIVE PREDICTION
# ======================================================

with tab1:
    st.subheader("Paste Bug Report Below")

    user_input = st.text_area("Bug Report Text:", height=180)

    if st.button("Predict Bug Category"):

        if user_input.strip() == "":
            st.warning("Please enter bug report text.")
        else:
            vector = tfidf.transform([user_input])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0][1]

            if prediction == 1:
                st.success("⚡ Performance Related Bug Detected")
            else:
                st.error("🐞 Non-Performance Bug")

            st.write(f"Confidence Score: {probability:.2f}")
            st.progress(float(probability))

# ======================================================
# TAB 2 — MODEL EVALUATION
# ======================================================

with tab2:
    st.subheader("Model Performance Metrics")

    try:
        data = pd.read_csv("Title+Body.csv").fillna("")
    except:
        st.error("Dataset not found in repository.")
        st.stop()

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

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
