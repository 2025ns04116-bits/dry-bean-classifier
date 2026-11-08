import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("🫘 Dry Bean Variety Classifier")
st.caption("UCI Dry Bean dataset · 7 bean classes · 16 shape features · 5 ML models")

MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'kNN': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Support Vector Machine': 'model/svm.pkl',
}


@st.cache_resource
def load(path):
    return joblib.load(path)


scaler = load('model/scaler.pkl')
enc = load('model/label_encoder.pkl')


def get_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(y, proba, multi_class='ovr', labels=model.classes_)
    except ValueError:
        auc = float('nan')
    return pred, {
        'Accuracy': accuracy_score(y, pred),
        'AUC': auc,
        'Precision': precision_score(y, pred, average='weighted', zero_division=0),
        'Recall': recall_score(y, pred, average='weighted', zero_division=0),
        'F1': f1_score(y, pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y, pred),
    }


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
choice = st.sidebar.selectbox("Select model", list(MODEL_FILES))

if file is None:
    st.info("👈 Upload test_data.csv from the sidebar to see results.")
    st.stop()

df = pd.read_csv(file)
if 'Class' not in df.columns:
    st.error("The CSV must contain a 'Class' column with the true labels.")
    st.stop()

st.subheader("📄 Uploaded data")
st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head())

X = scaler.transform(df.drop(columns=['Class']))
y = enc.transform(df['Class'])

model = load(MODEL_FILES[choice])
pred, m = get_metrics(model, X, y)

# ---------------- Metrics ----------------
st.subheader(f"📊 Evaluation metrics — {choice}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

# ---------------- Confusion matrix + report ----------------
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt='d', cmap='YlGn',
                xticklabels=enc.classes_, yticklabels=enc.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, pred, target_names=enc.classes_,
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

# ---------------- Compare all models ----------------
if st.checkbox("🏆 Compare all 5 models"):
    rows = {name: get_metrics(load(p), X, y)[1] for name, p in MODEL_FILES.items()}
    comp = pd.DataFrame(rows).T.round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color='lightgreen'))
    st.bar_chart(comp['Accuracy'])
