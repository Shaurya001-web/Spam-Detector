from pathlib import Path

import streamlit as st

from model_backend import predict_spam, train_spam_model


st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")
st.title("📩 SMS/Email Spam Detector")
st.caption("Frontend for your notebook model using TF-IDF + RandomForest")


@st.cache_resource
def get_artifacts(dataset_path: str):
    return train_spam_model(dataset_path)


dataset_path = Path(__file__).parent / "combined_dataset.csv"

if not dataset_path.exists():
    st.error("`combined_dataset.csv` not found in project folder.")
    st.stop()

try:
    artifacts = get_artifacts(str(dataset_path))
except Exception as exc:  # pragma: no cover - UI fallback
    st.error(f"Failed to train model: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Rows Used", artifacts.row_count)
col2.metric("Model Accuracy", f"{artifacts.accuracy:.2%}")
col3.metric("Spam Ratio", f"{artifacts.spam_count / artifacts.row_count:.2%}")

st.divider()
st.subheader("Try a message")

message = st.text_area(
    "Enter message text",
    placeholder="Congratulations! You won a free trip. Click now...",
    height=140,
)

predict_btn = st.button("Predict", type="primary")

if predict_btn:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        label, confidence = predict_spam(message, artifacts)
        if label == "Spam":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.write(f"Confidence: **{confidence:.2%}**")

st.divider()
st.subheader("Quick examples")

examples = [
    "Hey, are we still meeting at 6 pm today?",
    "WINNER!! Claim your $500 gift card now by clicking this link.",
    "Your OTP is 482931. Do not share it with anyone.",
]

for sample in examples:
    if st.button(sample):
        label, confidence = predict_spam(sample, artifacts)
        st.write(f"**{sample}**")
        st.write(f"Prediction: **{label}** | Confidence: **{confidence:.2%}**")
