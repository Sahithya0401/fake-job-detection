import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake Job Detection System")

job_text = st.text_area(
    "Paste Job Description Here"
)

if st.button("Predict"):

    text_vector = vectorizer.transform(
        [job_text]
    )

    prediction = model.predict(
        text_vector
    )[0]

    if prediction == 1:
        st.error("Fraudulent Job Posting")
    else:
        st.success("Legitimate Job Posting")