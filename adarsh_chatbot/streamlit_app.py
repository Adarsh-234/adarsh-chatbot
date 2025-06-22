import streamlit as st
import spacy
import joblib

st.title("🩺 AI Medical Detection Chatbot")

# Load models
nlp = spacy.load("en_core_web_sm")
model = joblib.load("model/rf_model.pkl")
mlb = joblib.load("model/mlb.pkl")

def extract_symptoms(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ == "SYMPTOM"]

user_input = st.text_input("Describe your symptoms:")

if user_input:
    symptoms = extract_symptoms(user_input)
    st.write("✅ Extracted symptoms:", symptoms)
    if symptoms:
        binary_input = mlb.transform([symptoms])
        prediction = model.predict(binary_input)
        st.write("🧠 Predicted condition:", prediction[0])
    else:
        st.warning("Could not detect any known symptoms.")