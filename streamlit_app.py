import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl','multinomial_naive_bayes_model.pkl','support_vector_machine_(svm)_model.pkl')  # Replace with your model file
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Replace with your vectorizer

# Define phishing keywords (for highlighting)
PHISHING_KEYWORDS = [
    "urgent", "verify", "account", "suspended", "password", 
    "login", "click", "immediately", "security", "bank"
]

# Preprocess text (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

# Highlight phishing keywords
def highlight_keywords(text):
    for word in PHISHING_KEYWORDS:
        if word in text.lower():
            text = text.replace(word, f"<span style='color:red'>{word}</span>")
    return text

# Streamlit UI
st.title("📧 Phishing Email Detector")
st.markdown("""
    This app uses **NLP and Machine Learning** to detect phishing emails.  
    Paste an email below to check if it's safe or malicious.
""")

# User input
user_input = st.text_area("**Paste the email text here:**", height=200)

if st.button("Analyze Email"):
    if user_input:
        # Preprocess and vectorize
        processed_text = preprocess_text(user_input)
        text_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf)[0]
        
        # Display results
        st.subheader("🔍 Results")
        if prediction == "Phishing Email":
            st.error(f"**⚠️ Warning: Phishing Email Detected!** (Confidence: {proba[1]*100:.2f}%)")
        else:
            st.success(f"**✅ Safe Email** (Confidence: {proba[0]*100:.2f}%)")
        
        # Show highlighted keywords
        st.subheader("📌 Suspicious Keywords Found")
        highlighted_text = highlight_keywords(user_input)
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Explain prediction
        st.subheader("🤖 How This Works")
        st.write("""
            - The model analyzes **TF-IDF features** from the email text.
            - It checks for **phishing keywords** (highlighted in red).
            - **SVM classifier** predicts if the email is malicious.
        """)
    else:
        st.warning("Please enter an email to analyze.")
