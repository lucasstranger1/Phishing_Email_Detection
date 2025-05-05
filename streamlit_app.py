import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.utils.validation import check_is_fitted

# Configure page
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS (properly formatted)
st.markdown("""
<style>
.main {padding: 2rem;}
.stSelectbox div[data-baseweb="select"] {margin-bottom: 1rem;}
.stButton>button {width: 100%;}
.metric-card {border-radius: 0.5rem; padding: 1rem; background-color: #f0f2f6;}
.phishing {color: #ff4b4b;}
.safe {color: #00b050;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all models with validation"""
    try:
        vectorizer = joblib.load('unified_vectorizer.pkl')
        models = {
            "Logistic Regression": joblib.load('logistic_regression_unified_model.pkl'),
            "Naive Bayes": joblib.load('naive_bayes_unified_model.pkl'),
            "Support Vector Machine": joblib.load('svm_unified_model.pkl')
        }
        
        # Verify all models are properly trained
        for name, model in models.items():
            check_is_fitted(model)
            
        return models, vectorizer
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def preprocess_text(text):
    """Clean email text consistently"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

def highlight_keywords(text):
    """Mark suspicious phrases"""
    keywords = ["urgent", "verify", "account", "password", "login", 
               "click", "bank", "suspended", "immediately", "security"]
    for word in keywords:
        if word in text.lower():
            text = text.replace(word, f'<span style="color: #ff4b4b; font-weight: bold;">{word}</span>')
            text = text.replace(word.title(), f'<span style="color: #ff4b4b; font-weight: bold;">{word.title()}</span>')
    return text

def main():
    st.title("🛡️ Phishing Email Detector")
    st.markdown("Analyze emails using multiple machine learning models")
    
    # Load models
    models, vectorizer = load_models()
    
    # Input area
    email_text = st.text_area(
        "Paste email content here:", 
        height=200,
        help="The system will analyze the text for phishing indicators"
    )
    
    # Sample emails
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Phishing Example"):
            st.session_state.email_text = """Urgent: Your account will be suspended!
            
            Dear user,
            
            We detected unusual activity. Verify your identity immediately:
            http://fake-bank.com/secure-login
            
            Click the link within 24 hours to avoid account termination."""
            
    with col2:
        if st.button("Load Safe Example"):
            st.session_state.email_text = """Meeting Reminder
            
            Hi team,
            
            Just a reminder about our quarterly review meeting tomorrow at 10AM.
            Please bring your project updates.
            
            Best regards,
            John Smith
            HR Department"""
    
    if st.button("Analyze Email", type="primary"):
        if not email_text.strip():
            st.warning("Please enter email content to analyze")
            return
            
        try:
            # Preprocess and vectorize
            processed_text = preprocess_text(email_text)
            X = vectorizer.transform([processed_text])
            
            # Display analysis
            st.subheader("Email Analysis")
            st.markdown(highlight_keywords(email_text), unsafe_allow_html=True)
            
            # Model predictions
            st.subheader("Model Predictions")
            cols = st.columns(len(models))
            
            for (name, model), col in zip(models.items(), cols):
                with col:
                    try:
                        pred = model.predict(X)[0]
                        proba = model.predict_proba(X)[0]
                        confidence = max(proba)
                        
                        if pred == "Phishing Email":
                            col.error(f"**{name}**")
                            col.metric(
                                "Result", 
                                "PHISHING", 
                                f"{confidence*100:.1f}% confidence"
                            )
                        else:
                            col.success(f"**{name}**")
                            col.metric(
                                "Result", 
                                "SAFE", 
                                f"{confidence*100:.1f}% confidence"
                            )
                            
                    except Exception as e:
                        col.error(f"{name} failed: {str(e)}")
                        
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()