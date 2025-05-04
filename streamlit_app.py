
# Load the trained model and TF-IDF vectorizer
 # Replace with your vectorizer
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Configure Page ---
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Model & Vectorizer ---
@st.cache_resource  # Cache for performance
def load_assets():
    model = joblib.load('logistic_regression_model.pkl','multinomial_naive_bayes_model.pkl','support_vector_machine_(svm)_model.pkl')  # Replace with your model file
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') 
    return model, vectorizer

model, vectorizer = load_assets()

# --- UI Components ---
# Sidebar (for additional options)
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app detects phishing emails using NLP and **SVM** (97.8% accuracy).  
    [GitHub Repo](#) | [Dataset Info](#)
    """)
    st.divider()
    st.write("**Settings**")
    show_keywords = st.toggle("Highlight suspicious words", True)

# Main Interface
st.title("🛡️ PhishGuard AI")
st.subheader("Paste an email to check for phishing attempts")

# Input Area
user_input = st.text_area(
    "**Email Text**", 
    height=250,
    placeholder="Paste the email content here...",
    help="Example: 'Urgent: Your account will be suspended! Click here to verify...'"
)

# Analyze Button
col1, col2, _ = st.columns([1, 1, 4])
with col1:
    analyze_btn = st.button("Analyze Email", type="primary")
with col2:
    if st.button("Load Sample Email"):
        user_input = """Dear Customer,
        Your bank account has been compromised. 
        Verify your identity immediately: http://fake-bank.com/login
        """

# --- Results Section ---
if analyze_btn and user_input:
    st.divider()
    
    # Preprocess and predict
    processed_text = re.sub(r'[^\w\s]', '', user_input.lower())
    X = vectorizer.transform([processed_text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # Probability of phishing
    
    # Display Results
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            if pred == "Phishing Email":
                st.error("**⚠️ PHISHING DETECTED**")
            else:
                st.success("**✅ SAFE EMAIL**")
        with col2:
            st.metric("Confidence", f"{proba*100:.1f}%")
    
    # Keyword Highlighting
    if show_keywords:
        phishing_keywords = ["urgent", "verify", "account", "password", "click", "bank"]
        highlighted_text = user_input
        for word in phishing_keywords:
            highlighted_text = highlighted_text.replace(word, f"<mark style='background-color: #ffcccc'>{word}</mark>")
        st.markdown("**🔍 Suspicious Keywords**")
        st.markdown(highlighted_text, unsafe_allow_html=True)
    
    # Explanation
    st.divider()
    st.markdown("""
    **How This Works**  
    - The model analyzes word patterns using **TF-IDF** and **SVM**.  
    - Phishing emails often contain urgent language (highlighted above).  
    - Confidence >80% indicates high risk.
    """)

elif analyze_btn:
    st.warning("Please enter an email to analyze.")