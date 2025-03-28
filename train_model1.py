import os
import joblib
import pandas as pd
import spacy
import re
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Define paths
DESKTOP_DIR = r"C:\\Users\\ART\\Desktop"
TEMPLATES_DIR = os.path.join(DESKTOP_DIR, "templates")
TRAINING_CSV_PATH = os.path.join(TEMPLATES_DIR, "training.csv")
MODEL_PATH = os.path.join(DESKTOP_DIR, "random_forest_model.pkl")
VECTORIZER_PATH = os.path.join(DESKTOP_DIR, "tfidf_vectorizer.pkl")

# Ensure directories exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)

def extract_info(text):
    doc = nlp(text)
    skills, education, certifications = set(), set(), set()
    
    # Skill Extraction using Named Entity Recognition
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL"]:
            skills.add(ent.text.lower())
    
    # Education Extraction
    education_patterns = re.findall(r"\b(BSc|MSc|PhD|Bachelor|Master|Doctorate|MBA)\b", text, re.IGNORECASE)
    education.update(education_patterns)
    
    # Certification Extraction
    cert_patterns = re.findall(r"\b(Certified|Certificate|Accredited|License)\b", text, re.IGNORECASE)
    certifications.update(cert_patterns)
    
    return {"skills": list(skills), "education": list(education), "certifications": list(certifications)}

def train_model():
    try:
        logging.debug("📂 Loading data")
        if not os.path.exists(TRAINING_CSV_PATH):
            raise FileNotFoundError(f"❌ No such file: '{TRAINING_CSV_PATH}'")
        
        df = pd.read_csv(TRAINING_CSV_PATH)
        required_columns = {"resume_text", "label"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"❌ Missing required columns: {required_columns - set(df.columns)}")
        
        df.dropna(subset=["resume_text", "label"], inplace=True)
        if df.empty:
            raise ValueError("❌ Training data is empty after removing missing values.")
        
        # Extract NLP features (Optional)
        df["nlp_features"] = df["resume_text"].apply(lambda text: extract_info(text))
        
        # Train TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3), stop_words='english', min_df=2, max_df=0.9)
        X = vectorizer.fit_transform(df["resume_text"])
        y = df["label"]
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)
        
        logging.debug("🛠️ Training Random Forest model")
        model.fit(X_train, y_train)
        
        # Test Set Predictions
        y_pred = model.predict(X_test)
        
        # Save Model and Vectorizer
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        logging.debug(f"✅ Model saved at {MODEL_PATH}")
        logging.debug(f"✅ Vectorizer saved at {VECTORIZER_PATH}")
        
    except Exception as e:
        logging.error(f"❌ Training failed: {str(e)}")

if __name__ == "__main__":
    train_model()
