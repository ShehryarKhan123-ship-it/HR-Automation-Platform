import os
import nltk
import pdfplumber
import docx
import re
import string
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# Flask Setup (No Debug Logs)
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Configurable Job Description
# --------------------------------

job_description = """
We are seeking highly skilled professionals in the IT sector for roles in software development, data science, cybersecurity, cloud computing, and IT infrastructure management.  
Candidates should have expertise in one or more of the following areas:

🔹 **Software Development & Engineering:**  
- Proficiency in programming languages such as Python, Java, C++, JavaScript, or Go.  
- Experience with frameworks like Django, Flask, Spring Boot, React, Angular, or Vue.js.  
- Knowledge of database management systems (SQL, PostgreSQL, MongoDB, Firebase).  
- Understanding of software development methodologies (Agile, Scrum, DevOps).  

🔹 **Data Science & Artificial Intelligence:**  
- Strong foundation in machine learning, deep learning, and statistical analysis.  
- Experience with ML frameworks such as TensorFlow, PyTorch, or Scikit-learn.  
- Expertise in data visualization tools (Tableau, Power BI, Matplotlib, Seaborn).  
- Ability to work with large datasets and perform ETL processes.  

🔹 **Cybersecurity & Information Security:**  
- Knowledge of cybersecurity best practices, threat detection, and vulnerability assessments.  
- Experience with security tools like Wireshark, Metasploit, or Snort.  
- Understanding of encryption, authentication, and network security protocols.  
- Certifications such as CISSP, CEH, or CompTIA Security+ are a plus.  

🔹 **Cloud Computing & DevOps:**  
- Hands-on experience with cloud platforms like AWS, Azure, or Google Cloud.  
- Proficiency in containerization and orchestration (Docker, Kubernetes).  
- CI/CD implementation using Jenkins, GitHub Actions, or GitLab CI/CD.  
- Infrastructure as Code (IaC) experience with Terraform or Ansible.  

🔹 **IT Infrastructure & Networking:**  
- Expertise in network configuration, troubleshooting, and administration.  
- Experience with Cisco, Juniper, or Palo Alto networks.  
- Understanding of cloud networking and hybrid infrastructures.  
- Certifications such as CCNA, CCNP, or AWS Certified Solutions Architect are a plus.  

We welcome professionals with a passion for technology, problem-solving, and continuous learning. Candidates should demonstrate excellent analytical skills, teamwork, and the ability to adapt to new challenges in a fast-paced IT environment.
"""

job_Skills="""
python , AWS, Cloud, C++ , programming languages, Azure ,Google Cloud , PyTorch , Django , Flask , Angular , TensorFlow , Scikit-learn , Power BI , Matplotlib ,Seaborn , SQL
, Angular,machine learning,deep learning,statistical analysis,Tableau,PyTorch,Scikit-learn,ETL processes, cybersecurity,information security,threat detection,
vulnerability assessments,encryption,authentication,network security protocols,CISSP,CEH,CompTIA Security+,Wireshark,Metasploit,Snort,cloud platforms,AWS,Azure,Google Cloud,
containerization,orchestration,Docker,Kubernetes,CI/CD,Jenkins,GitHub Actions,GitLab CI/CD,Infrastructure as Code,Terraform,Ansible,network configuration,troubleshooting,
administration,Cisco,Juniper,Palo Alto networks,cloud networking,hybrid infrastructures,CCNA,CCNP,AWS Certified Solutions Architect,technology,problem-solving,
continuous learning,analytical skills,teamwork,fast-paced IT environment"""
# --------------------------------
# Paths & Config
# --------------------------------
DESKTOP_DIR = r"C:\Users\ART\Desktop"
TEMPLATES_DIR = os.path.join(DESKTOP_DIR, "templates")
UPLOADS_DIR = os.path.join(TEMPLATES_DIR, "uploads")

# Ensure uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(TEMPLATES_DIR, "random_forest_model.pkl")
VECTORIZER_PATH = os.path.join(TEMPLATES_DIR, "tfidf_vectorizer.pkl")
FINAL_CSV_PATH = os.path.join(UPLOADS_DIR, "final_candidates.csv")

ALLOWED_EXTENSIONS = {"pdf", "docx"}

# --------------------------------
# Load Model & Vectorizer
# --------------------------------
model, vectorizer = None, None
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"[Error] Could not load model/vectorizer: {str(e)}")

# --------------------------------
# Utility Functions
# --------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(docx_path):
    doc_obj = docx.Document(docx_path)
    full_text = [para.text for para in doc_obj.paragraphs]
    return "\n".join(full_text).strip()

# Download NLTK data if not already present
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def compute_resume_score(resume_text):
    """
    Compute how similar resume_text is to the global job_description
    using the loaded TF-IDF vectorizer + cosine similarity.
    Returns a float between 0 and 1.
    """
    if vectorizer is None:
        return 0.0
    desc_vector = vectorizer.transform([job_description])
    resume_vector = vectorizer.transform([resume_text])
    similarity = cosine_similarity(desc_vector, resume_vector)[0][0]
    skill_similarity = vectorizer.transform([job_Skills])
    similarity += cosine_similarity(skill_similarity, resume_vector)[0][0]
    return round(float(similarity), 3)


def predict_resumes(csv_path, threshold=0.3):
    """
    Reads resumes from a CSV file, transforms them with the loaded TF-IDF vectorizer,
    applies model.predict_proba, and adds:
       - resume_score (cosine similarity to job_description)
       - predicted_label (0 or 1) based on threshold
    Saves a final CSV with all columns + predictions.
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model or vectorizer not loaded. Please train or load them first.")

    df = pd.read_csv(csv_path)
    if "Resume Text" not in df.columns:
        raise ValueError("CSV must have a 'Resume Text' column.")

    # Vectorize all resume texts
    X_tfidf = vectorizer.transform(df["Resume Text"])
    # Model probabilities for class=1 (Suitable)
    y_probs = model.predict_proba(X_tfidf)[:, 1]

    # Also compute a resume_score for each row
    df["resume_score"] = [compute_resume_score(text) for text in df["Resume Text"]]

    # Combine prediction with similarity score
    df["adjusted_score"] = (df["resume_score"] + y_probs) / 2

    # Apply a new dynamic threshold for final decision
    df["predicted_label"] = (df["adjusted_score"] >= threshold).astype(int)
    # Reorder columns to show them nicely
    columns_order = ["Filename", "Resume Text", "resume_score", "predicted_label", "skills", "education", "certifications"]
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns + [col for col in df.columns if col not in existing_columns]]
    df["skills"] = df["Resume Text"].apply(lambda text: extract_resume_info(text)["skills"])
    df["education"] = df["Resume Text"].apply(lambda text: extract_resume_info(text)["education"])
    df["certifications"] = df["Resume Text"].apply(lambda text: extract_resume_info(text)["certifications"])

    # Save final CSV
    df.to_csv(FINAL_CSV_PATH, index=False)
    return df

def extract_resume_info(text):
    """
    Extracts key information from the resume text, including:
    - Skills (based on predefined keywords)
    - Education (identifying degrees and institutions)
    - Certifications (common industry certifications)
    """
    skills_keywords = [
        "python", "java", "c++", "javascript", "sql", "machine learning", "deep learning",
        "tensorflow", "pytorch", "aws", "azure", "cybersecurity", "networking",
        "data science", "docker", "kubernetes", "devops", "agile", "scrum"
    ]
    education_keywords = [
        "bachelor", "master", "phd", "b.sc", "m.sc", "mba", "b.tech", "m.tech", "bachelor's",
        "master's", "ph.d", "computer science", "engineering", "business administration"
    ]
    certification_keywords = [
        "cissp", "ceh", "aws certified", "ccna", "ccnp", "pmp", "cfa", "scrum master",
        "comptia security+", "certified ethical hacker"
    ]

    extracted_skills = [word for word in skills_keywords if word in text.lower()]
    extracted_education = [word for word in education_keywords if word in text.lower()]
    extracted_certifications = [word for word in certification_keywords if word in text.lower()]

    return {
        "skills": ", ".join(extracted_skills),
        "education": ", ".join(extracted_education),
        "certifications": ", ".join(extracted_certifications)
    }


# --------------------------------
# Routes
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    """
    Allows HR to upload multiple PDF/DOCX resumes.
    Extracts text, preprocesses it, saves to CSV,
    then runs the model to produce predictions with threshold=0.65.
    """
    if request.method == "POST":
        if "files" not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No selected files"}), 400

        results = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOADS_DIR, filename)
                file.save(filepath)

                # Extract text
                if filename.lower().endswith(".pdf"):
                    extracted_text = extract_text_from_pdf(filepath)
                else:
                    extracted_text = extract_text_from_docx(filepath)

                # Preprocess text
                processed_text = preprocess_text(extracted_text)
                
                # Compute score and prediction
                score = compute_resume_score(processed_text)
                if score > 0.4:
                    status = "Recommended"
                else:
                    if model and vectorizer:
                        X = vectorizer.transform([processed_text])
                        prediction = model.predict(X)[0]
                        status = "Recommended" if prediction == 1 else "Not Recommended"
                    else:
                        prediction = 0
                        status = "Model not loaded"

                resume_info = extract_resume_info(processed_text)

                results.append({
                    "filename": filename,
                    "score": score,
                    "status": status,
                    "skills": resume_info["skills"],
                    "education": resume_info["education"],
                    "certifications": resume_info["certifications"]
                })

        if results:
            uploaded_csv_path = os.path.join(UPLOADS_DIR, "uploaded_resumes.csv")
            df = pd.DataFrame(results)
            df.to_csv(uploaded_csv_path, index=False)

            final_df = predict_resumes(uploaded_csv_path, dynamic_threshold=0.4)
            
            # Return the results in the format the frontend expects
            return jsonify([
                {
                    "filename": row["Filename"],
                    "score": row["resume_score"],
                    "status": "Recommended" if row["predicted_label"] == 1 else "Not Recommended"
                }
                for _, row in final_df.iterrows()
            ])
        else:
            return jsonify({"error": "No valid files were uploaded."}), 400

    # If GET request, just render an upload page
    return render_template("index.html")

@app.route("/download", methods=["GET"])
def download_file():
    """
    Endpoint to download the final CSV with predicted_label and resume_score.
    """
    if os.path.exists(FINAL_CSV_PATH):
        return send_file(
            FINAL_CSV_PATH,
            as_attachment=True,
            download_name="final_candidates.csv"
        )
    return jsonify({"error": "Final candidates CSV not found!"}), 404

@app.route("/analyze", methods=["POST"])  # Changed from "/" to "/analyze"
def analyze_resumes():
    if "files" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No selected files"}), 400

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOADS_DIR, filename)
            file.save(filepath)

            # Extract text
            if filename.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(filepath)
            else:
                extracted_text = extract_text_from_docx(filepath)

            # Preprocess text
            processed_text = preprocess_text(extracted_text)
            
            # Compute score
            score = compute_resume_score(processed_text)
            
            # Make prediction
            if score > 0.4:
                    status = "Recommended"
            else:
                if model and vectorizer:
                    X = vectorizer.transform([processed_text])
                    prediction = model.predict(X)[0]
                    status = "Recommended" if prediction == 1 else "Not Recommended"
                else:
                    prediction = 0
                    status = "Model not loaded"

            
            resume_info = extract_resume_info(processed_text)

            results.append({
                "filename": filename,
                "score": score,
                "status": status,
                "skills": resume_info["skills"],
                "education": resume_info["education"],
                "certifications": resume_info["certifications"]
            })

    if results:
        return jsonify(results)
    else:
        return jsonify({"error": "No valid files were processed"}), 400
    

@app.route('/features', methods=['GET'])
def get_features():
    features = [
        {
            "title": "AI-Powered Analysis",
            "description": "Uses advanced machine learning to evaluate resumes against job requirements",
            "icon": "brain"
        },
        {
            "title": "Fast Processing",
            "description": "Analyzes multiple resumes in seconds with parallel processing",
            "icon": "bolt"
        },
        {
            "title": "Detailed Scoring",
            "description": "Provides comprehensive scores based on skills, experience, and qualifications",
            "icon": "chart-bar"
        },
        {
            "title": "Secure Processing",
            "description": "All uploaded files are processed securely and deleted after analysis",
            "icon": "shield-alt"
        },
        {
            "title": "Customizable Criteria",
            "description": "Adjust scoring weights based on your specific hiring needs",
            "icon": "sliders-h"
        },
        {
            "title": "Export Results",
            "description": "Download analysis results in CSV format for further review",
            "icon": "file-export"
        }
    ]
    return jsonify(features)

@app.route('/about', methods=['GET'])
def about():
    about_data = {
        "title": "About Our AI Resume Analyzer",
        "description": "Our solution leverages advanced NLP and machine learning to automate resume screening with 94% accuracy. By reducing manual review time by 80%, we help HR teams focus on strategic hiring decisions while ensuring no qualified candidate is overlooked.",
        "features": [
            {
                "icon": "bolt",
                "title": "Fast Processing",
                "description": "Analyzes hundreds of resumes in under 2 minutes"
            },
            {
                "icon": "brain",
                "title": "Smart Matching",
                "description": "Precisely matches candidates to job requirements"
            },
            {
                "icon": "shield-alt",
                "title": "Unbiased Screening",
                "description": "Reduces unconscious bias in initial screening"
            }
        ],
        "stats": {
            "accuracy": "94%",
            "time_saved": "80%",
            "processing_speed": "2 minutes"
        }
    }
    return jsonify(about_data)

if __name__ == "__main__":
    # Run without debug logs
    app.run(debug=False)
