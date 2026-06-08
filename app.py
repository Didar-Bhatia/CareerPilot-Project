from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import PyPDF2
import io
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. INITIAL SETUP ---
app = Flask(__name__)

# Load Job Skills
csv_file_path = "skill_job_1757498364347.csv"
try:
    df = pd.read_csv(csv_file_path)
    df['role'] = df['role'].str.strip()
    df['tokenized_skills'] = df['skills'].apply(lambda x: x.lower().split() if isinstance(x, str) else [])
    skills_as_strings = df['tokenized_skills'].apply(lambda x: " ".join(x))
    v = TfidfVectorizer()
    matrix = v.fit_transform(skills_as_strings)
    vocabulary = v.get_feature_names_out()
    print(" Job skills loaded.")
except FileNotFoundError:
    print(f" FATAL: {csv_file_path} not found.")
    vocabulary, df, matrix = [], pd.DataFrame(), None

# Load Prep Links
prep_links_csv_path = "prep_links.csv"
try:
    prep_links_df = pd.read_csv(prep_links_csv_path).set_index('JobRole')
    print(" Prep links loaded.")
except FileNotFoundError:
    print(" WARNING: prep_links.csv not found.")
    prep_links_df = None

# Load Live Jobs
live_jobs_csv_path = "live_jobs.csv"
try:
    live_jobs_df = pd.read_csv(live_jobs_csv_path)
    live_jobs_df['JobRole'] = live_jobs_df['JobRole'].str.strip()
    print(" Live jobs loaded.")
except FileNotFoundError:
    live_jobs_df = pd.DataFrame()

# --- NEW: Load Salary Data ---
salary_csv_path = "salary_data.csv"
try:
    salary_df = pd.read_csv(salary_csv_path)
    salary_df['role'] = salary_df['role'].str.strip()
    # Create a dictionary for faster lookup: {'Data Scientist': '$120k - $160k', ...}
    salary_dict = dict(zip(salary_df['role'], salary_df['salary_range']))
    print(" Salary data loaded.")
except FileNotFoundError:
    print(" WARNING: salary_data.csv not found.")
    salary_dict = {}

# --- 2. HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_file_stream):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file_stream)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def check_formatting(text):
    """Checks for email, phone, and word count."""
    issues = []

    # 1. Email Check
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if not re.search(email_pattern, text):
        issues.append({"type": "critical", "msg": "Critical: No email address found."})

    # 2. Phone Check (Generic 10+ digits lookahead)
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    if not re.search(phone_pattern, text):
        issues.append({"type": "warning", "msg": "Warning: No standard phone number found."})

    # 3. Word Count Check
    word_count = len(text.split())
    if word_count < 200:
        issues.append({"type": "warning", "msg": f"Resume is too short ({word_count} words). Aim for 300+."})
    elif word_count > 2000:
        issues.append({"type": "warning", "msg": f"Resume is too long ({word_count} words). Aim for under 2 pages."})

    return issues

def get_common_words(text):
    """Returns top 10 most frequent words (excluding common stop words)."""
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = set(['the', 'and', 'to', 'of', 'in', 'for', 'with', 'a', 'on', 'is', 'an', 'as', 'are', 'this', 'by', 'it', 'be', 'or', 'at', 'from', 'that', 'which', 'your', 'my', 'i', 'will', 'skills', 'experience', 'work'])
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2 and not w.isdigit()]
    return Counter(filtered_words).most_common(10)

def fetch_live_jobs(job_title):
    if live_jobs_df.empty: return []
    try:
        matched_jobs = live_jobs_df[live_jobs_df['JobRole'] == job_title]
        formatted_jobs = []
        for _, job in matched_jobs.head(5).iterrows():
            formatted_jobs.append({
                "title": job['JobTitle'],
                "company": job['Company'],
                "url": job['ApplyURL']
            })
        return formatted_jobs
    except:
        return []

# --- 3. ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a valid PDF file'}), 400

    if matrix is None or df.empty:
        return jsonify({'error': 'Server is not ready. CSV file is missing.'}), 500

    try:
        # 1. Extract Text
        resume_text = extract_text_from_pdf(io.BytesIO(file.read()))

        # --- NEW: Job Description Matcher ---
        jd_text = request.form.get('job_description', '').strip()
        jd_match_score = None
        jd_missing_keywords = []

        if jd_text:
            # 1. Vectorize both texts (Resume vs JD)
            text_list = [resume_text, jd_text]
            tfidf = TfidfVectorizer().fit_transform(text_list)
            vectors = tfidf.toarray()

            # 2. Calculate Cosine Similarity
            cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            jd_match_score = round(cosine_sim * 100, 2)

            # 3. Find Keywords in JD that are missing in Resume
            jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
            resume_words_lower = set(re.findall(r'\b\w+\b', resume_text.lower()))

            # Filter out common stop words
            stop_words = set(['the', 'and', 'to', 'for', 'with', 'a', 'in', 'of', 'is', 'are', 'on', 'at', 'be', 'this', 'that'])
            missing_in_resume = jd_words - resume_words_lower
            # Get top 10 missing words that are longer than 4 chars
            jd_missing_keywords = [w for w in missing_in_resume if w not in stop_words and len(w) > 4][:10]

        # 2. Formatting Check
        formatting_issues = check_formatting(resume_text)

        # 3. Keyword Analysis
        top_keywords = get_common_words(resume_text)

        # 4. Skills Extraction
        resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
        extracted_skills_set = set([skill for skill in vocabulary if skill in resume_words])

        if not extracted_skills_set:
            return jsonify({'error': 'No relevant skills found in resume.'}), 200

        # 5. Recommendation Logic
        skills_string = " ".join(extracted_skills_set)
        resume_vector = v.transform([skills_string])
        cosine_sim_matrix = cosine_similarity(resume_vector, matrix)
        top_indices = np.argsort(cosine_sim_matrix[0])[::-1][:5]

        recommended_jobs = []
        for i in top_indices:
            score = round(cosine_sim_matrix[0][i] * 100, 2)
            if score > 0:
                role_name = df.iloc[i]['role']
                job_skills_set = set(df.iloc[i]['tokenized_skills'])
                matched_skills = list(extracted_skills_set.intersection(job_skills_set))
                missing_skills = list(job_skills_set.difference(extracted_skills_set))

                # Get Salary
                salary_est = salary_dict.get(role_name, "Market Rate N/A")

                recommended_jobs.append({
                    'role': role_name,
                    'score': score,
                    'matched_skills': matched_skills,
                    'missing_skills': missing_skills,
                    'salary': salary_est
                })

        # 6. Prep & Live Jobs (Top Match Only)
        prep_links_list = []
        live_jobs_list = []
        if recommended_jobs:
            top_role = recommended_jobs[0]['role']
            # Prep
            if prep_links_df is not None:
                try:
                    row = prep_links_df.loc[top_role]
                    prep_links_list = [
                        {'name': row['ResourceName1'], 'url': row['ResourceLink1']},
                        {'name': row['ResourceName2'], 'url': row['ResourceLink2']},
                        {'name': row['ResourceName3'], 'url': row['ResourceLink3']}
                    ]
                    prep_links_list = [l for l in prep_links_list if pd.notna(l['name'])]
                except: pass
            # Live Jobs
            live_jobs_list = fetch_live_jobs(top_role)

        # 6. Return all data
        return jsonify({
            'skills': list(extracted_skills_set),
            'jobs': recommended_jobs,
            'prep_links': prep_links_list,
            'live_jobs': live_jobs_list,
            'formatting_issues': formatting_issues,
            'top_keywords': top_keywords,
            'jd_score': jd_match_score,
            'jd_missing': jd_missing_keywords
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)