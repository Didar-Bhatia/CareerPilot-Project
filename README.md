Resume Analyser 
A Smart Job + Skill Tracker for Students

This project is a web-based application designed to automate the resume screening process. It parses uploaded PDF resumes, evaluates them against specific job descriptions, and recommends suitable job roles based on extracted technical skills. The application provides actionable insights for candidates, including formatting audits, missing keywords, salary estimations, and relevant interview preparation resources.

Technical Stack
* Backend Framework: Flask
* Data Processing: Pandas, NumPy
* Natural Language Processing: Scikit-learn (TfidfVectorizer, Cosine Similarity)
* Document Parsing: PyPDF2

Core Functionalities
1. Automated Text Extraction: Reads and extracts raw text from PDF resumes.
2. Job Description (JD) Matching: Calculates a cosine similarity score between the resume text and a user-provided job description to determine alignment. It also identifies top missing keywords.
3. Skill Extraction & Recommendation: Vectorizes the resume text using TF-IDF and compares it against a predefined matrix of job roles and skills to recommend the top 5 matching positions.
4. Formatting Audit: Scans the document for critical structural elements, such as the presence of an email address, a standard phone number, and optimal word count.
5. Market & Prep Integration: Maps recommended roles to estimated salary ranges and fetches live job postings along with targeted interview preparation links.

Critical Analysis & Architecture Review

Positives (Strengths):
* Low Overhead: Utilizing Scikit-learn's TfidfVectorizer instead of heavy deep-learning models keeps the application lightweight and responsive. 
* Caching Strategy: Loading the job skills matrix, prep links, and salary data into memory during the initial server startup minimizes disk I/O, significantly speeding up the analysis route.
* Comprehensive Feedback Loop: The tool goes beyond simple keyword extraction by providing actionable steps (missing JD keywords, live job links), creating a highly practical tool for job seekers.

Negatives (Limitations):
* Brittle PDF Parsing: The reliance on PyPDF2 is a significant bottleneck. This library frequently struggles with complex, multi-column layouts commonly found in modern resumes, which can result in garbled text extraction and pipeline failure.
* Lack of Semantic Context: The skill extraction logic relies on exact string matching within a static vocabulary. Unlike natural language processing pipelines built with tools like spaCy, the current TF-IDF approach lacks contextual understanding and will fail to recognize multi-word technical skills or synonyms (e.g., treating "ML" and "Machine Learning" as entirely different entities).
* Scalability Concerns: Storing the entire TF-IDF matrix in-memory is viable for a prototype but will cause severe RAM consumption issues if the job dataset scales to an enterprise level. Additionally, the lack of file size limits on the upload route exposes the server to memory overload vulnerabilities.
