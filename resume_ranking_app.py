import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "  # Ensure spacing between pages
    return text.strip() if text else None  # Handle unreadable PDFs

# Function to rank resumes
def rank_resume(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words='english')  # Remove common words
    vectors = vectorizer.fit_transform(documents).toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit App
st.title("📄 AI Resume Screening and Candidate Ranking")

# Job Description Input
st.header("📝 Job Description")
job_description = st.text_area("Enter the Job Description")

# File Uploader
st.header("📂 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    with st.spinner("Processing resumes..."):
        resumes = []
        valid_files = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resumes.append(text)
                valid_files.append(file.name)  # Store only valid filenames
            else:
                st.warning(f"⚠️ Unable to extract text from {file.name}")

        if resumes:
            scores = rank_resume(job_description, resumes)
            results = pd.DataFrame({"Resume": valid_files, "Score": np.round(scores * 100, 2)})
            results = results.sort_values(by="Score", ascending=False)
            st.write(results)
        else:
            st.error("No valid resumes were processed.")

    # Display Scores
    results = pd.DataFrame({"Resume":[file.name for file in uploaded_files], "Score": scores })
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)