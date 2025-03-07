import streamlit as st
from PyPDF2 import PdfReader 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 


#function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text+=page.extract_text()
    return text


def rank_resume(job_description, resumes):
    #combine job description with resumes
    documents = [job_description] + resumes 
    vectorizer = TfidfVectorizer().fit_transform(documents) 
    vectors = vectorizer.toarray() # vectors is assigned here


    # calculate cosine similarity
    job_description_vector = vectors[0] # Now accessible within the function
    resume_vectors = vectors [1:] # Now accessible within the function
    cosine_similarities = cosine_similarity ([job_description_vector], resume_vectors).flatten()

    return cosine_similarities # Return the result


# streamlit app
st.title("AI Resume Screening and Candidate Ranking system")


#Job Description input
st.header("Job Description")
job_description = st.text_area ("Enter the Job Description")


#File Upoader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files = True)


if uploaded_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for files in uploaded_files:
        text = extract_text_from_pdf(files)
        resumes.append(text)


    # rank resumes
    scores = rank_resume(job_description, resumes)


    # Display Scores
    results = pd.DataFrame({"Resume":[file.name for file in uploaded_files], "Score": scores })
    results = results.sort_values(by="Score", ascending=False)

    st.write(results)