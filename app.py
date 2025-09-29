import streamlit as st
import re
import docx2txt
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """Lowercase, remove non-alphanumerics, and trim."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # replace with space (not delete)
    return text.strip()

st.title("📄 Resume Screening App")
st.write("Upload your resume and paste a job description to check match percentage.")

resume_file = st.file_uploader("Upload your Resume (docx/txt)", type=["docx", "txt"])
job_desc = st.text_area("Paste Job Description here")

if st.button("Check Match"):
    if not resume_file:
        st.warning("Please upload a resume file.")
    elif not job_desc.strip():
        st.warning("Please paste a job description.")
    else:
        try:
            resume_text = ""

            if resume_file.name.lower().endswith(".docx"):
                # Save temp file and process
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(resume_file.read())
                    tmp_path = tmp.name
                try:
                    resume_text = docx2txt.process(tmp_path) or ""
                finally:
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

                # Fallback: if still empty, read raw text
                if not resume_text.strip():
                    st.info("⚠ docx2txt gave empty result, trying raw read...")
                    raw = resume_file.getvalue()
                    resume_text = raw.decode("utf-8", errors="ignore")

            else:  # TXT FILE
                raw = resume_file.read()
                try:
                    resume_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    resume_text = raw.decode("latin-1", errors="ignore")

            # Debug: show first 200 chars of raw resume
            st.text_area("🔍 Extracted Resume Text (before cleaning):", resume_text[:500], height=150)

            # Clean texts
            resume_text = clean_text(resume_text)
            jd_text = clean_text(job_desc)

            if not resume_text:
                st.warning("Resume appears empty after cleaning. Please check the file contents/format.")
            elif not jd_text:
                st.warning("Job description appears empty after cleaning.")
            else:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform([resume_text, jd_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                st.success(f"✅ Resume Match: {round(similarity * 100, 2)}%")

        except Exception as e:
            st.error(f"Error processing file: {e}")
