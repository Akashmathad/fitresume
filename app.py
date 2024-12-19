import streamlit as st
import google.generativeai as genai
import fitz  
import os
import io
import zipfile
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  

def get_gemini_response(input_text, pdf_text, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_text, pdf_text, prompt])
    return response.text


def clean_job_description(job_description):
    return ' '.join([line.strip() for line in job_description.splitlines() if line.strip()])


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        combined_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            combined_text += page.get_text("text")
        pdf_document.close()
        return combined_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def semantic_match(job_description, resumes_text):
    job_embedding = semantic_model.encode(job_description, convert_to_tensor=True)
    results = []
    for resume in resumes_text:
        resume_embedding = semantic_model.encode(resume['text'], convert_to_tensor=True)
        similarity = util.cos_sim(job_embedding, resume_embedding).item()
        results.append({
            "file_name": resume['file_name'],
            "similarity": round((similarity ** 0.5) * 100, 2),  # Scale score for readability
            "text": resume['text'],
            "file_data": resume['file_data']
        })
    return sorted(results, key=lambda x: x['similarity'], reverse=True)


st.set_page_config(page_title="Resume Matching and Summary System", layout="wide")


st.markdown('<h1 class="main-title">FIT RESUME</h1>', unsafe_allow_html=True)
st.markdown("<div class='centered'>", unsafe_allow_html=True)


input_text = st.text_area("📝 Enter your Job Description text here", key="input")
uploaded_files = st.file_uploader(
    "📂 Upload Resumes (Multiple PDFs allowed)", type="pdf", accept_multiple_files=True
)
num_resumes = st.number_input("🔢 Number of Top Resumes to Return", min_value=1, value=5, step=1)
st.markdown('</div>', unsafe_allow_html=True)


submit = st.button("📊 Find Match and Summarize")
submit_separate = st.button("📁 Classify Resumes by Domain")
input_prompt = """
You are an ATS specialist skilled in resume analysis and keyword optimization. Evaluate the resume against the job description.  
1st line: experience level (Fresher, Junior, Mid, Senior) Ex: Experience Level: Junior .  
2nd line: Provide a brief summary (100-150 words). Focus on experience and projects if the candidate is experienced. For freshers, emphasize projects, certifications, and skills. If the resume does not match the job description, respond only with "Mismatched."
"""

if submit:
    if uploaded_files and input_text.strip():
        with st.spinner("Processing resumes..."):
          
            cleaned_description = clean_job_description(input_text)

            
            resumes_text = []
            for file in uploaded_files:
                file.seek(0)
                text = extract_text_from_pdf(file)
                if text:
                    resumes_text.append({
                        "file_name": file.name,
                        "file_data": file.read(),
                        "text": text
                    })

           
            matched_resumes = semantic_match(cleaned_description, resumes_text)
            top_resumes = matched_resumes[:num_resumes]

           
            results = []
            for resume in top_resumes:
                summary = get_gemini_response(cleaned_description, resume["text"], input_prompt)
                results.append({
                    "file_name": resume["file_name"],
                    "summary": summary,
                    "similarity": resume["similarity"],
                    "file_data": resume["file_data"]
                })

           
            st.markdown("<h2 class='sub-header'>📊 Top Matching Resumes</h2>", unsafe_allow_html=True)
            matched_files = []
            for result in results:
                st.markdown(f"### {result['file_name']} | {result['similarity']}% Match")
                st.write(result['summary'])
                st.download_button(
                    label="📥 Download Resume",
                    data=result["file_data"],
                    file_name=result["file_name"],
                    mime="application/pdf",
                )
                matched_files.append((result["file_name"], result["file_data"]))

            
            if matched_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for file_name, file_data in matched_files:
                        zip_file.writestr(file_name, file_data)
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download All Matched Resumes",
                    data=zip_buffer,
                    file_name="matched_resumes.zip",
                    mime="application/zip",
                )
    else:
        st.error("📂 Please upload resumes and provide a job description to proceed.")

def classify_resume_domain(resume_text):
    """
    Use Google Gemini API to classify a resume's domain.
    Example domains: AI/ML Engineer, Data Scientist, Web Developer, etc.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Classify the following resume into its respective domain:Software Engineer, Full Stack Developer, Frontend Developer, Backend Developer, Android Developer, iOS Developer, DevOps Engineer, Data Scientist, Machine Learning Engineer, Data Analyst, Cloud Engineer, Database Administrator, UI/UX Designer, Security Engineer, Game Developer, Embedded Systems Engineer, Software Architect, QA Engineer, Systems Administrator, Network Engineer, Business Intelligence Developer, Java developer and just give me domain name without any explanation and it should be 100% accurate.:\n\n{resume_text}\n\nDomain:"
    try:
        response = model.generate_content([prompt])
        domain = response.text.strip().split("\n")[0]  
        return domain
    except Exception as e:
        return "General" 



if submit_separate:
    if uploaded_files:
        st.write(f"✅ Classifying resumes into domains...")
        domain_classification = {}

        with st.spinner("Classifying resumes by domain..."):
            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file) 
                if resume_text: 
                    try:
                       
                        domain = classify_resume_domain(resume_text)

                        
                        valid_domains = [
                            "Software Engineer", "Full Stack Developer", "Frontend Developer", "Backend Developer", 
                            "Android Developer", "iOS Developer", "DevOps Engineer", "Data Scientist", 
                            "Machine Learning Engineer", "Data Analyst", "Cloud Engineer", "Database Administrator", 
                            "UI/UX Designer", "Security Engineer", "Game Developer", "Embedded Systems Engineer", 
                            "Software Architect", "QA Engineer", "Systems Administrator", "Network Engineer", 
                            "Business Intelligence Developer", "Java Developer"
                        ]
                        if domain not in valid_domains:
                            domain = "General"

                        
                        if domain not in domain_classification:
                            domain_classification[domain] = []
                        domain_classification[domain].append((file.name, file.getvalue()))
                    except Exception as e:
                        st.error(f"Error classifying resume {file.name}: {e}")
                        continue

       
        st.markdown('<h2 class="sub-header">📁 Classified Resumes by Domain</h2>', unsafe_allow_html=True)
        for domain, files in domain_classification.items():
            st.markdown(f"### {domain}")
            for file_name, file_data in files:
                st.write(f"📄 {file_name}")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file_name, file_data in files:
                    zip_file.writestr(file_name, file_data)
            zip_buffer.seek(0)

            st.download_button(
                label=f"📥 Download {domain} Resumes",
                data=zip_buffer,
                file_name=f"{domain}_resumes.zip",
                mime="application/zip",
            )
