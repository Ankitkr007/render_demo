# main.py
import streamlit as st
import chardet
import fitz  # PyMuPDF
import pandas as pd
import traceback
from agents.jd_summarizer import JDSummarizer
from agents.cv_parser import CVParser
from agents.matching_engine import MatchingEngine
from database.db_handler import DBHandler
from agents.email_scheduler import EmailScheduler
import logging

# Filter out the specific RuntimeError warning from Streamlit's watcher
logger = logging.getLogger('streamlit.watcher.local_sources_watcher')
logger.setLevel(logging.ERROR)  # Only show errors, not warnings
# Configure page
st.set_page_config(
    page_title="AI Recruitment System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def read_csv_with_encoding(file_path_or_buffer):
    try:
        if hasattr(file_path_or_buffer, 'read'):
            raw_data = file_path_or_buffer.read()
            result = chardet.detect(raw_data)
            file_path_or_buffer.seek(0)
            return pd.read_csv(file_path_or_buffer, encoding=result['encoding'])
        else:
            with open(file_path_or_buffer, 'rb') as f:
                result = chardet.detect(f.read())
            return pd.read_csv(file_path_or_buffer, encoding=result['encoding'])
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path_or_buffer, encoding='latin1')
        except Exception as e:
            st.error(f"Failed to read file: {str(e)}")
            st.stop()

def handle_job_loading():
    if 'use_sample' not in st.session_state:
        st.session_state.use_sample = False
        
    st.session_state.use_sample = st.toggle("Use Sample Jobs", value=st.session_state.use_sample)
    
    if st.session_state.use_sample:
        try:
            return read_csv_with_encoding("jobs.csv")
        except FileNotFoundError:
            st.error("Sample jobs.csv not found in project directory")
            sample_df = pd.DataFrame({
                'Job Title': ['Software Engineer', 'Data Scientist', 'Product Manager'],
                'Job Description': [
                    """We are looking for a Software Engineer with Python, Java, and C++ experience. Responsibilities include developing applications, writing clean code, and troubleshooting issues. Bachelor's degree in Computer Science required.""",
                    """Data Scientist position requiring machine learning, statistics, and Python skills. Analyze data to provide business insights. Experience with NLP and deep learning a plus. Master's degree preferred.""",
                    """Product Manager to lead product development. Define product vision, work with engineering teams, and analyze market trends. 3+ years experience in tech product management required."""
                ]
            })
            sample_df.to_csv("jobs.csv", index=False)
            return sample_df
    else:
        job_file = st.file_uploader("Upload Job Descriptions (CSV)", type=["csv"])
        if not job_file:
            st.info("Please upload a CSV file with job descriptions or toggle 'Use Sample Jobs'")
            st.stop()
        return read_csv_with_encoding(job_file)

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = " ".join([page.get_text() for page in doc]).strip()
        uploaded_file.seek(0)
        return text
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
        print(f"PDF extraction error: {traceback.format_exc()}")
        return ""



@st.cache_resource
def load_models():
    try:
        with st.spinner("Loading language models..."):
            # No need to load SentenceTransformer anymore
            matching_engine = MatchingEngine(model_name="nomic-embed-text")
            return matching_engine, JDSummarizer(), CVParser(), matching_engine
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        st.stop()


def display_json_as_table(json_data, title=None):
    if title:
        st.subheader(title)
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{key.replace('_', ' ').title()}:**")
            with col2:
                if isinstance(value, list):
                    if value:
                        st.markdown("<br>".join([f"• {item}" for item in value]), unsafe_allow_html=True)
                    else:
                        st.markdown("*None specified*")
                elif isinstance(value, dict):
                    st.json(value)
                elif value is None:
                    st.markdown("*None specified*")
                else:
                    st.markdown(str(value))
    else:
        st.json(json_data)

# Main App
def main():
    st.title("AI Recruitment System 🚀")
    db = DBHandler()

    with st.sidebar:
        st.subheader("About AI Recruiter")
        st.write("""
        This system uses AI to automate the recruitment process:
        1. Job Description Analysis Agent
        2. CV Parsing Agent
        3. Matching Agent
        4. Email Scheduling Agent
        """)
        if st.checkbox("Check Email Configuration"):
            scheduler = EmailScheduler()
            status = scheduler.test_email_connection()
            st.info(status)
            if "not set" in status:
                st.code("EMAIL_USER=your-email@gmail.com\nEMAIL_PASSWORD=your-app-password")

    # Job Uploading
    try:
        job_df = handle_job_loading()
    except Exception as e:
        st.error(f"Job Loading Error: {str(e)}")
        st.stop()

    if 'Job Title' not in job_df.columns or 'Job Description' not in job_df.columns:
        st.error("CSV must contain 'Job Title' and 'Job Description' columns")
        st.stop()

    selected_job = st.selectbox("Select Job Title", job_df['Job Title'].unique())
    selected_jd = job_df[job_df['Job Title'] == selected_job].iloc[0]['Job Description']

    with st.expander("Show Full Job Description"):
        st.write(selected_jd)

    uploaded_files = st.file_uploader("Upload Candidate CVs (PDF)", type=["pdf"], accept_multiple_files=True)
    process_btn = st.button("Process Applications", type="primary", disabled=len(uploaded_files) == 0)

    if process_btn:
        matching_engine, jd_summarizer, cv_parser, _ = load_models()

        jd_summary = jd_summarizer.summarize(selected_jd)
        display_json_as_table(jd_summary, "Job Summary")
        jd_embedding = matching_engine.get_embedding(selected_jd)
        job_id = db.create_job(
            title=selected_job,
            raw_description=selected_jd,
            summary=jd_summary,
            embedding=jd_embedding
        )

        candidates = []
        progress_bar = st.progress(0, text="Processing CVs...")

        for i, file in enumerate(uploaded_files):
            progress_bar.progress(i / len(uploaded_files), text=f"Processing ({i+1}/{len(uploaded_files)})")
            cv_text = extract_text_from_pdf(file)
            if not cv_text:
                continue
            cv_data = cv_parser.parse(cv_text)
            cv_embedding = matching_engine.get_embedding(cv_text)
            score = matching_engine.calculate_match(jd_embedding, cv_embedding)
            candidate_id = db.create_candidate(
                job_id=job_id,
                cv_text=cv_text,
                cv_data=cv_data,
                embedding=cv_embedding,
                score=score
            )
            candidates.append({**cv_data, "score": score, "id": candidate_id, "filename": file.name})

        progress_bar.progress(1.0, text="Processing complete!")

        # Store in session state for persistence
        st.session_state["candidates"] = candidates
        st.session_state["selected_job"] = selected_job

        # Prepare shortlist
        shortlisted = [c for c in candidates if c["score"] >= 65]
        st.session_state["shortlisted"] = shortlisted

    # Show results if candidates are available
    if "candidates" in st.session_state and st.session_state["candidates"]:
        candidates = st.session_state["candidates"]

        st.header("Candidate Rankings")
        candidates_df = pd.DataFrame(candidates).sort_values("score", ascending=False)
        if 'score' in candidates_df.columns:
            candidates_df['score'] = candidates_df['score'].apply(lambda x: f"{x:.2f}%")
        display_cols = ["name", "email", "score", "filename"]
        available_cols = [col for col in display_cols if col in candidates_df.columns]
        st.dataframe(candidates_df[available_cols], use_container_width=True)

        if st.checkbox("Show Detailed Profiles"):
            for candidate in candidates:
                with st.expander(f"{candidate.get('name', 'Unknown')} - {candidate.get('filename', '')}"):
                    display_data = {k: v for k, v in candidate.items() if k not in ['id', 'filename', 'score']}
                    st.markdown(f"**Match Score**: {float(candidate.get('score', 0)):.2f}%")
                    display_json_as_table(display_data)

        shortlisted = st.session_state.get("shortlisted", [])
        if shortlisted:
            st.success(f"✅ Shortlisted {len(shortlisted)} candidates")
            st.subheader("Shortlisted Candidates")
            shortlisted_df = pd.DataFrame(shortlisted)
            if 'score' in shortlisted_df.columns:
                shortlisted_df['score'] = shortlisted_df['score'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(shortlisted_df[available_cols], use_container_width=True)

    # Email Button Section
    if st.button("Send Interview Invites"):
        scheduler = EmailScheduler()
        status = scheduler.test_email_connection()

        shortlisted = st.session_state.get("shortlisted", [])
        selected_job = st.session_state.get("selected_job", "Job")

        if not shortlisted:
            st.warning("No shortlisted candidates found. Please process applications first.")
        elif "successful" in status:
            for candidate in shortlisted:
                email = candidate.get("email", "unknown@example.com")
                name = candidate.get("name", "Candidate")

                success, message = scheduler.send_interview_invite(email, name, selected_job)
                if success:
                    db.create_email(
                        candidate_id=candidate["id"],
                        content=f"Interview invite sent for {selected_job}"
                    )
                    st.success(f"Email sent to {name} at {email}")
                else:
                    st.error(f"Failed to send email to {email}: {message}")
        else:
            # Simulation mode
            st.warning("Email credentials not configured. Running in simulation mode.")
            st.info("In production, emails would be sent to:")
            for candidate in shortlisted:
                email = candidate.get("email", "unknown@example.com")
                name = candidate.get("name", "Candidate")
                st.markdown(f"- {name}: {email}")


if __name__ == "__main__":
    app()
