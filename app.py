import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import google.generativeai as genai
from PIL import Image
import requests
import io
import base64
import time
import functools
from googleapiclient.discovery import build
import pytesseract
from concurrent.futures import ThreadPoolExecutor

# Set page config
st.set_page_config(
    page_title="InterviewMaster AI - Professional Interview Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# INTERVIEW ROUNDS AND DIFFICULTY LEVELS
# =========================================================================

INTERVIEW_ROUNDS = {
    "Technical Round": [
        "Data Structures",
        "Algorithms",
        "Object-Oriented Programming",
        "System Design",
        "Database Management",
        "Operating Systems",
        "Computer Networks",
        "Web Development",
        "Mobile Development",
        "Cloud Computing",
        "Distributed Systems",
        "Software Engineering"
    ],
    "Coding Round": [
        "Python Coding",
        "JavaScript Coding",
        "Java Coding",
        "C++ Coding",
        "SQL Queries",
        "Algorithm Implementation",
        "Data Structure Implementation",
        "API Development",
        "Frontend Coding",
        "Backend Coding",
        "Full Stack Coding",
        "React Coding"
    ],
    "HR/Behavioral Round": [
        "Leadership Experience",
        "Team Collaboration",
        "Conflict Resolution",
        "Problem Solving",
        "Time Management",
        "Work Ethics",
        "Career Goals",
        "Company Culture Fit",
        "Strengths and Weaknesses",
        "Past Projects",
        "Adaptability",
        "Communication Skills"
    ],
    "Concept Round": [
        "Data Science Concepts",
        "Machine Learning",
        "Artificial Intelligence",
        "DevOps Principles",
        "Cybersecurity Basics",
        "Computer Science Fundamentals",
        "Software Development Lifecycle",
        "Testing Methodologies",
        "Frontend Technologies",
        "Backend Technologies",
        "Database Concepts",
        "API Design Principles"
    ],
    "Aptitude Round": [
        "Logical Reasoning",
        "Numerical Ability",
        "Verbal Reasoning",
        "Data Interpretation",
        "Analytical Thinking",
        "Quantitative Aptitude",
        "Spatial Reasoning",
        "Mathematical Puzzles",
        "Critical Reasoning",
        "Probability and Statistics"
    ],
    "Algorithm Coding Round": [
        "Array Manipulation",
        "String Processing",
        "Linked Lists",
        "Tree Traversal",
        "Graph Algorithms",
        "Dynamic Programming",
        "Sorting Algorithms",
        "Searching Algorithms",
        "Recursion Problems",
        "Bit Manipulation",
        "Greedy Algorithms",
        "Hash Table Implementation"
    ]
}

DIFFICULTY_LEVELS = [
    "Easy",
    "Medium",
    "Hard"
]

# =========================================================================
# CACHE FUNCTIONS FOR PERFORMANCE
# =========================================================================

# Cache for API results to improve performance
@st.cache_data(ttl=3600)
def cached_generate_interview_questions(round_type, topic, difficulty, num_questions=5):
    """Cached version of generate_interview_questions"""
    return generate_interview_questions(round_type, topic, difficulty, num_questions)

@st.cache_data(ttl=3600)
def cached_evaluate_answer(question, answer, round_type, topic, difficulty):
    """Cached version of evaluate_answer"""
    return evaluate_answer(question, answer, round_type, topic, difficulty)

@st.cache_data(ttl=3600)
def cached_generate_report_summary(interview_data):
    """Cached version of generate_report_summary"""
    return generate_report_summary(interview_data)

@st.cache_data(ttl=3600)
def cached_analyze_resume(resume_image):
    """Cached version of analyze_resume"""
    return analyze_resume(resume_image)

@st.cache_data(ttl=3600)
def cached_search_google(query, num_results=3):
    """Cached version of search_google"""
    return search_google(query, num_results)

@st.cache_data(ttl=3600)
def cached_verify_answer(question, answer, round_type, topic):
    """Cached version of verify_answer"""
    return verify_answer(question, answer, round_type, topic)

# =========================================================================
# GEMINI API FUNCTIONS
# =========================================================================

def initialize_gemini_api(api_key):
    """Initialize the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    return True

def generate_interview_questions(round_type, topic, difficulty, num_questions=5):
    """
    Generate interview questions using Gemini API based on the specified parameters.
    
    Args:
        round_type: Type of interview round (Technical, HR, etc.)
        topic: Specific topic for questions
        difficulty: Difficulty level (Easy, Medium, Hard)
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Different prompt for coding rounds
        if "Coding" in round_type:
            prompt = f"""Generate {num_questions} {difficulty} level coding interview questions for a {round_type} interview on the topic "{topic}".
            The questions should be challenging but appropriate for the {difficulty} difficulty level.
            Each question should require writing actual code.
            Format the response as a numbered list with just the questions, no additional text.
            Each question should be a specific coding task that requires implementation of a function, algorithm, or data structure.
            Include specific requirements like input/output formats, constraints, and examples if appropriate.
            """
        else:
            prompt = f"""Generate {num_questions} {difficulty} level interview questions for a {round_type} round interview on the topic "{topic}".
            The questions should be challenging but appropriate for the {difficulty} difficulty level.
            Format the response as a numbered list with just the questions, no additional text.
            Each question should be concise and clear.
            """
        
        response = model.generate_content(prompt)
        
        # Process the response to extract questions
        raw_text = response.text
        lines = raw_text.strip().split('\n')
        
        # Clean up the questions
        questions = []
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Try to find and extract questions with numbering
            parts = line.strip().split('.', 1)
            if len(parts) > 1 and parts[0].strip().isdigit():
                questions.append(parts[1].strip())
            else:
                # If no numbering found, just add the line if it looks like a question
                if '?' in line or len(line) > 20:  # Simple heuristic for questions
                    questions.append(line.strip())
        
        # Ensure we have enough questions
        return questions[:num_questions]
    
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return [f"Failed to generate questions: {str(e)}"]

def evaluate_answer(question, answer, round_type, topic, difficulty):
    """
    Evaluate the user's answer to a question using Gemini API.
    
    Args:
        question: The interview question
        answer: User's answer to evaluate
        round_type: Type of interview round
        topic: Specific topic of the question
        difficulty: Difficulty level
        
    Returns:
        Dictionary containing evaluation score and feedback
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Different prompt for coding rounds
        if "Coding" in round_type:
            prompt = f"""As an expert {round_type} interviewer specializing in {topic}, evaluate the following code solution to this {difficulty} level question:

Question: {question}

Code Solution: {answer}

Provide your evaluation in the following format:
Score: [Give a score out of 10]
Feedback: [Provide constructive feedback about the code's correctness, efficiency, and style]
Improvements: [Suggest specific improvements the candidate could make]

Be fair but thorough in your assessment. Consider the:
- Correctness (does it solve the problem correctly?)
- Efficiency (time and space complexity)
- Code quality (readability, style, naming conventions)
- Error handling (edge cases considered)
- Comments and documentation
"""
        else:
            prompt = f"""As an expert {round_type} interviewer specializing in {topic}, evaluate the following answer to this {difficulty} level question:

Question: {question}

Answer: {answer}

Provide your evaluation in the following format:
Score: [Give a score out of 10]
Feedback: [Provide constructive feedback about the answer's strengths and weaknesses]
Improvements: [Suggest specific improvements the candidate could make]

Be fair but thorough in your assessment. Consider the technical accuracy, completeness, clarity, and relevance of the answer.
"""
        
        response = model.generate_content(prompt)
        
        # Process the response to extract evaluation components
        raw_text = response.text
        
        # Default values
        evaluation = {
            "score": 0,
            "feedback": "Unable to evaluate answer",
            "improvements": "N/A"
        }
        
        # Extract score
        if "Score:" in raw_text:
            score_text = raw_text.split("Score:")[1].split("\n")[0].strip()
            try:
                # Extract numeric score (handles formats like "7/10" or just "7")
                score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                    # Normalize to be out of 10 if needed
                    if "/10" in score_text or "/10." in score_text:
                        evaluation["score"] = score
                    else:
                        evaluation["score"] = min(score, 10)  # Cap at 10 if higher
            except:
                evaluation["score"] = 5  # Default to middle score if parsing fails
        
        # Extract feedback
        if "Feedback:" in raw_text:
            if "Improvements:" in raw_text:
                feedback = raw_text.split("Feedback:")[1].split("Improvements:")[0].strip()
            else:
                feedback = raw_text.split("Feedback:")[1].strip()
            evaluation["feedback"] = feedback
        
        # Extract improvements
        if "Improvements:" in raw_text:
            improvements = raw_text.split("Improvements:")[1].strip()
            evaluation["improvements"] = improvements
        
        return evaluation
    
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        return {
            "score": 0,
            "feedback": f"Error during evaluation: {str(e)}",
            "improvements": "Try again or continue with the next question"
        }

def generate_report_summary(interview_data):
    """
    Generate a summary report of the entire interview using Gemini API.
    
    Args:
        interview_data: Dictionary containing all interview information
        
    Returns:
        Dictionary with report sections
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare questions and answers for the prompt
        qa_pairs = ""
        for i, (q, a, e) in enumerate(zip(interview_data["questions"], 
                                          interview_data["answers"], 
                                          interview_data["evaluations"])):
            qa_pairs += f"Q{i+1}: {q}\nA{i+1}: {a}\nScore: {e['score']}/10\n\n"
        
        prompt = f"""As an expert interviewer, provide a comprehensive summary report for the following {interview_data['round_type']} interview on {interview_data['topic']} at {interview_data['difficulty']} difficulty level:

{qa_pairs}

The candidate's average score was {interview_data['total_score']/len(interview_data['questions']):.1f}/10.

Provide your report in the following format:
1. Overall Performance: [A paragraph summarizing the overall performance]
2. Strengths: [Bullet point list of the candidate's demonstrated strengths]
3. Areas for Improvement: [Bullet point list of areas where the candidate can improve]
4. Specific Recommendations: [3-5 actionable recommendations for improvement]
5. Final Verdict: [Hiring recommendation based on this interview performance]

Be professional, constructive, and specific in your feedback.
"""
        
        response = model.generate_content(prompt)
        
        # Return the raw report text 
        return {
            "summary": response.text,
            "average_score": interview_data['total_score']/len(interview_data['questions'])
        }
    
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return {
            "summary": f"Error generating report: {str(e)}",
            "average_score": interview_data['total_score']/len(interview_data['questions']) if len(interview_data['questions']) > 0 else 0
        }

# =========================================================================
# RESUME ANALYSIS FUNCTIONS
# =========================================================================

def extract_text_from_image(image):
    """
    Extract text from uploaded resume image using pytesseract OCR.
    
    Args:
        image: PIL Image object
    
    Returns:
        Extracted text as string
    """
    try:
        # Convert image to grayscale for better OCR
        img_gray = image.convert('L')
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img_gray)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

def analyze_resume(resume_image):
    """
    Analyze the resume using Gemini multimodal capabilities.
    
    Args:
        resume_image: The uploaded resume image
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # First, try to extract text from image using OCR
        extracted_text = extract_text_from_image(resume_image)
        
        # If text extraction fails or is limited, use Gemini's image understanding
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare the image for Gemini API
        bytes_data = io.BytesIO()
        resume_image.save(bytes_data, format=resume_image.format)
        bytes_data = bytes_data.getvalue()
        
        prompt = """Analyze this resume image in detail and provide the following information:
1. Candidate Name
2. Contact Information
3. Education (degrees, institutions, years)
4. Experience (companies, roles, durations)
5. Skills (technical, soft skills)
6. Projects
7. Certifications
8. Areas of expertise

Also provide a brief assessment of the candidate's strengths based on the resume.
If any information is not visible or unclear, indicate so rather than guessing.
"""

        # For better performance, send both the image and extracted text
        if extracted_text:
            prompt += f"\n\nHere is the OCR-extracted text to assist your analysis:\n{extracted_text}"
        
        response = model.generate_content([prompt, bytes_data])
        
        return {
            "analysis": response.text,
            "extracted_text": extracted_text
        }
    
    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return {
            "analysis": f"Error analyzing resume: {str(e)}",
            "extracted_text": ""
        }

# =========================================================================
# GOOGLE SEARCH FUNCTIONS
# =========================================================================

def search_google(query, num_results=3):
    """
    Search Google for a query and return results.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of search result dictionaries
    """
    try:
        # API key and CSE ID should be set in environment variables
        api_key = os.getenv("GOOGLE_API_KEY", "")
        cse_id = os.getenv("GOOGLE_CSE_ID", "")
        
        if not api_key or not cse_id:
            return [{"title": "Google Search API credentials not found", 
                    "link": "#", 
                    "snippet": "Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables."}]
        
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Execute the search
        result = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        
        # Extract search results
        search_results = []
        if "items" in result:
            for item in result["items"]:
                search_results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
        
        return search_results
    
    except Exception as e:
        st.error(f"Error searching Google: {str(e)}")
        return [{"title": f"Error searching Google: {str(e)}", 
                "link": "#", 
                "snippet": "Please try again later or check API credentials."}]

def verify_answer(question, answer, round_type, topic):
    """
    Verify the technical accuracy of an answer using Google Search.
    
    Args:
        question: The interview question
        answer: User's answer to verify
        round_type: Type of interview round
        topic: Specific topic of the question
        
    Returns:
        Dictionary with verification results
    """
    try:
        # Generate a search query based on the question and answer
        search_query = f"{question} {topic}"
        
        # Search Google for related information
        search_results = search_google(search_query)
        
        # Let Gemini analyze the search results and verify the answer
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare search results for the prompt
        search_info = ""
        for i, result in enumerate(search_results):
            search_info += f"Source {i+1}: {result['title']}\n"
            search_info += f"URL: {result['link']}\n"
            search_info += f"Description: {result['snippet']}\n\n"
        
        prompt = f"""As a technical verification expert for {topic}, verify the accuracy of this answer to a {round_type} interview question.

Question: {question}

Candidate's Answer: {answer}

Here are some reference sources from a web search:
{search_info}

Provide your verification in the following format:
Accuracy: [Rate the technical accuracy from 0-10]
Verification: [Explain whether the answer is technically accurate based on the search results]
Corrections: [If there are any technical inaccuracies, provide corrections]
Sources: [Reference which sources support or contradict the answer]
"""
        
        response = model.generate_content(prompt)
        
        # Extract verification components from the response
        raw_text = response.text
        
        # Default values
        verification = {
            "accuracy": 0,
            "verification": "Unable to verify answer",
            "corrections": "N/A",
            "sources": "No sources available"
        }
        
        # Extract accuracy
        if "Accuracy:" in raw_text:
            accuracy_text = raw_text.split("Accuracy:")[1].split("\n")[0].strip()
            try:
                # Extract numeric accuracy (handles formats like "7/10" or just "7")
                accuracy_match = re.search(r'(\d+(?:\.\d+)?)', accuracy_text)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))
                    # Normalize to be out of 10 if needed
                    if "/10" in accuracy_text or "/10." in accuracy_text:
                        verification["accuracy"] = accuracy
                    else:
                        verification["accuracy"] = min(accuracy, 10)  # Cap at 10 if higher
            except:
                verification["accuracy"] = 5  # Default to middle score if parsing fails
        
        # Extract verification
        if "Verification:" in raw_text:
            if "Corrections:" in raw_text:
                verify = raw_text.split("Verification:")[1].split("Corrections:")[0].strip()
            else:
                verify = raw_text.split("Verification:")[1].strip()
            verification["verification"] = verify
        
        # Extract corrections
        if "Corrections:" in raw_text:
            if "Sources:" in raw_text:
                corrections = raw_text.split("Corrections:")[1].split("Sources:")[0].strip()
            else:
                corrections = raw_text.split("Corrections:")[1].strip()
            verification["corrections"] = corrections
        
        # Extract sources
        if "Sources:" in raw_text:
            sources = raw_text.split("Sources:")[1].strip()
            verification["sources"] = sources
        
        return verification
    
    except Exception as e:
        st.error(f"Error verifying answer: {str(e)}")
        return {
            "accuracy": 0,
            "verification": f"Error during verification: {str(e)}",
            "corrections": "Unable to provide corrections",
            "sources": "No sources available"
        }

# =========================================================================
# REPORT GENERATOR
# =========================================================================

def generate_performance_report(interview_data):
    """
    Generate a comprehensive performance report from interview data.
    
    Args:
        interview_data: Dictionary containing all interview information and results
        
    Returns:
        HTML report as a string
    """
    if not interview_data or not interview_data.get("questions"):
        return "<h2>No interview data available</h2>"
    
    # Calculate interview stats
    total_questions = len(interview_data["questions"])
    average_score = interview_data["total_score"] / total_questions if total_questions > 0 else 0
    
    # Generate interview summary using Gemini
    summary_report = cached_generate_report_summary(interview_data)
    
    # Create a DataFrame for question-level analysis
    qa_data = []
    for i, (q, a, e) in enumerate(zip(interview_data["questions"], 
                                     interview_data["answers"], 
                                     interview_data["evaluations"])):
        qa_data.append({
            "question_number": i + 1,
            "question": q,
            "answer": a,
            "score": e["score"],
            "feedback": e["feedback"],
            "improvements": e["improvements"]
        })
    
    df = pd.DataFrame(qa_data)
    
    # Create visualizations
    
    # Score distribution
    fig_score_dist = px.histogram(df, x="score", nbins=10, 
                                 title="Score Distribution",
                                 labels={"score": "Score", "count": "Number of Questions"},
                                 color_discrete_sequence=["#3366CC"])
    fig_score_dist.update_layout(xaxis_range=[0, 10])
    
    # Score per question
    fig_score_per_q = px.bar(df, x="question_number", y="score", 
                           title="Score per Question",
                           labels={"question_number": "Question Number", "score": "Score"},
                           color_discrete_sequence=["#3366CC"])
    fig_score_per_q.update_layout(yaxis_range=[0, 10])
    
    # Performance gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=average_score,
        title={"text": "Overall Performance"},
        gauge={
            "axis": {"range": [0, 10]},
            "bar": {"color": "#3366CC"},
            "steps": [
                {"range": [0, 3.33], "color": "#FF4136"},
                {"range": [3.33, 6.66], "color": "#FFDC00"},
                {"range": [6.66, 10], "color": "#2ECC40"}
            ]
        }
    ))
    fig_gauge.update_layout(height=300)
    
    # Generate a timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "summary": summary_report,
        "stats": {
            "total_questions": total_questions,
            "average_score": average_score,
            "timestamp": timestamp
        },
        "qa_data": df,
        "visualizations": {
            "score_distribution": fig_score_dist,
            "score_per_question": fig_score_per_q,
            "performance_gauge": fig_gauge
        }
    }

# =========================================================================
# MAIN APPLICATION UI
# =========================================================================

def main():
    """Main application function"""
    
    # Use session state to maintain state across reruns
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
        st.session_state.interview_phase = "setup"
        st.session_state.current_question_index = 0
        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.evaluations = []
        st.session_state.total_score = 0
        st.session_state.interview_data = {}
        st.session_state.resume_analyzed = False
        st.session_state.resume_analysis = {}
        st.session_state.verification_results = {}
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header styling */
        h1 {
            color: #3366CC;
            text-align: center;
            padding-bottom: 1.5rem;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 2rem;
        }
        
        /* Card-like container for content */
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: bold;
        }
        
        /* Form fields */
        div[data-baseweb="select"] {
            margin-bottom: 1rem;
        }
        
        /* Metrics styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #3366CC !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #3366CC;
        }
        
        /* Success messages */
        .stSuccess {
            padding: 1rem;
            border-radius: 5px;
        }
        
        /* Evaluation container */
        .evaluation-container {
            background-color: #f9f9f9;
            padding: 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            border-left: 5px solid #3366CC;
        }
        
        /* Footer styling */
        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header with enhanced styling
    st.markdown("""
    <h1>🎯 InterviewMaster AI <span style="font-size:0.8em; font-weight:normal; color:#555;">Professional Interview Simulator</span></h1>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration and controls
    with st.sidebar:
        st.header("Interview Configuration")
        
        # API Key input - Allow user override if needed
        api_key = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
        
        if api_key:
            if not st.session_state.initialized or api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                with st.spinner("Initializing API..."):
                    success = initialize_gemini_api(api_key)
                    if success:
                        st.session_state.initialized = True
                        st.success("API initialized successfully!")
                    else:
                        st.error("Failed to initialize API. Please check your API key.")
        else:
            st.warning("Please enter a Gemini API key to proceed.")
        
        st.divider()
        
        # Resume Upload - New feature
        st.subheader("Resume Analysis")
        resume_file = st.file_uploader("Upload your resume (image format)", type=["jpg", "jpeg", "png"])
        
        if resume_file and not st.session_state.resume_analyzed:
            try:
                with st.spinner("Analyzing your resume..."):
                    # Load and process image
                    resume_image = Image.open(resume_file)
                    
                    # Analyze resume
                    st.session_state.resume_analysis = cached_analyze_resume(resume_image)
                    st.session_state.resume_analyzed = True
                    
                    st.success("Resume analyzed successfully!")
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")
        
        if st.session_state.resume_analyzed:
            if st.button("Clear Resume"):
                st.session_state.resume_analyzed = False
                st.session_state.resume_analysis = {}
                st.rerun()
        
        st.divider()
        
        # Interview setup controls - only show if API is initialized
        if st.session_state.initialized:
            round_type = st.selectbox("Interview Round", list(INTERVIEW_ROUNDS.keys()))
            topic = st.selectbox("Topic", INTERVIEW_ROUNDS[round_type])
            difficulty = st.selectbox("Difficulty Level", DIFFICULTY_LEVELS)
            num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=5)
            
            if st.button("Start Interview"):
                with st.spinner("Generating interview questions..."):
                    # Reset interview state
                    st.session_state.interview_phase = "questioning"
                    st.session_state.current_question_index = 0
                    st.session_state.questions = cached_generate_interview_questions(
                        round_type, topic, difficulty, num_questions
                    )
                    st.session_state.answers = [""] * len(st.session_state.questions)
                    st.session_state.evaluations = [{}] * len(st.session_state.questions)
                    st.session_state.total_score = 0
                    st.session_state.verification_results = {}
                    
                    # Store interview settings
                    st.session_state.interview_data = {
                        "round_type": round_type,
                        "topic": topic,
                        "difficulty": difficulty,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "questions": st.session_state.questions,
                        "answers": st.session_state.answers,
                        "evaluations": st.session_state.evaluations,
                        "total_score": 0
                    }
                    
                    st.success("Interview questions generated!")
                    st.rerun()
        
        # Add helpful information
        st.divider()
        st.caption("Developed by AI Interview Simulator Pro Team")
        st.caption("© 2023 All Rights Reserved")
    
    # Main content area
    if not st.session_state.initialized:
        # Welcome page when not initialized
        st.markdown("""
        ## Welcome to AI Interview Simulator Pro
        
        This application helps you prepare for technical and behavioral interviews by:
        
        - Generating realistic interview questions based on your preferences
        - Evaluating your answers with AI-based feedback
        - Analyzing your resume to provide personalized insights
        - Verifying technical answers against authoritative sources
        - Providing comprehensive performance reports
        
        ### Getting Started
        
        1. Enter your Gemini API key in the sidebar
        2. Upload your resume for personalized interview experience (optional)
        3. Configure your interview settings
        4. Start the interview and answer the questions
        5. Receive instant feedback and comprehensive analysis
        
        ### Benefits
        
        - Practice in a stress-free environment
        - Get immediate feedback on your answers
        - Track your progress across multiple practice sessions
        - Identify areas for improvement
        - Prepare more effectively for real interviews
        """)
        
        # Display resume analysis if available
        if st.session_state.resume_analyzed:
            st.markdown("### Resume Analysis")
            with st.expander("View Resume Analysis", expanded=True):
                st.markdown(st.session_state.resume_analysis.get("analysis", "No analysis available"))
    
    elif st.session_state.interview_phase == "setup":
        # Setup phase - Show instructions and resume analysis if available
        st.markdown("""
        ## Interview Setup
        
        Configure your interview settings in the sidebar and click "Start Interview" to begin.
        """)
        
        # Display resume analysis if available
        if st.session_state.resume_analyzed:
            st.markdown("### Resume Analysis")
            with st.expander("View Resume Analysis", expanded=True):
                st.markdown(st.session_state.resume_analysis.get("analysis", "No analysis available"))
    
    elif st.session_state.interview_phase == "questioning":
        # Questioning phase - Show current question and answer input
        progress = st.session_state.current_question_index / len(st.session_state.questions)
        st.progress(progress)
        
        st.subheader(f"Question {st.session_state.current_question_index + 1} of {len(st.session_state.questions)}")
        st.markdown(f"**{st.session_state.questions[st.session_state.current_question_index]}**")
        
        # Check if we have previous answer for this question
        previous_answer = st.session_state.answers[st.session_state.current_question_index]
        user_answer = st.text_area("Your Answer:", value=previous_answer, height=200)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Previous Question", disabled=st.session_state.current_question_index == 0):
                st.session_state.current_question_index -= 1
                st.rerun()
        
        with col2:
            # Improved submit answer button style and placement
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Submit Answer", use_container_width=True, type="primary"):
                    if not user_answer.strip():
                        st.warning("Please provide an answer before submitting.")
                    else:
                        with st.spinner("Evaluating your answer..."):
                            # Save the answer
                            st.session_state.answers[st.session_state.current_question_index] = user_answer
                            
                            # Evaluate the answer
                            evaluation = cached_evaluate_answer(
                                st.session_state.questions[st.session_state.current_question_index],
                                user_answer,
                                st.session_state.interview_data["round_type"],
                                st.session_state.interview_data["topic"],
                                st.session_state.interview_data["difficulty"]
                            )
                            
                            st.session_state.evaluations[st.session_state.current_question_index] = evaluation
                            st.session_state.total_score += evaluation["score"]
                            st.session_state.interview_data["evaluations"] = st.session_state.evaluations
                            st.session_state.interview_data["total_score"] = st.session_state.total_score
                            
                            # Verify answer using Google Search for technical rounds
                            if "Technical" in st.session_state.interview_data["round_type"] or "Coding" in st.session_state.interview_data["round_type"]:
                                with ThreadPoolExecutor() as executor:
                                    # Run verification in a parallel thread to avoid blocking
                                    future = executor.submit(cached_verify_answer,
                                        st.session_state.questions[st.session_state.current_question_index],
                                        user_answer,
                                        st.session_state.interview_data["round_type"],
                                        st.session_state.interview_data["topic"]
                                    )
                                    # Store verification results
                                    verification = future.result()
                                    if not st.session_state.verification_results:
                                        st.session_state.verification_results = {}
                                    st.session_state.verification_results[st.session_state.current_question_index] = verification
                            
                            # Show the evaluation
                            st.success("Answer evaluated!")
                            
                            # Display evaluation
                            st.subheader("Evaluation")
                            
                            # Score indicator
                            score_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=evaluation["score"],
                                title={"text": "Score"},
                                gauge={
                                    "axis": {"range": [0, 10]},
                                    "bar": {"color": "#3366CC"},
                                    "steps": [
                                        {"range": [0, 3.33], "color": "#FF4136"},
                                        {"range": [3.33, 6.66], "color": "#FFDC00"},
                                        {"range": [6.66, 10], "color": "#2ECC40"}
                                    ]
                                }
                            ))
                            score_gauge.update_layout(height=250, width=400)
                            st.plotly_chart(score_gauge)
                            
                            # Display feedback and improvements
                            st.markdown("#### Feedback")
                            st.markdown(evaluation.get("feedback", "No feedback available"))
                            
                            st.markdown("#### Suggested Improvements")
                            st.markdown(evaluation.get("improvements", "No improvements suggested"))
                            
                            # Display verification results if available
                            if (st.session_state.verification_results and 
                                st.session_state.current_question_index in st.session_state.verification_results):
                                verification = st.session_state.verification_results[st.session_state.current_question_index]
                                
                                st.markdown("#### Technical Verification")
                                st.markdown(f"**Accuracy: {verification.get('accuracy', 0)}/10**")
                                st.markdown(verification.get("verification", "No verification available"))
                                
                                if verification.get("corrections", "N/A") != "N/A":
                                    st.markdown("#### Corrections")
                                    st.markdown(verification.get("corrections", "No corrections needed"))
                                
                                st.markdown("#### Sources")
                                st.markdown(verification.get("sources", "No sources available"))
        
        with col3:
            if st.button("Next Question" if st.session_state.current_question_index < len(st.session_state.questions) - 1 else "Finish Interview"):
                # Save current answer if not already saved
                if user_answer != st.session_state.answers[st.session_state.current_question_index]:
                    st.session_state.answers[st.session_state.current_question_index] = user_answer
                
                # Move to next question or finish interview
                if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                    st.session_state.current_question_index += 1
                else:
                    st.session_state.interview_phase = "summary"
                
                st.rerun()
    
    elif st.session_state.interview_phase == "summary":
        # Summary phase - Show performance report
        st.subheader("Interview Performance Report")
        
        with st.spinner("Generating performance report..."):
            # Generate comprehensive report
            report = generate_performance_report(st.session_state.interview_data)
            
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", report["stats"]["total_questions"])
            with col2:
                st.metric("Average Score", f"{report['stats']['average_score']:.1f}/10")
            with col3:
                st.metric("Interview Date", report["stats"]["timestamp"])
            
            # Performance visualizations
            st.plotly_chart(report["visualizations"]["performance_gauge"], use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(report["visualizations"]["score_distribution"], use_container_width=True)
            with col2:
                st.plotly_chart(report["visualizations"]["score_per_question"], use_container_width=True)
            
            # Interview summary
            st.markdown("## Interview Summary")
            st.markdown(report["summary"]["summary"])
            
            # Question-by-question breakdown
            st.markdown("## Question-by-Question Breakdown")
            for i, row in report["qa_data"].iterrows():
                with st.expander(f"Question {i+1}: {row['question'][:100]}...", expanded=False):
                    st.markdown(f"**Question:** {row['question']}")
                    st.markdown(f"**Your Answer:** {row['answer']}")
                    st.markdown(f"**Score:** {row['score']}/10")
                    st.markdown(f"**Feedback:** {row['feedback']}")
                    st.markdown(f"**Improvements:** {row['improvements']}")
                    
                    # Display verification results if available
                    if (st.session_state.verification_results and 
                        i in st.session_state.verification_results):
                        verification = st.session_state.verification_results[i]
                        
                        st.markdown("**Technical Verification**")
                        st.markdown(f"Accuracy: {verification.get('accuracy', 0)}/10")
                        st.markdown(verification.get("verification", "No verification available"))
                        
                        if verification.get("corrections", "N/A") != "N/A":
                            st.markdown("**Corrections:**")
                            st.markdown(verification.get("corrections", "No corrections needed"))
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start New Interview"):
                    # Reset interview state
                    st.session_state.interview_phase = "setup"
                    st.session_state.current_question_index = 0
                    st.session_state.questions = []
                    st.session_state.answers = []
                    st.session_state.evaluations = []
                    st.session_state.total_score = 0
                    st.session_state.interview_data = {}
                    st.session_state.verification_results = {}
                    st.rerun()
            
            # Enhanced download options with multiple file formats
            col1, col2 = st.columns(2)
            with col1:
                # CSV format - data only
                report_csv = report["qa_data"].to_csv(index=False)
                st.download_button(
                    label="Download Report CSV",
                    data=report_csv,
                    file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download raw data in CSV format"
                )
                
            with col2:
                # Create a formatted HTML report for PDF-like experience
                # Build the HTML report without using f-strings with backslashes
                
                # Define CSS as a regular string
                css_style = """
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                        h1 { color: #3366CC; text-align: center; }
                        h2 { color: #3366CC; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                        .stats { display: flex; justify-content: space-between; margin: 20px 0; }
                        .stat-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center; width: 30%; }
                        .stat-value { font-size: 24px; font-weight: bold; margin: 10px 0; color: #3366CC; }
                        .question { background-color: #f9f9f9; padding: 15px; margin: 15px 0; border-left: 5px solid #3366CC; }
                        .score { font-weight: bold; color: #3366CC; }
                        .feedback { margin-top: 10px; }
                        .improvements { margin-top: 10px; color: #555; }
                        .timestamp { text-align: right; font-style: italic; margin-top: 30px; color: #777; }
                    </style>
                """
                
                # Start building the HTML report using string concatenation instead of f-strings with CSS
                html_report = """
                <html>
                <head>
                    <title>InterviewMaster AI - Detailed Performance Report</title>
                """ + css_style + """
                </head>
                <body>
                    <h1>InterviewMaster AI - Detailed Performance Report</h1>
                    
                    <div class="stats">
                        <div class="stat-box">
                            <div>Total Questions</div>
                            <div class="stat-value">""" + str(report["stats"]["total_questions"]) + """</div>
                        </div>
                        <div class="stat-box">
                            <div>Average Score</div>
                            <div class="stat-value">""" + f"{report['stats']['average_score']:.1f}" + """/10</div>
                        </div>
                        <div class="stat-box">
                            <div>Interview Date</div>
                            <div class="stat-value">""" + report["stats"]["timestamp"] + """</div>
                        </div>
                    </div>
                    
                    <h2>Interview Summary</h2>
                    <div>""" + report["summary"]["summary"].replace("\n", "<br>") + """</div>
                    
                    <h2>Question-by-Question Breakdown</h2>
                """
                
                # Add each question to the report
                for i, row in report["qa_data"].iterrows():
                    question_html = """
                    <div class="question">
                        <h3>Question """ + str(i+1) + """: """ + row["question"] + """</h3>
                        <div><strong>Your Answer:</strong> """ + row["answer"] + """</div>
                        <div class="score"><strong>Score:</strong> """ + str(row["score"]) + """/10</div>
                        <div class="feedback"><strong>Feedback:</strong> """ + row["feedback"] + """</div>
                        <div class="improvements"><strong>Improvements:</strong> """ + row["improvements"] + """</div>
                    """
                    
                    # Add verification results if available
                    if (st.session_state.verification_results and i in st.session_state.verification_results):
                        verification = st.session_state.verification_results[i]
                        question_html += """
                        <div><strong>Technical Verification:</strong> 
                            Accuracy: """ + str(verification.get('accuracy', 0)) + """/10<br>
                            """ + verification.get("verification", "No verification available") + """
                        </div>
                        """
                        
                        if verification.get("corrections", "N/A") != "N/A":
                            question_html += """
                            <div><strong>Corrections:</strong> """ + verification.get("corrections", "") + """</div>
                            """
                    
                    question_html += "</div>"
                    html_report += question_html
                
                # Close the HTML document
                html_report += """
                    <div class="timestamp">Report generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download Detailed HTML Report",
                    data=html_report,
                    file_name=f"detailed_interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    help="Download a complete formatted report with all details"
                )

if __name__ == "__main__":
    main()
