import streamlit as st
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import google.generativeai as genai

# Set page config
st.set_page_config(
    page_title="Interv.AI",
    page_icon="🎙️",
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
# PLACEHOLDER FOR REMOVED AUDIO PROCESSING FUNCTIONS
# =========================================================================

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
        model = genai.GenerativeModel('gemini-2.5-pro')
        
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
        model = genai.GenerativeModel('gemini-1.5-flash-pro')
        
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
        model = genai.GenerativeModel('gemini-1.5-flash-pro')
        
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
# REPORT GENERATOR
# =========================================================================

def generate_performance_report(interview_data):
    """
    Generate a comprehensive performance report from interview data.
    
    Args:
        interview_data: Dictionary containing all interview information and results
        
    Returns:
        None (displays the report in the Streamlit app)
    """
    st.title("🧾 Interview Performance Report")
    
    # Display interview metadata
    st.subheader("Interview Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Interview Type", interview_data["round_type"])
    with col2:
        st.metric("Topic", interview_data["topic"])
    with col3:
        st.metric("Difficulty", interview_data["difficulty"])
    
    st.metric("Overall Score", f"{interview_data['average_score']:.1f}/10")
    
    # Display the AI-generated summary
    st.subheader("Performance Summary")
    st.markdown(interview_data["report_summary"])
    
    # Q&A breakdown with scores
    st.subheader("Question-by-Question Breakdown")
    
    qa_data = []
    for i, (question, answer, evaluation) in enumerate(zip(
            interview_data["questions"], 
            interview_data["answers"], 
            interview_data["evaluations"])):
        qa_data.append({
            "Question Number": i+1,
            "Question": question,
            "Your Answer": answer,
            "Score": evaluation["score"],
            "Feedback": evaluation["feedback"]
        })
    
    qa_df = pd.DataFrame(qa_data)
    
    # Display the Q&A in a nice format
    for i, row in qa_df.iterrows():
        with st.expander(f"Question {i+1}: {row['Question'][:100]}..."):
            st.markdown(f"**Question:** {row['Question']}")
            st.markdown(f"**Your Answer:** {row['Your Answer']}")
            st.markdown(f"**Score:** {row['Score']}/10")
            st.markdown(f"**Feedback:** {row['Feedback']}")
    
    # Visualizations
    st.subheader("Performance Visualization")
    
    # Radar chart of scores using Plotly
    fig = go.Figure()
    
    # Truncate long questions for the radar chart
    labels = [f"Q{i+1}: {q[:30]}..." for i, q in enumerate(interview_data["questions"])]
    values = [e["score"] for e in interview_data["evaluations"]]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Score distribution bar chart
    score_counts = qa_df['Score'].value_counts().sort_index()
    
    fig = px.bar(
        x=score_counts.index,
        y=score_counts.values,
        labels={'x': 'Score', 'y': 'Number of Questions'},
        title='Score Distribution'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.subheader("Download Report")
    
    # Convert interview data to CSV
    csv_data = pd.DataFrame({
        'Question': interview_data["questions"],
        'Your Answer': interview_data["answers"],
        'Score': [e["score"] for e in interview_data["evaluations"]],
        'Feedback': [e["feedback"] for e in interview_data["evaluations"]],
        'Improvements': [e["improvements"] for e in interview_data["evaluations"]]
    })
    
    csv = csv_data.to_csv(index=False)
    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Try again or return to main menu buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try Another Interview", key="try_again"):
            # Reset relevant session state variables
            st.session_state.interview_started = False
            st.session_state.interview_completed = False
            st.session_state.current_question_index = 0
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.evaluations = []
            st.session_state.selected_round = None
            st.session_state.selected_topic = None
            st.session_state.selected_difficulty = None
            st.session_state.total_score = 0
            st.rerun()
    
    with col2:
        if st.button("Return to Main Menu", key="main_menu"):
            # Reset all session state variables
            for key in list(st.session_state.keys()):
                if key not in ["gemini_api_initialized", "api_key"]:
                    del st.session_state[key]
            st.session_state.initialized = True
            st.session_state.interview_started = False
            st.session_state.interview_completed = False
            st.rerun()

# =========================================================================
# INTERVIEW PAGE
# =========================================================================

def show_interview_page():
    """Display the interview interface once the interview has started"""
    
    # Generate questions if not already generated
    if not st.session_state.questions:
        with st.spinner("Generating interview questions..."):
            st.session_state.questions = generate_interview_questions(
                st.session_state.selected_round,
                st.session_state.selected_topic,
                st.session_state.selected_difficulty,
                st.session_state.num_questions
            )
    
    # Display current interview progress
    total_questions = len(st.session_state.questions)
    current_index = st.session_state.current_question_index
    
    st.progress(current_index / total_questions)
    st.markdown(f"**Question {current_index + 1} of {total_questions}**")
    
    # Display interview metadata
    st.sidebar.markdown("## Current Interview")
    st.sidebar.markdown(f"**Round:** {st.session_state.selected_round}")
    st.sidebar.markdown(f"**Topic:** {st.session_state.selected_topic}")
    st.sidebar.markdown(f"**Difficulty:** {st.session_state.selected_difficulty}")
    
    # If the interview is complete, show the report
    if st.session_state.interview_completed:
        show_report_page()
        return
    
    # Get current question
    current_question = st.session_state.questions[current_index]
    
    # Display the question
    question_container = st.container()
    with question_container:
        st.subheader("Question:")
        st.markdown(f"{current_question}")
    
    # Answer section
    answer_container = st.container()
    with answer_container:
        st.subheader("Your Answer:")
        
        # Check if we've received an answer for this question
        if current_index < len(st.session_state.answers):
            # Show previously recorded answer
            st.markdown(f"{st.session_state.answers[current_index]}")
            
            # Also show evaluation if available
            if current_index < len(st.session_state.evaluations):
                evaluation = st.session_state.evaluations[current_index]
                st.markdown("---")
                st.subheader("Evaluation:")
                st.markdown(f"**Score:** {evaluation['score']}/10")
                st.markdown(f"**Feedback:** {evaluation['feedback']}")
                st.markdown(f"**Suggested Improvements:** {evaluation['improvements']}")
        else:
            # Text input only (with special handling for coding questions)
            use_code_editor = "Coding" in st.session_state.selected_round
            
            if use_code_editor:
                answer_text = st.text_area("Type your code answer:", height=300, key="code_answer", 
                                          help="Write your code solution here. Use proper indentation and formatting.")
                st.info("For coding questions, please include comments explaining your approach.")
            else:
                answer_text = st.text_area("Type your answer:", height=200, key="text_answer")
            
            if st.button("Submit Answer", key="submit_text_answer"):
                if answer_text.strip():
                    st.session_state.answers.append(answer_text)
                    
                    # Evaluate the answer
                    with st.spinner("Evaluating your answer..."):
                        evaluation = evaluate_answer(
                            current_question,
                            answer_text,
                            st.session_state.selected_round,
                            st.session_state.selected_topic,
                            st.session_state.selected_difficulty
                        )
                        st.session_state.evaluations.append(evaluation)
                        st.session_state.total_score += evaluation["score"]
                    
                    st.rerun()
                else:
                    st.error("Please enter an answer before submitting.")
    
    # Navigation buttons
    nav_container = st.container()
    with nav_container:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if current_index > 0:
                if st.button("Previous Question", key="prev_question"):
                    st.session_state.current_question_index -= 1
                    st.rerun()
        
        with col2:
            # Only show next button if we have an answer for the current question
            if current_index < len(st.session_state.answers) - 1:
                if st.button("Next Question", key="next_question"):
                    st.session_state.current_question_index += 1
                    st.rerun()
            elif current_index < total_questions - 1 and current_index < len(st.session_state.answers):
                if st.button("Next Question", key="next_question"):
                    st.session_state.current_question_index += 1
                    st.rerun()
            elif current_index == total_questions - 1 and current_index < len(st.session_state.answers):
                if st.button("Finish Interview", key="finish_interview"):
                    st.session_state.interview_completed = True
                    
                    # Generate the final report summary
                    with st.spinner("Generating your interview report..."):
                        report_data = {
                            "round_type": st.session_state.selected_round,
                            "topic": st.session_state.selected_topic,
                            "difficulty": st.session_state.selected_difficulty,
                            "questions": st.session_state.questions,
                            "answers": st.session_state.answers,
                            "evaluations": st.session_state.evaluations,
                            "total_score": st.session_state.total_score
                        }
                        
                        report_summary = generate_report_summary(report_data)
                        st.session_state.report_summary = report_summary["summary"]
                        st.session_state.average_score = report_summary["average_score"]
                    
                    st.rerun()

def show_report_page():
    """Display the final interview report"""
    
    # Prepare report data
    report_data = {
        "round_type": st.session_state.selected_round,
        "topic": st.session_state.selected_topic,
        "difficulty": st.session_state.selected_difficulty,
        "questions": st.session_state.questions,
        "answers": st.session_state.answers,
        "evaluations": st.session_state.evaluations,
        "total_score": st.session_state.total_score,
        "report_summary": st.session_state.report_summary,
        "average_score": st.session_state.average_score
    }
    
    # Generate and display the report
    generate_performance_report(report_data)

# =========================================================================
# MAIN APPLICATION
# =========================================================================

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.interview_started = False
    st.session_state.interview_completed = False
    st.session_state.current_question_index = 0
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.evaluations = []
    st.session_state.selected_round = None
    st.session_state.selected_topic = None
    st.session_state.selected_difficulty = None
    st.session_state.total_score = 0
    st.session_state.gemini_api_initialized = False
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
    

# Main application header
st.title("🎙️ Interv.AI")
st.markdown("""
This application simulates a real interview experience with:
- Dynamic interview questions based on topics and difficulty
- Text input for your answers with specialized coding input options
- Specialized coding questions with code evaluation
- AI evaluation of your responses using Gemini 1.5 Pro
- Comprehensive performance report generation
""")

# Sidebar for interview configuration
with st.sidebar:
    st.header("Interview Configuration")
    
    # API Key input (if not provided in environment)
    if not st.session_state.api_key:
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        if api_key:
            st.session_state.api_key = api_key
    
    if st.session_state.api_key and not st.session_state.gemini_api_initialized:
        try:
            initialize_gemini_api(st.session_state.api_key)
            st.session_state.gemini_api_initialized = True
            st.success("Gemini API initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize Gemini API: {str(e)}")
    
    # Interview settings
    if st.session_state.gemini_api_initialized and not st.session_state.interview_started:
        st.subheader("Select Interview Type")
        
        # Round selection
        round_options = list(INTERVIEW_ROUNDS.keys())
        selected_round = st.selectbox("Interview Round", round_options)
        
        # Topic selection based on round
        if selected_round:
            topic_options = INTERVIEW_ROUNDS[selected_round]
            selected_topic = st.selectbox("Topic", topic_options)
            
            # Difficulty selection
            selected_difficulty = st.selectbox("Difficulty Level", DIFFICULTY_LEVELS)
            
            # Number of questions
            num_questions = st.slider("Number of Questions", min_value=3, max_value=10, value=5)
            
            # Start interview button
            if st.button("Start Interview"):
                if selected_round and selected_topic and selected_difficulty:
                    st.session_state.selected_round = selected_round
                    st.session_state.selected_topic = selected_topic
                    st.session_state.selected_difficulty = selected_difficulty
                    st.session_state.num_questions = num_questions
                    st.session_state.interview_started = True
                    st.rerun()
                else:
                    st.warning("Please select all interview parameters")

# Main content area
if not st.session_state.interview_started:
    if not st.session_state.gemini_api_initialized:
        st.info("Please provide a Gemini API key to start using the interview simulator.")
    else:
        st.info("Configure your interview settings in the sidebar and click 'Start Interview' to begin.")
    
    # Display example interview information
    st.subheader("Available Interview Rounds")
    rounds_data = []
    for round_name, topics in INTERVIEW_ROUNDS.items():
        rounds_data.append({"Round Type": round_name, "Available Topics": ", ".join(topics[:3]) + "..."})
    
    st.table(pd.DataFrame(rounds_data))
    
    # How it works section
    st.subheader("How It Works")
    st.markdown("""
    1. **Select Interview Parameters**: Choose the round type, topic, and difficulty level
    2. **Start the Interview**: The system will generate questions based on your selections
    3. **Answer Questions**: Type your answers (including specialized code editor for coding questions)
    4. **Get Evaluated**: The AI will evaluate your responses in real-time
    5. **Receive Feedback**: Get a detailed performance report after completing the interview
    """)
    
    # Features section
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("- 🎯 **Dynamic Question Generation**")
        st.markdown("- 💻 **Code-Specific Questions**")
        st.markdown("- 📝 **Specialized Input Options**")
    with col2:
        st.markdown("- 🧠 **Smart Answer Evaluation**")
        st.markdown("- 📊 **Performance Analytics**")
        st.markdown("- 📑 **Code Analysis**")
    with col3:
        st.markdown("- 📈 **Detailed Feedback Report**")
        st.markdown("- 💼 **Multiple Interview Types**")
        st.markdown("- 🚀 **Powered by Gemini 1.5 Pro**")
else:
    # Show interview page if interview started
    show_interview_page()

# Footer
st.markdown("---")
st.markdown("© 2023 AI Interview Simulator | Powered by Google Gemini 1.5 Pro API")
