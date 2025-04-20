import streamlit as st
import os
import pandas as pd
from utils.gemini_api import initialize_gemini_api
from assets.interview_rounds import INTERVIEW_ROUNDS, DIFFICULTY_LEVELS

# Set page config
st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.title("🎙️ AI-Powered Virtual Interview Simulator")
st.markdown("""
This application simulates a real interview experience with:
- Dynamic interview questions based on topics and difficulty
- Audio-based question delivery
- Speech recognition for your answers
- AI evaluation of your responses
- Performance report generation
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
    3. **Answer Verbally**: Speak into your microphone to answer the questions
    4. **Get Evaluated**: The AI will evaluate your responses in real-time
    5. **Receive Feedback**: Get a detailed performance report after completing the interview
    """)
    
    # Features section
    st.subheader("Key Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- 🎯 **Dynamic Question Generation**")
        st.markdown("- 🔊 **Audio-Driven Interaction**")
        st.markdown("- 🎙️ **Answer Recording & Recognition**")
    with col2:
        st.markdown("- 💡 **Answer Feedback / Evaluation**")
        st.markdown("- 🧾 **Scorecard / Feedback Report**")
        st.markdown("- 💼 **Multiple Interview Rounds**")
else:
    # Import and include the interview page if interview started
    from pages.interview import show_interview_page
    show_interview_page()

# Footer
st.markdown("---")
st.markdown("© 2023 AI Interview Simulator | Powered by Google Gemini 1.5 Pro API")
