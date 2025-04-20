import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
