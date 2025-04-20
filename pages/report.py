import streamlit as st
from utils.report_generator import generate_performance_report

def show_report():
    """
    Display interview report page.
    This page is for directly viewing a saved report, 
    but in the current implementation reports are shown directly after interview.
    """
    if not st.session_state.interview_completed:
        st.warning("No completed interview found. Please complete an interview first.")
        
        if st.button("Return to Main Menu"):
            st.session_state.interview_started = False
            st.rerun()
        return
    
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
