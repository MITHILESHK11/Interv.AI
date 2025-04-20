import streamlit as st
import time
from utils.gemini_api import generate_interview_questions, evaluate_answer
from utils.audio_processing import speak_text, recognize_speech, init_tts_engine, speak_with_countdown
from utils.report_generator import generate_performance_report

def show_interview_page():
    """Display the interview interface once the interview has started"""
    
    # Initialize or get TTS engine
    if "tts_engine" not in st.session_state:
        st.session_state.tts_engine = init_tts_engine()
    
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
        
        # Speak the question when the button is clicked
        if st.button("Listen to Question", key="speak_question"):
            with st.spinner("Speaking..."):
                speak_with_countdown(current_question, st.session_state.tts_engine)
    
    # Record answer
    answer_container = st.container()
    with answer_container:
        st.subheader("Your Answer:")
        
        # Check if we're in listening mode
        if "is_listening" not in st.session_state:
            st.session_state.is_listening = False
        
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
            # Initialize input method state if needed
            if "input_method" not in st.session_state:
                st.session_state.input_method = "voice"
            
            # Input method selection
            st.radio("Select how to provide your answer:", ["Voice", "Text"], key="input_method_choice", 
                     horizontal=True, on_change=lambda: setattr(st.session_state, "input_method", st.session_state.input_method_choice.lower()))
            
            # Voice input method
            if st.session_state.input_method == "voice":
                if not st.session_state.is_listening:
                    if st.button("Start Recording Answer", key="start_recording"):
                        st.session_state.is_listening = True
                        st.rerun()
                else:
                    # Listening mode active
                    speech_result = recognize_speech(timeout=60)
                    
                    if speech_result["success"]:
                        # Successfully recorded answer
                        answer_text = speech_result["text"]
                        st.session_state.answers.append(answer_text)
                        st.session_state.is_listening = False
                        
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
                        # Failed to record
                        st.error(speech_result["error"])
                        if st.button("Try Again", key="try_recording_again"):
                            st.session_state.is_listening = False
                            st.rerun()
            
            # Text input method
            else:  # text input
                # Check if we need special code editor for coding questions
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
                        
                        from utils.gemini_api import generate_report_summary
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
