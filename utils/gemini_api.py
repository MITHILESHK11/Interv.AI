import google.generativeai as genai
import os
import streamlit as st

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
                import re
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
