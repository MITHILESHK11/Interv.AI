import pyttsx3
import speech_recognition as sr
import streamlit as st
import threading
import time

# Initialize the text-to-speech engine
def init_tts_engine():
    """Initialize and return the text-to-speech engine"""
    engine = pyttsx3.init()
    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    return engine

# Text-to-speech function
def speak_text(text, engine=None, block=True):
    """
    Convert text to speech and speak it.
    
    Args:
        text: The text to be spoken
        engine: TTS engine, will initialize one if not provided
        block: Whether to block until speech is complete
    """
    if engine is None:
        engine = init_tts_engine()
    
    if block:
        engine.say(text)
        engine.runAndWait()
    else:
        # Use a thread to avoid blocking the Streamlit UI
        def speak_thread():
            engine.say(text)
            engine.runAndWait()
        
        thread = threading.Thread(target=speak_thread)
        thread.daemon = True  # Allow the thread to be terminated when the main program exits
        thread.start()
        return thread

# Speech recognition function
def recognize_speech(timeout=10):
    """
    Record audio from the microphone and convert it to text.
    
    Args:
        timeout: Maximum recording time in seconds
        
    Returns:
        A dictionary with the recognized text and status
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for audio
            try:
                audio = recognizer.listen(source, timeout=timeout)
                st.info("Processing your answer...")
                
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio)
                return {"success": True, "text": text, "error": None}
            
            except sr.WaitTimeoutError:
                return {"success": False, "text": "", "error": "No speech detected within the timeout period."}
            except sr.UnknownValueError:
                return {"success": False, "text": "", "error": "Could not understand the audio. Please try again."}
            except Exception as e:
                return {"success": False, "text": "", "error": f"Error recognizing speech: {str(e)}"}
    
    except Exception as e:
        return {"success": False, "text": "", "error": f"Error accessing microphone: {str(e)}"}

# Audio playback with countdown display
def speak_with_countdown(text, engine=None):
    """
    Speak text with a visual countdown in Streamlit.
    
    Args:
        text: Text to be spoken
        engine: TTS engine to use (optional)
    """
    if engine is None:
        engine = init_tts_engine()
    
    # Calculate approximate speaking time (rough estimate)
    words = len(text.split())
    speaking_time = max(3, words * 0.5)  # ~ 0.5 seconds per word, minimum 3 seconds
    
    # Start speaking in a non-blocking way
    thread = speak_text(text, engine, block=False)
    
    # Display countdown
    placeholder = st.empty()
    start_time = time.time()
    
    while thread.is_alive():
        elapsed = time.time() - start_time
        remaining = max(0, speaking_time - elapsed)
        placeholder.info(f"Speaking... {remaining:.1f}s")
        time.sleep(0.1)
    
    placeholder.empty()
