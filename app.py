import streamlit as st
import os
import re
import json
import google.generativeai as genai
from typing import List, Dict, Any

# ------------------- CONFIG ------------------- #
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  # from Streamlit secrets or env var

# ------------------- CUSTOM ERROR ------------------- #
class AIError(Exception):
    """Custom error class for AI API failures."""
    pass

# ------------------- GEMINI GENERATE FUNCTION ------------------- #
def generate_gemini_response(prompt: str, api_key: str, model_id: str = "gemini-1.5-flash", 
                           max_output_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    Calls the Gemini API with the given prompt.
    
    Args:
        prompt (str): The input text / instruction.
        api_key (str): Google AI Studio API key.
        model_id (str): The Gemini model to use.
        max_output_tokens (int): Maximum tokens to generate.
        temperature (float): Creativity of the response (0-1).
    
    Returns:
        str: The generated text from the model.
    
    Raises:
        AIError: If the API call fails or returns an error.
    """
    if not api_key:
        raise AIError("Missing GOOGLE_API_KEY.")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(
            prompt, 
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        )
        
        if not response or not response.text:
            raise AIError("Empty response from Gemini.")
            
        return response.text.strip()
        
    except Exception as e:
        raise AIError(f"Gemini API call failed: {str(e)}")

# ------------------- QUIZ FORMATTING ------------------- #
def format_quiz(raw_text: str):
    questions = []
    
    # Try to parse as JSON first (in case the model returns JSON)
    try:
        if raw_text.strip().startswith('[') or raw_text.strip().startswith('{'):
            quiz_data = json.loads(raw_text)
            if isinstance(quiz_data, list):
                return quiz_data
            elif isinstance(quiz_data, dict) and 'questions' in quiz_data:
                return quiz_data['questions']
    except json.JSONDecodeError:
        pass  # Not JSON, continue with regex parsing
    
    # Original regex parsing
    parts = re.split(r"(?:Q\d+[:.-])", raw_text, flags=re.IGNORECASE)

    for part in parts:
        if not part.strip():
            continue
        lines = [line.strip() for line in part.split("\n") if line.strip()]

        if not lines:
            continue

        question = lines[0]
        options = [line for line in lines[1:] if re.match(r"^[A-Da-d][).]", line)]
        answer = next((line for line in lines if "answer" in line.lower()), "Answer: Not provided")

        questions.append({
            "question": question,
            "options": options,
            "answer": answer
        })
    
    return questions

# ------------------- QUIZ CREATION ------------------- #
def create_quiz(topic: str, num_questions: int = 5):
    prompt = f"""
    Create {num_questions} multiple-choice quiz questions about {topic}.
    Format each question exactly like this:
    Q1: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Answer: [Correct letter]
    
    Make sure the questions are clear and the answers are correct.
    """
    
    try:
        raw_output = generate_gemini_response(prompt, GEMINI_API_KEY)
        quiz_data = format_quiz(raw_output)
        
        # If we got empty quiz data, raise an error
        if not quiz_data:
            raise AIError("AI-generated quiz was empty or invalid.")
            
        return quiz_data[:num_questions]  # Ensure we return only the requested number
    
    except AIError as e:
        raise e

# ------------------- STREAMLIT UI ------------------- #
st.set_page_config(page_title="Quiz Generator 🎯", page_icon="📘", layout="centered")

st.title("📘 AI Quiz Generator")
st.write("Enter a topic and generate an interactive quiz powered by Google Gemini!")

# Add API key input field if not set in environment
if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.text_input("Google AI Studio API Key", type="password")
    st.info("Get your API key from https://aistudio.google.com/app/apikey")
    st.info("Using model: Gemini 1.5 Flash")

topic = st.text_input("Enter topic", placeholder="e.g. Python, Databases, Cybersecurity")
num_questions = st.slider("Number of questions", 2, 10, 5)

if st.button("Generate Quiz"):
    if not GEMINI_API_KEY:
        st.error("❌ Google API Key is required. Please enter your key above.")
    elif not topic.strip():
        st.error("❌ Please enter a topic for your quiz.")
    else:
        with st.spinner("⚡ Generating quiz..."):
            try:
                quiz = create_quiz(topic, num_questions)
                st.session_state.quiz = quiz
                st.session_state.answers = {}
                st.success("✅ Quiz generated successfully!")
            except AIError as e:
                st.error(f"❌ Failed to generate quiz: {str(e)}")
                st.info("💡 Tip: Make sure your Google API key is valid.")

# ------------------- QUIZ DISPLAY ------------------- #
if "quiz" in st.session_state:
    st.subheader("📝 Quiz Time!")
    st.write(f"Topic: {topic}")

    for idx, q in enumerate(st.session_state.quiz, 1):
        st.markdown(f"**Q{idx}: {q['question']}**")

        # Show radio options
        choice = st.radio(
            f"Select your answer for Q{idx}",
            q["options"],
            key=f"q{idx}",
            index=None
        )
        st.session_state.answers[f"Q{idx}"] = choice

    if st.button("Submit Quiz"):
        score = 0
        st.subheader("📊 Results")

        for idx, q in enumerate(st.session_state.quiz, 1):
            user_ans = st.session_state.answers.get(f"Q{idx}")
            correct_ans = q["answer"]
            
            # Extract just the letter from the answer
            correct_letter = re.search(r"[A-D]", correct_ans.upper())
            correct_letter = correct_letter.group(0) if correct_letter else ""

            if user_ans and user_ans[0].upper() == correct_letter:
                st.success(f"Q{idx}: ✅ Correct ({user_ans})")
                score += 1
            else:
                st.error(f"Q{idx}: ❌ Incorrect (Your answer: {user_ans} | Correct: {correct_ans})")

        st.info(f"🏆 Final Score: {score}/{len(st.session_state.quiz)}")
        
        # Add option to generate a new quiz
        if st.button("Generate New Quiz"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
