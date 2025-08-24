import streamlit as st
import os
import re
import json
import requests
from typing import List, Dict, Any

# ------------------- CONFIG ------------------- #
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")  # from Streamlit secrets or env var
MODEL_ID = "microsoft/DialoGPT-medium"  # Better model for quiz generation

# ------------------- CUSTOM HF ERROR ------------------- #
class HFError(Exception):
    """Custom error class for Hugging Face API failures."""
    pass

# ------------------- HF GENERATE FUNCTION ------------------- #
def generate(prompt: str, hf_token: str, model_id: str) -> str:
    """
    Calls the Hugging Face Inference API with the given prompt.
    
    Args:
        prompt (str): The input text / instruction.
        hf_token (str): Hugging Face API token (hf_xxx).
        model_id (str): The Hugging Face model repo ID.
    
    Returns:
        str: The generated text from the model.
    
    Raises:
        HFError: If the API call fails or Hugging Face returns an error.
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": prompt},
            timeout=120,  # Increased timeout for larger models
        )
        
        # Check if the model is still loading
        if response.status_code == 503:
            est_time = "unknown"
            try:
                data = response.json()
                if "estimated_time" in data:
                    est_time = f"{data['estimated_time']:.1f}"
            except:
                pass
            raise HFError(f"Model is loading. Please try again in {est_time} seconds.")
        
        response.raise_for_status()
        data = response.json()

        # Handle Hugging Face error message
        if isinstance(data, dict) and "error" in data:
            raise HFError(data["error"])

        # Handle different response formats from different models
        if isinstance(data, list) and len(data) > 0:
            # For text generation models
            if "generated_text" in data[0]:
                return data[0]["generated_text"]
            # For conversational models like DialoGPT
            elif "conversation" in data[0]:
                return data[0]["conversation"]["generated_responses"][0]
        
        # For models that return a dictionary directly
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
            
        # Fallback - return the raw data as string
        return str(data)

    except requests.exceptions.RequestException as e:
        raise HFError(f"Request failed: {e}")

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
    Generate {num_questions} multiple-choice quiz questions on the topic '{topic}'.
    Each question must follow this format:
    Q1: <question>
    A) option 1
    B) option 2
    C) option 3
    D) option 4
    Answer: <correct option letter>
    
    Make sure the questions are diverse and cover different aspects of {topic}.
    """
    
    try:
        raw_output = generate(prompt, HF_TOKEN, MODEL_ID)
        quiz_data = format_quiz(raw_output)
        
        # If we got empty quiz data, raise an error
        if not quiz_data:
            raise HFError("AI-generated quiz was empty or invalid.")
            
        return quiz_data[:num_questions]  # Ensure we return only the requested number
    
    except HFError as e:
        raise e

# ------------------- STREAMLIT UI ------------------- #
st.set_page_config(page_title="Quiz Generator 🎯", page_icon="📘", layout="centered")

st.title("📘 AI Quiz Generator")
st.write("Enter a topic and generate an interactive quiz powered by Hugging Face!")

# Add token input field if not set in environment
if not HF_TOKEN:
    HF_TOKEN = st.text_input("Hugging Face API Token", type="password")
    st.info("Get your token from https://huggingface.co/settings/tokens")

topic = st.text_input("Enter topic", placeholder="e.g. Python, Databases, Cybersecurity")
num_questions = st.slider("Number of questions", 2, 10, 5)

if st.button("Generate Quiz"):
    if not HF_TOKEN:
        st.error("❌ Hugging Face API Token is required. Please enter your token above.")
    elif not topic.strip():
        st.error("❌ Please enter a topic for your quiz.")
    else:
        with st.spinner("⚡ Generating quiz..."):
            try:
                quiz = create_quiz(topic, num_questions)
                st.session_state.quiz = quiz
                st.session_state.answers = {}
                st.success("✅ Quiz generated successfully!")
            except HFError as e:
                st.error(f"❌ Failed to generate quiz: {str(e)}")

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
