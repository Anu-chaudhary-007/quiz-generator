import streamlit as st
import os
import re
from providers.hf import generate, HFError

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
MODEL_ID = "tiiuae/falcon-7b-instruct"  # Change model if needed


# ------------------- FORMAT QUIZ ------------------- #
def format_quiz(raw_text: str):
    questions = []
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


# ------------------- CREATE QUIZ ------------------- #
def create_quiz(topic: str, num_questions: int = 5):
    prompt = f"""
    Generate {num_questions} multiple-choice quiz questions on the topic '{topic}'.
    Each question must follow this format:
    Q1: <question>
    A) option 1
    B) option 2
    C) option 3
    D) option 4
    Answer: <correct option>
    """
    try:
        raw_output = generate(prompt, HF_TOKEN, MODEL_ID)
        return format_quiz(raw_output)
    except HFError as e:
        return [{"error": str(e)}]


# ------------------- STREAMLIT UI ------------------- #
st.set_page_config(page_title="Quiz Generator 🎯", page_icon="📘", layout="centered")

st.title("📘 AI Quiz Generator")
st.write("Enter a topic and generate an interactive quiz powered by Hugging Face!")

topic = st.text_input("Enter topic", placeholder="e.g. Python, Databases, Cybersecurity")
num_questions = st.slider("Number of questions", 2, 10, 5)

if st.button("Generate Quiz"):
    if not HF_TOKEN:
        st.error("❌ Hugging Face API Token not found. Set it as an environment variable.")
    else:
        with st.spinner("⚡ Generating quiz..."):
            quiz = create_quiz(topic, num_questions)

        if quiz and "error" in quiz[0]:
            st.error(f"Error: {quiz[0]['error']}")
        else:
            st.session_state.quiz = quiz
            st.session_state.answers = {}


# ------------------- QUIZ DISPLAY ------------------- #
if "quiz" in st.session_state:
    st.subheader("📝 Quiz Time!")

    for idx, q in enumerate(st.session_state.quiz, 1):
        st.markdown(f"**Q{idx}: {q['question']}**")

        # Show radio options
        choice = st.radio(
            f"Select your answer for Q{idx}",
            q["options"],
            key=f"q{idx}"
        )
        st.session_state.answers[f"Q{idx}"] = choice

    if st.button("Submit Quiz"):
        score = 0
        st.subheader("📊 Results")

        for idx, q in enumerate(st.session_state.quiz, 1):
            user_ans = st.session_state.answers.get(f"Q{idx}")
            correct_ans = q["answer"]

            if user_ans and user_ans[0].upper() in correct_ans.upper():
                st.success(f"Q{idx}: ✅ Correct ({user_ans})")
                score += 1
            else:
                st.error(f"Q{idx}: ❌ Wrong (Your: {user_ans} | {correct_ans})")

        st.info(f"🏆 Final Score: {score}/{len(st.session_state.quiz)}")
