def build_quiz_prompt(topic: str, difficulty: str, n: int) -> str:
    return f"""
You are a quiz generator. Create {n} multiple-choice questions on the topic "{topic}".
Difficulty: {difficulty}.
Each question must include exactly 4 options labeled A, B, C, D and provide the correct option letter.
Return ONLY valid JSON with this schema:

{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "string",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "answer": "A|B|C|D",
      "explanation": "short explanation"
    }}
  ]
}}

Do not include backticks or commentary. Return JSON only.
"""

