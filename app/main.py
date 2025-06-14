import streamlit as st
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_pdf_text():
    text = ""
    for file in os.listdir("pdf_text"):
        if file.endswith(".txt"):
            with open(os.path.join("pdf_text", file), "r", encoding="utf-8") as f:
                text += f.read() + "\n"
    return text

def ask_question(text, question):
    prompt = f"""
다음은 시험 범위입니다. 질문에 대해 정확히 답하고, 어떤 문장에서 근거를 찾았는지도 말해줘.

시험 내용:
{text}

질문:
{question}

답변:"""

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content

st.title("📘 오픈북 시험 도우미 GPT")

question = st.text_input("궁금한 내용을 입력하세요:")

if question:
    with st.spinner("답변 생성 중..."):
        pdf_text = load_pdf_text()
        if not pdf_text.strip():
            st.error("❗ 'pdf_text' 폴더에 .txt 파일이 없습니다.")
        else:
            answer = ask_question(pdf_text, question)
            st.success("✅ 답변 완료:")
            st.write(answer)
