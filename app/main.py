import os
from openai import OpenAI
import streamlit as st

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "다음 텍스트를 바탕으로 질문에 답하십시오."},
            {"role": "user", "content": f"{context}\n\n질문: {question}"}
        ]
    )
    return response.choices[0].message.content.strip()

def load_txt():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as file:
        return file.read()

# Streamlit UI
st.title("📘 오픈북 시험 질문 도우미")
question = st.text_input("질문을 입력하세요:")

if question:
    context = load_txt()
    answer = ask_gpt(question, context)
    st.subheader("✍️ GPT의 답변")
    st.write(answer)
