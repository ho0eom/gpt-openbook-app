import os
import streamlit as st
import openai

# 환경변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT에 질문 던지는 함수
def ask_gpt(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "다음 텍스트를 바탕으로 질문에 답하십시오."},
            {"role": "user", "content": f"{context}\n\n질문: {question}"}
        ],
    )
    return response.choices[0].message.content.strip()

# 텍스트 파일 로드
def load_txt():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        return f.read()

# Streamlit UI
st.title("📘 오픈북 시험 질문 도우미")
question = st.text_input("질문을 입력하세요:")

if question:
    context = load_txt()
    with st.spinner("답변을 불러오는 중..."):
        answer = ask_gpt(question, context)
    st.subheader("✍️ GPT의 답변")
    st.write(answer)
