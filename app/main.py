import streamlit as st
import os
from openai import OpenAI

# OpenAI API 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 텍스트 파일 로딩
def load_pdf_text():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as file:
        return file.read()

# GPT에게 질문과 context 주고 답변 받기
def ask_question(question, context):
    prompt = f"""다음은 오픈북 시험을 위한 질문입니다. 주어진 문맥을 참고하여 답변하세요.

문맥:
{context}

질문:
{question}

답변:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 필요 시 gpt-4로 변경 가능
        messages=[
            {"role": "system", "content": "당신은 PDF 내용을 기반으로 정확한 답변을 제공하는 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# Streamlit 앱 UI
st.title("📖 오픈북 PDF Q&A")
st.write("PDF 내용 기반 질문을 입력하면 AI가 근거와 함께 답변합니다.")

question = st.text_input("질문을 입력하세요")

if question:
    with st.spinner("PDF와 GPT를 참고 중입니다..."):
        context = load_pdf_text()
        answer = ask_question(question, context)
        st.markdown("### ✅ 답변")
        st.write(answer)
