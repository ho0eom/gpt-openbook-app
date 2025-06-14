import os
import streamlit as st
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# 텍스트 불러오기
@st.cache_data
def load_text():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        return f.read()

pdf_text = load_text()

st.title("📖 오픈북 PDF Q&A")
st.write("PDF 내용 기반 질문을 입력하면 AI가 근거와 함께 답변합니다.")

question = st.text_input("질문을 입력하세요")

if question:
    messages = [
        {"role": "system", "content": "너는 주어진 텍스트를 참고해서 질문에 답변하는 어시스턴트야. 반드시 텍스트에서 근거를 찾아서 답변해야 해."},
        {"role": "user", "content": f"아래 텍스트를 참고해서 질문에 답변해줘:\n\n{pdf_text}\n\n질문: {question}"}
    ]

    with st.spinner("답변 생성 중..."):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        answer = response.choices[0].message.content
        st.success("답변:")
        st.write(answer)
