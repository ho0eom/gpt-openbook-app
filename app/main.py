# ✅ main.py (복붙용 전체 코드 - openai>=1.0.0 기준, txt 기반, Streamlit UI)

import os
import streamlit as st
from openai import OpenAI

# ✅ OpenAI 클라이언트 초기화 (환경변수에서 API 키 읽음)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ your_pdf.txt에서 context 불러오기
def load_context():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        return f.read()

# ✅ GPT 호출 함수 (최신 API 방식)
def ask_gpt(question, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 텍스트 기반 오픈북 시험 도우미야. 사용자의 질문에 해당 내용이 있으면 관련 내용을 근거로 삼아 답변하고, 출처도 알려줘."},
            {"role": "user", "content": f"질문: {question}\n\n자료 내용:\n{context[:3000]}"}  # context 길이 제한: 3000자까지만
        ]
    )
    return response.choices[0].message.content

# ✅ Streamlit UI
st.title("📖 오픈북 시험 도우미")
st.write("your_pdf.txt를 기반으로 GPT가 답변해줘요.")

question = st.text_input("질문을 입력하세요:")

if question:
    with st.spinner("답변 생성 중..."):
        context = load_context()
        answer = ask_gpt(question, context)
        st.markdown("---")
        st.subheader("✍️ GPT의 답변")
        st.write(answer)
