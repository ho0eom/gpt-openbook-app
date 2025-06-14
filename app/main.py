import os
import openai
import streamlit as st

# ✅ 1. 환경변수에서 API 키 불러오기 (openai==1.3.5 방식)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 2. 텍스트 파일 불러오기
def load_pdf_text():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        return f.read()

# ✅ 3. GPT에게 질문하고 답변 받기
def ask_gpt(question, context):
    prompt = f"""다음 문서를 참고해서 질문에 답해주세요.

문서 내용:
{context}

질문: {question}
답변:"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 똑똑한 오픈북 시험 도우미야. 주어진 문서에서만 답을 찾아야 해."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# ✅ 4. Streamlit 앱 UI 구성
st.set_page_config(page_title="GPT 오픈북 시험 도우미")
st.title("📘 GPT 오픈북 시험 도우미")

question = st.text_input("질문을 입력하세요:")

if question:
    with st.spinner("답변 생성 중..."):
        context = load_pdf_text()
        answer = ask_gpt(question, context)
        st.markdown("### ✅ 답변")
        st.write(answer)
