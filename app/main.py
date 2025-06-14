import os
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from io import StringIO

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PDF 파일에서 텍스트 추출 함수
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.set_page_config(page_title="GPT Openbook PDF QA", layout="wide")
st.title("📖 GPT 오픈북 시험 도우미")

uploaded_file = st.file_uploader("📄 시험 범위 PDF 업로드", type="pdf")

if uploaded_file:
    st.success("PDF 업로드 완료 ✅")

    # 텍스트 추출
    with st.spinner("📚 PDF에서 텍스트 추출 중..."):
        extracted_text = extract_text_from_pdf(uploaded_file)

    st.success("텍스트 추출 완료!")

    # 사용자 질문 입력
    question = st.text_input("❓ 궁금한 질문을 입력하세요:")

    if question:
        with st.spinner("🤖 GPT가 답변 중..."):
            prompt = f"""
다음은 시험 범위 PDF에서 추출한 내용입니다.

{text[:4000]}

위 내용을 바탕으로 다음 질문에 답해줘:

"{question}"

가능하면 구체적으로 페이지를 언급해줘.
"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            answer = response.choices[0].message.content
            st.markdown("### ✅ GPT의 답변")
            st.write(answer)
