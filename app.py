import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from PIL import Image
import pytesseract
import os

# 🔥 (OCR 경로를 여기 정확히 설정해줘야 함)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ GPT-4o mini 모델 설정
Settings.llm = OpenAI(model="gpt-4o", temperature=0.2)
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

# ✅ PDF 텍스트 인덱스 생성
@st.cache_resource
def load_index():
    docs = SimpleDirectoryReader("pdf_text").load_data()
    return VectorStoreIndex.from_documents(docs)

index = load_index()
query_engine = index.as_query_engine()

# ✅ Streamlit UI
st.title("📘 GPT-4o Mini 오픈북 시험 도우미")

# 사진 업로드
uploaded_file = st.file_uploader("📸 질문이 담긴 사진 업로드 (또는 아래에 직접 입력)", type=["png", "jpg", "jpeg"])

# 기본 질문 텍스트
question = ""

# 사진이 올라오면 OCR로 변환
if uploaded_file:
    image = Image.open(uploaded_file)
    question = pytesseract.image_to_string(image)
    question = question.strip()  # 혹시 모를 공백 제거
    st.text_area("📝 인식된 질문", value=question)

# 사진 없으면 직접 입력
else:
    question = st.text_input("❓ 질문을 입력하세요:")

# 질문이 있으면 처리
if question:
    with st.spinner("🔍 문서를 검색 중입니다..."):
        refined_question = f"""\
아래 질문에 대해 
- 답만 출력해. 
- 설명이나 이유는 쓰지마.

질문: {question}
"""
        res = query_engine.query(refined_question)

    # 답변 출력
    answer = getattr(res, "response", "") or getattr(res, "responsestr", str(res))
    answer = str(answer).strip()
    st.markdown("### 🧠 답")
    st.write(answer)

    # 출처 출력
    if hasattr(res, "source_nodes"):
        st.markdown("### 📚 출처")
        for node_with_score in res.source_nodes:
            node = node_with_score.node
            meta = node.metadata
            fname = meta.get("file_name", meta.get("file_path", ""))
            page_num = fname.split(".")[0].split("_")[-1]
            text = node.get_text() if hasattr(node, "get_text") else getattr(node, "text", "")
            snippet = text.strip().replace("\n", " ")
            snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
            st.write(f"- 페이지 {page_num}: “{snippet}”")
