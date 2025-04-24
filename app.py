import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
import os

# 🔐 OpenAI API 키 입력 (아래에 네 키 붙여넣기!)


# ✅ GPT-4o mini 모델 설정
Settings.llm = OpenAI(model="gpt-4o", temperature=0.2)
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=100)

# ✅ PDF 텍스트 인덱스 생성 (캐시 사용)
@st.cache_resource
def load_index():
    docs = SimpleDirectoryReader("pdf_text").load_data()
    return VectorStoreIndex.from_documents(docs)

# ✅ 인덱스 로드 및 질의엔진 설정
index = load_index()
query_engine = index.as_query_engine()

# ✅ Streamlit UI 구성
st.title("📘 GPT-4o Mini 오픈북 시험 도우미")

question = st.text_input("❓ 질문을 입력하세요:")

if question:
    with st.spinner("🔍 문서를 검색 중입니다..."):
        response = query_engine.query(question)
        st.markdown("🧠 **GPT의 답변:**")
        st.write(response)
