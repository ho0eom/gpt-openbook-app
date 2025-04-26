import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings

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

question = st.text_input("❓ 질문을 입력하세요:")

if question:
    with st.spinner("🔍 문서를 검색 중입니다..."):
        res = query_engine.query(question)

    # — 답변 출력 —
    answer = getattr(res, "response", "") or getattr(res, "responsestr", "")
    st.markdown("### 🧠 답")
    st.write(answer)

    # — 출처 출력 —
    st.markdown("#### 📚 출처")
    for node_with_score in getattr(res, "source_nodes", []):
        node = node_with_score.node
        meta = node.metadata
        fname = meta.get("file_name", meta.get("file_path", ""))
        page_num = fname.split(".")[0].split("_")[-1]
        text = node.get_text() if hasattr(node, "get_text") else getattr(node, "text", "")
        snippet = text.strip().replace("\n", " ")
        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
        st.write(f"- 페이지 {page_num}: “{snippet}”")

