import os
import streamlit as st
import openai
import tiktoken
import numpy as np
from rapidfuzz import fuzz

# — 환경 변수에서 키 읽기 —
openai.api_key = os.getenv("OPENAI_API_KEY")

# — 설정값 —
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-3.5-turbo"
CHUNK_SIZE      = 1000   # 글자 단위
CHUNK_OVERLAP   = 200
TOP_K           = 5
FUZZY_THRESHOLD = 60     # 퍼지비교 컷오프

# — 텍스트 로드 + 청크화 + 오버랩 —
@st.cache_data(show_spinner=False)
def load_and_chunk(path: str):
    text = open(path, "r", encoding="utf-8").read()
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + CHUNK_SIZE]
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# — 임베딩 계산 —
@st.cache_data(show_spinner=False)
def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = []
    for txt in texts:
        resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=txt)
        embeddings.append(resp["data"][0]["embedding"])
    return np.array(embeddings)

# — 질문 임베딩 + 상위 K 선택 + 퍼지 필터링 —
def retrieve_best_chunk(question: str, chunks: list[str], embeddings: np.ndarray):
    # 질문 임베딩
    q_emb = openai.embeddings.create(model=EMBEDDING_MODEL, input=question)["data"][0]["embedding"]
    q_vec = np.array(q_emb)

    # 코사인 유사도 계산
    sims = (embeddings @ q_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-8
    )
    idxs = np.argsort(-sims)[:TOP_K]

    # 퍼지매칭으로 2차 필터
    filtered = [(i, chunks[i]) for i in idxs if fuzz.partial_ratio(question, chunks[i]) >= FUZZY_THRESHOLD]
    if not filtered:
        filtered = [(i, chunks[i]) for i in idxs]  # 컷오프 미달 시 상위K로 대체

    return filtered[0]  # (index, text)

# — GPT에게 질문 & 근거 인용 지시 —
def ask_gpt(question: str, context: str):
    prompt = f"""
아래는 문서의 일부입니다. **오직 이 문맥 안에서만** 질문에 답하고, **답의 근거 문장은 문맥에서 똑같이 인용**하세요.


❓ 질문: {question}

▶️ 답변 형식:
1) 답변:
2) 근거 인용(문장 그대로):
"""
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# — Streamlit UI —
st.title("📘 오픈북 Q&A (임베딩+퍼지 검색)")
st.write("오타·동의어 OK · 오직 주어진 문맥에서 답과 근거 인용")

question = st.text_input("❓ 질문을 입력하세요")
if question:
    with st.spinner("문서 로딩 및 임베딩 중..."):
        chunks     = load_and_chunk("pdf_text/your_pdf.txt")
        embeddings = embed_texts(chunks)

    idx, best_chunk = retrieve_best_chunk(question, chunks, embeddings)
    st.markdown(f"**▶️ 선택된 청크 인덱스:** {idx}")

    with st.spinner("AI가 답변을 생성 중..."):
        answer = ask_gpt(question, best_chunk)

    st.subheader("✅ GPT의 답변")
    st.write(answer)
