# app/main.py
import os
import streamlit as st
import openai
import tiktoken
from rapidfuzz import fuzz
from typing import List, Tuple
import numpy as np

# — 환경변수에서 API 키 세팅 —
openai.api_key = os.getenv("OPENAI_API_KEY")

# — 설정값 —
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
TOP_K = 3
FUZZY_THRESHOLD = 50  # RapidFuzz 점수 필터

# — 1) 텍스트 로드 & 청크 분리 —
@st.cache_data(show_spinner=False)
def load_and_chunk(path: str) -> Tuple[List[Tuple[int,str]], List[List[float]]]:
    full = open(path, "r", encoding="utf-8").read()
    # 겹침(chunk) 기반 분리
    chunks = []
    pos = 0
    page = 1
    while pos < len(full):
        chunk = full[pos : pos + CHUNK_SIZE]
        chunks.append((page, chunk))
        pos += CHUNK_SIZE - CHUNK_OVERLAP
        page += 1
    # 2) 임베딩 한번만 계산
    embeddings = []
    for _, txt in chunks:
        resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=txt)
        embeddings.append(resp["data"][0]["embedding"])
    return chunks, embeddings

# — 3) semantic + fuzzy 다단계 검색 —
def retrieve(query: str,
             chunks: List[Tuple[int,str]],
             embeddings: List[List[float]]) -> Tuple[List[int], str]:
    # 3-1) 질문 임베딩
    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=query)
    q_emb = np.array(resp["data"][0]["embedding"])
    # 3-2) 코사인 유사도 계산
    sims = []
    for i, emb in enumerate(embeddings):
        sims.append((i, float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb)*np.linalg.norm(emb)))))
    sims.sort(key=lambda x: x[1], reverse=True)
    # 3-3) top-k 후보 중 fuzzy 필터 후 재정렬
    candidates = []
    for idx, _ in sims[:TOP_K]:
        score = fuzz.token_set_ratio(query, chunks[idx][1])
        if score >= FUZZY_THRESHOLD:
            candidates.append((idx, score))
    if not candidates:
        # fuzzy 못 넘은 경우 semantic top1 만 사용
        candidates = [(sims[0][0], fuzz.token_set_ratio(query, chunks[sims[0][0]][1]))]
    # 최종 1순위
    best_idx = max(candidates, key=lambda x: x[1])[0]
    # top-k 전체 합쳐서 컨텍스트로 전달
    top_idxs = [i for i, _ in sims[:TOP_K]]
    combined = "\n\n---\n\n".join(chunks[i][1] for i in top_idxs)
    pages = [chunks[i][0] for i in top_idxs]
    return pages, combined

# — 4) GPT에게 질문 던지기 —
def ask_gpt(query: str, pages: List[int], context: str) -> str:
    system = (
        "You are an assistant that answers questions *only* from the provided context. "
        "If the answer isn’t in the context, say you couldn’t find it."
    )
    user = (
        f"[=== CONTEXT START (pages {pages}) ===]\n"
        f"{context}\n"
        f"[=== CONTEXT END ===]\n\n"
        f"Question: {query}\n\n"
        "Please answer in one sentence and **quote** the exact sentence from the context as evidence."
    )
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

# — Streamlit UI —
st.title("📘 오픈북 Q&A (임베딩+퍼지 검색)")
st.write("오타·동의어 OK · 오직 주어진 문맥에서 답과 근거 인용")

# 초기 로드
chunks, embeddings = load_and_chunk("pdf_text/your_pdf.txt")

query = st.text_input("❓ 질문을 입력하세요")
if query:
    with st.spinner("검색 중…"):
        pages, ctx = retrieve(query, chunks, embeddings)
    with st.spinner("AI가 답변 생성 중…"):
        answer = ask_gpt(query, pages, ctx[:5000])
    st.subheader("✅ 답변")
    st.write(answer)
