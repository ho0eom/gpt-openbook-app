import os
import streamlit as st
from openai import OpenAI
from rapidfuzz import fuzz
from typing import List, Tuple

# — 환경변수에서 API 키 읽기 —
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 설정
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# — 텍스트 로드 및 청크 분리 + 임베딩 생성 —
@st.cache_data
def load_and_chunk(path: str) -> Tuple[List[str], List[List[float]]]:
    text = open(path, "r", encoding="utf-8").read()
    # 슬라이딩 윈도우 청크
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i : i + CHUNK_SIZE]
        chunks.append(chunk)
    # 임베딩
    embeddings = []
    for txt in chunks:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=txt)
        # --- 여기만 변경 ---
        embeddings.append(resp.data[0].embedding)
    return chunks, embeddings

# — 쿼리 임베딩 생성 & 유사도 상위 K개 찾기 + 퍼지 매칭 필터링 ---
@st.cache_data
def semantic_search(query: str, chunks, embeddings) -> List[Tuple[int,str]]:
    q_emb = client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
    # 코사인 유사도 계산
    from numpy import dot
    from numpy.linalg import norm
    scores = [
        dot(q_emb, emb) / (norm(q_emb)*norm(emb)+1e-8)
        for emb in embeddings
    ]
    # 상위 K 인덱스
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    # 퍼지 매칭으로 추가 후보 걸러내기
    results = []
    for i in top_idxs:
        score = fuzz.partial_ratio(query, chunks[i])
        if score > 50:  # 유사도 임계값
            results.append((i, chunks[i]))
    return results

# — GPT에 질문 던지기 (주어진 문맥에서만 답하도록 프롬프트 고정) —
def ask_gpt(question: str, contexts: List[Tuple[int,str]]) -> str:
    prompt = "아래 청크들만 보고, 질문에 답하세요. 각 답 뒤에 반드시 인용 문장(청크에서 그대로 복붙한 문장)을 제시해야 합니다.\n\n"
    for idx, chunk in contexts:
        prompt += f"[청크 {idx}]\n{chunk}\n\n"
    prompt += f"질문: {question}\n\n답변:"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# — Streamlit UI —
st.title("📘 오픈북 Q&A (임베딩+퍼지 검색)")
st.write("오타·동의어 OK · 오직 주어진 문맥에서 답과 근거 인용")

question = st.text_input("❓ 질문을 입력하세요")

if question:
    chunks, embeddings = load_and_chunk("pdf_text/your_pdf.txt")
    candidates = semantic_search(question, chunks, embeddings)
    if not candidates:
        st.warning("관련 문맥을 찾지 못했습니다.")
    else:
        with st.spinner("AI가 답변을 만들고 있어요..."):
            answer = ask_gpt(question, candidates)
        st.subheader("✅ 답변 및 인용 근거")
        st.write(answer)
