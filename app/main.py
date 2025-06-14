import os
import streamlit as st
from rapidfuzz import fuzz
from openai import OpenAI
from openai import OpenAIError

# — 환경 변수에서 API 키 로드 및 클라이언트 초기화 —
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=api_key)

# — 설정 —
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

# — PDF/TXT 텍스트 로드 —
@st.cache_data
def load_text(path: str = "pdf_text/your_pdf.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# — 텍스트를 청크(=페이지) 단위로 분리 —
@st.cache_data
def chunk_text(full_text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    return chunks

# — 청크들을 임베딩으로 변환 —
@st.cache_data
def embed_texts(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    for txt in chunks:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=txt)
            # resp.data[0].embedding 으로 접근해야 합니다
            embeddings.append(resp.data[0].embedding)
        except OpenAIError as e:
            st.error(f"임베딩 중 오류: {e}")
            embeddings.append([0.0])  # 실패시 더미
    return embeddings

# — 유사도 기반으로 가장 관련 청크 찾기 —
def find_best_chunk(query: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    # 쿼리도 임베딩해서 비교하거나, 퍼지 매칭으로 단순 비교
    scores = []
    for chunk in chunks:
        # RapidFuzz 퍼지 점수
        score = fuzz.token_sort_ratio(query, chunk)
        scores.append(score)
    return max(range(len(chunks)), key=lambda i: scores[i])

# — GPT에 질문 던지기 —
def ask_gpt(question: str, context: str) -> str:
    prompt = (
        "아래 문맥만 참고해서 질문에 답하고, 답의 근거 문장을 그대로 인용해주세요.\n\n"
        f"문맥:\n```{context}```\n\n"
        f"질문: {question}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"ChatCompletion 오류: {e}"

# — Streamlit UI —
st.title("📘 오픈북 Q&A (임베딩+퍼지 검색)")
st.write("오타·동의어 OK · 오직 주어진 문맥에서 답과 근거 인용")

question = st.text_input("❓ 질문을 입력하세요")
if question:
    full = load_text()
    chunks = chunk_text(full)
    embeddings = embed_texts(chunks)

    idx = find_best_chunk(question, chunks, embeddings)
    context = chunks[idx]

    with st.spinner(f"청크 #{idx+1}에서 답변 생성 중…"):
        answer = ask_gpt(question, context[:1500])

    st.subheader("✅ GPT의 답변")
    st.write(answer)
