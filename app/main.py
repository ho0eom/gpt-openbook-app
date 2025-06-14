# app/main.py
import os
import streamlit as st
from openai import OpenAI
from rapidfuzz import process, fuzz

# 1) OpenAI v1 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) PDF/TXT 텍스트 로드
@st.cache_data
def load_text():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        return f.read()

# 3) 페이지(청크) 단위로 분리
@st.cache_data
def split_pages(full_text: str):
    if "[Page " in full_text:
        raw = full_text.split("[Page ")
        pages = []
        for chunk in raw:
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                num_str, content = chunk.split("]", 1)
                pages.append((int(num_str), content.strip()))
            except ValueError:
                continue
        return pages
    else:
        # 고정 길이 청크 분리
        chunks = []
        step = 2000
        for i in range(0, len(full_text), step):
            chunks.append((i // step + 1, full_text[i : i + step]))
        return chunks

# 4) RapidFuzz 로 유사도 기준 상위 페이지 찾기
def search_best_page(query: str, pages: list[tuple[int,str]]):
    # 텍스트 리스트만 추출
    texts = [txt for _, txt in pages]
    # 유사도 상위 3개 청크 후보
    results = process.extract(
        query, texts,
        scorer=fuzz.token_set_ratio,
        limit=3
    )
    # 최고점 청크 인덱스
    best_idx = results[0][2]
    return pages[best_idx]  # (page_num, content)

# 5) GPT에 질문 보내기
def ask_gpt(question: str, page_num: int, page_txt: str):
    prompt = f"""아래는 책의 {page_num}쪽(혹은 청크) 일부입니다.
이 내용을 바탕으로 질문에 답하고, 답의 근거 문장도 함께 제시하세요.

{page_txt}

질문: {question}
"""
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":prompt}],
    )
    return resp.choices[0].message.content.strip()

# 6) Streamlit UI
st.title("📘 오픈북 시험 질문 도우미")
st.write("PDF/TXT에서 가장 관련 있는 페이지를 골라 AI가 답하고 근거를 드려요.")

question = st.text_input("❓ 질문을 입력하세요")

if question:
    full = load_text()
    pages = split_pages(full)
    page_num, page_txt = search_best_page(question, pages)
    if page_txt:
        with st.spinner(f"{page_num}쪽 내용을 분석 중..."):
            # 텍스트 길이가 너무 길면 앞부분만
            snippet = page_txt[:1500]
            answer = ask_gpt(question, page_num, snippet)
        st.subheader(f"✅ GPT의 답변 (페이지 {page_num})")
        st.write(answer)
    else:
        st.error("관련된 페이지를 찾지 못했습니다.")
