import os
import streamlit as st
import openai
from rapidfuzz import fuzz

# ❗️ OpenAI 구버전 API 사용
openai.api_key = os.getenv("OPENAI_API_KEY")

# — PDF/TXT 텍스트 로드 —
@st.cache_data
def load_text():
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        return f.read()

# — 페이지(청크) 단위로 분리 —
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
        chunks = []
        step = 2000
        for i in range(0, len(full_text), step):
            chunks.append((i // step + 1, full_text[i : i + step]))
        return chunks

# — 퍼지 매칭으로 가장 관련 페이지 찾기 —
def search_best_page(query: str, pages: list[tuple[int, str]]):
    best_score, best = 0, (None, "")
    for num, txt in pages:
        # partial_ratio: 짧은 query → 긴 txt 중 best substring match
        score = fuzz.partial_ratio(query, txt)
        if score > best_score:
            best_score, best = score, (num, txt)
    return best  # (page_num, content)

# — GPT에게 질문 던지기 —
def ask_gpt(question: str, page_num: int, page_txt: str):
    prompt = f"""아래는 책의 {page_num}쪽(혹은 그에 대응하는 청크) 일부입니다.
이 내용을 바탕으로 질문에 답하고, 답의 근거 문장도 함께 제시하세요.

질문: {question}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# — Streamlit UI —
st.title("📘 오픈북 시험 질문 도우미")
st.write("PDF/TXT에서 가장 관련 있는 페이지를 골라 AI가 답하고 근거를 드려요.")

question = st.text_input("❓ 질문을 입력하세요")

if question:
    full = load_text()
    pages = split_pages(full)
    page_num, page_txt = search_best_page(question, pages)
    if page_txt:
        with st.spinner(f"{page_num}쪽 내용을 분석 중..."):
            # 페이지 텍스트는 1500자까지만 보냅니다
            answer = ask_gpt(question, page_num, page_txt[:1500])
        st.subheader(f"✅ GPT의 답변 (페이지 {page_num})")
        st.write(answer)
    else:
        st.error("관련된 페이지를 찾지 못했습니다.")
