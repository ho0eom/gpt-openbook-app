import os
import streamlit as st
import openai

# — 환경변수에서 API 키 설정 —
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
        # 페이지 정보 없으면 2000자씩 잘라서 chunk 번호 부여
        chunks = []
        step = 2000
        for i in range(0, len(full_text), step):
            chunks.append((i // step + 1, full_text[i : i + step]))
        return chunks

# — 키워드 기반으로 가장 관련 페이지 찾기 —
def search_best_page(query: str, pages: list[tuple[int, str]]):
    best_score = 0
    best_page = (None, "")
    tokens = [t for t in query.lower().split() if t]
    for num, txt in pages:
        score = sum(txt.lower().count(tok) for tok in tokens)
        if score > best_score:
            best_score = score
            best_page = (num, txt)
    return best_page  # (page_num, content)

# — GPT에게 질문 던지기 —
def ask_gpt(question: str, page_num: int, page_txt: str):
    prompt = f"""아래는 문서의 {page_num}쪽 일부입니다. 
이 내용을 바탕으로 질문에 답하고, 답의 근거가 되는 문장도 **그대로 인용**해서 제시해 주세요.

{page_txt}

질문: {question}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

# — Streamlit UI —
st.set_page_config(page_title="📘 오픈북 Q&A", layout="wide")
st.title("📘 오픈북 Q&A")
st.write("PDF/TXT에서 가장 관련 있는 페이지를 찾아 AI가 답하고 근거를 인용해 드려요.")

question = st.text_input("❓ 질문을 입력하세요")

if question:
    full = load_text()
    pages = split_pages(full)
    page_num, page_txt = search_best_page(question, pages)
    if page_txt:
        with st.spinner(f"{page_num}쪽 내용을 분석 중…"):
            # 긴 페이지 텍스트는 최대 1500자만 전달
            answer = ask_gpt(question, page_num, page_txt[:1500])
        st.subheader(f"✅ GPT의 답변 (페이지 {page_num})")
        st.write(answer)
    else:
        st.error("관련된 페이지를 찾지 못했습니다.")
