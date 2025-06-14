# force redeploy

import streamlit as st
import os
from openai import OpenAI

print("✅ 앱 실행 시작됨")

# ✅ OpenAI 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ PDF 텍스트 불러오기
try:
    with open("pdf_text/your_pdf.txt", "r", encoding="utf-8") as f:
        pdf_text = f.read()
    print("📄 PDF 텍스트 불러오기 성공")
except FileNotFoundError:
    st.error("❌ 텍스트 파일을 찾을 수 없습니다. 'pdf_text/your_pdf.txt' 경로를 확인하세요.")
    st.stop()

# ✅ 페이지별로 텍스트 나누기
pages = pdf_text.split("[Page ")
pages = [p for p in pages if p.strip()]
print(f"📑 총 {len(pages)}개 페이지 불러옴")

# ✅ 검색 함수 정의
def search_best_page(query):
    best_score = 0
    best_page = None
    best_page_num = 0

    for page in pages:
        try:
            page_num = int(page.split("]")[0])
            content = page.split("]")[1]
            score = sum([content.lower().count(q.lower()) for q in query.split()])
            if score > best_score:
                best_score = score
                best_page = content
                best_page_num = page_num
        except:
            continue

    return best_page, best_page_num

# ✅ Streamlit UI 시작
st.title("📘 오픈북 시험 도우미")
st.caption("PDF 내용을 기반으로 질문에 대한 답을 GPT가 찾아줍니다.")
question = st.text_input("❓ 궁금한 내용을 입력하세요:")

if question:
    print("❓ 질문 입력됨:", question)
    page_text, page_num = search_best_page(question)

    if page_text:
        prompt = f"""아래는 어떤 책의 일부입니다. 이 내용을 바탕으로 질문에 답하고, 답의 근거가 되는 문장도 함께 제시하세요.

내용 (책 {page_num}쪽 일부):
{page_text[:1500]}

질문:
{question}
"""
        print("🧠 GPT에 보낼 프롬프트 준비됨")

        with st.spinner("GPT가 답변을 생성 중입니다..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt.encode('utf-8').decode()}]
                )
                answer = response.choices[0].message.content
                st.success(f"✅ GPT의 답변 (📄 {page_num}페이지):")
                st.write(answer)
                print("✅ GPT 응답 완료")
            except Exception as e:
                st.error(f"❌ GPT 호출 오류: {str(e)}")
                print("❌ GPT 호출 실패:", str(e))
    else:
        st.warning("❌ 관련된 페이지를 찾지 못했습니다.")
        print("⚠️ 검색 결과 없음")
