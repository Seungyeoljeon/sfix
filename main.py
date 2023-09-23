# from dotenv import load_dotenv
# load_dotenv()
import streamlit as st
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()

st.title('AI 커뮤니케이션 코치 스픽스')
person = st.text_input('지원자 설명')
jobdescription = st.text_input('채용공고')


if st.button('예상 질문 생성'):
    with st.spinner('질문 생성 중입니다...예상 10초?!'):
        recomendq = chat_model.predict(person +"에 대해서" + jobdescription + "채용공고의 내용을 기반으로 1분동안 답변할만한 면접관의 질문 1개를 만들어줘")
        st.write('예상질문 :',recomendq)
    question = recomendq
    st.text('답변을 입력하세요')
    answer = st.text_input('답변 입력')
else:
    st.write('질문을 입력하세요')
    question = st.text_input('면접관 질문')
    st.text('답변을 입력하세요')
    answer = st.text_input('답변 입력')


if st.button('분석 시작'):
    with st.spinner('답변 분석 중입니다...최대 1분?!'):
        result = chat_model.predict(person + "이" + question + "에 대한 질문에 대해서 답변으로" + answer +"을 1분 동안 했다. 이 면접 연습에 대해서 커뮤니케이션 코치로 질문에 대한 지원자의 답변에 대해서 질문 의도, 답변 길이, 개선 답변을 제시한다.개선 답변은 [직무역량], [구체적인경험], [방안과 결과], [정리] 내용을 포함한다.")
        st.write('위 질문에 대한 모법 답안은?', result)
else:
    st.write('모범답안을 받으려면 클릭')

st.button("리셋", type="primary")