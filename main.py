import streamlit as st
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()

# 초기 세션 상태 설정
if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False
if 'show_answer_input' not in st.session_state:
    st.session_state.show_answer_input = True
if 'recomendq' not in st.session_state:
    st.session_state.recomendq = ""


st.title('AI 커뮤니케이션 코치 스픽스')
person = st.text_input('지원자 설명')
jobdescription = st.text_input('채용공고')


if st.button('예상 질문 생성'):
    with st.spinner('질문 생성 중입니다...예상 10초?!'):
        recomendq = chat_model.predict(person +"에 대해서" + jobdescription + "채용공고의 내용을 기반으로 1분동안 답변할만한 면접관의 질문 1개를 만들어줘")
        st.session_state.show_questions = True
        st.session_state.show_answer_input = True

# 예상 질문이 있으면 표시
if st.session_state.show_questions:
    st.write('예상질문:', st.session_state.recomendq)

if st.session_state.show_answer_input:
    question = st.text_input('면접관 질문', value=st.session_state.recomendq if st.session_state.show_questions else "")
    st.text('답변을 입력하세요')
    answer = st.text_area('답변 입력')

    if st.button('분석 시작'):
        with st.spinner('답변 분석 중입니다...최대 1분?!'):
            result = chat_model.predict(person + "이" + question + "에 대한 질문에 대해서 답변으로" + answer +"을 1분 동안 했다. 이 면접 연습에 대해서 커뮤니케이션 코치로 질문에 대한 지원자의 답변에 대해서 질문 의도, 답변 길이, 개선 답변을 제시한다.개선 답변은 [직무역량], [구체적인경험], [방안과 결과], [정리] 내용을 포함한다.")
            st.write('위 질문에 대한 모법 답안은?', result)
else:
    st.write('모범답안을 받으려면 클릭')

if st.button("리셋"):
    st.session_state.show_questions = False
    st.session_state.show_answer_input = True
    st.session_state.recomendq = ""