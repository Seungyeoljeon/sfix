import streamlit as st
from langchain.chat_models import ChatOpenAI
chat_model = ChatOpenAI()

# 초기 세션 상태 설정
if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False
if 'show_answer_input' not in st.session_state:
    st.session_state.show_answer_input = True
if 'recomendq' not in st.session_state:
    st.session_state.recomendq = "기본 예상 질문" #초기값


st.title('AI 커뮤니케이션 코치 스픽스')
person = st.text_area('지원자 설명')
jobdescription = st.text_area('채용공고')

st.caption('입력 예시 입니다.')
st.caption('지원자 설명 :저는 컴퓨터 공학을 전공한 신입 개발자입니다. 학교에서는 Python과 Java를 사용하여 여러 프로젝트를 진행했습니다. 또한, 오픈 소스 프로젝트에 참여하여 실제 문제를 해결하는 경험을 했습니다. 팀워크와 커뮤니케이션 능력을 중요하게 생각하며, 늘 새로운 것을 배우고 성장하려고 노력합니다.')
st.caption('채용공고 : 우리 회사는 역동적인 개발 팀을 구성하고 있습니다. 현재 Java와 Python을 주로 사용하는 웹 개발자를 찾고 있습니다. 필수 요건은 다음과 같습니다:')
st.caption('1. 컴퓨터 공학 또는 관련 분야의 학사 이상의 학위 2. Python, Java에 대한 깊은 이해 3. Git과 같은 버전 관리 도구 사용 경험 4. 팀워크와 커뮤니케이션 능력 5. RESTful API 개발 경험 우대사항: 1. 클라우드 서비스(AWS, Azure 등) 사용 경험 2. CI/CD 파이프라인 구축 경험 ')

if st.button('예상 질문 생성'):
    with st.spinner('질문 생성 중입니다...예상 10초?!'):
        st.session_state.recomendq = chat_model.predict(person +"에 대해서" + jobdescription + "채용공고의 내용을 기반으로 1분동안 답변할만한 면접관의 질문 1개를 만들어줘")
        st.session_state.show_questions = True
        st.session_state.show_answer_input = True

# 예상 질문 표시 
if st.session_state.show_questions:
    st.write('예상질문:', st.session_state.recomendq)

if st.session_state.show_answer_input:
    question = st.text_area('면접관 질문', value=st.session_state.recomendq if st.session_state.show_questions else "")
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