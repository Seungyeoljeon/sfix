__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#import
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever

#Stream 받아 줄 Hander 만들기
class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)


# from langchain.llms import CTransformers
chat_model = ChatOpenAI(model="gpt-4", temperature=0)
# llm = CTransformers(
#     model="llama-2-7b-chat.ggmlv3.q2_K.bin",
#     model_type="llama"
# )
button(username="sfix", floating=True, width=221)


st.title('AI 면접 코치 스픽스!')
st.caption('입력 예시 입니다.')
st.caption('자기 소개 :저는 컴퓨터 공학을 전공한 신입 개발자입니다. 학교에서는 Python과 Java를 사용하여 여러 프로젝트를 진행했습니다. 또한, 오픈 소스 프로젝트에 참여하여 실제 문제를 해결하는 경험을 했습니다. 팀워크와 커뮤니케이션 능력을 중요하게 생각하며, 늘 새로운 것을 배우고 성장하려고 노력합니다.')
st.caption('상황 설명 : 아래 채용 공고를 읽고 면접을 가는 상황입니다. 우리 회사는 역동적인 개발 팀을 구성하고 있습니다. 현재 Java와 Python을 주로 사용하는 웹 개발자를 찾고 있습니다. 필수 요건은 다음과 같습니다:')
st.caption('1. 컴퓨터 공학 또는 관련 분야의 학사 이상의 학위 2. Python, Java에 대한 깊은 이해 3. Git과 같은 버전 관리 도구 사용 경험 4. 팀워크와 커뮤니케이션 능력 5. RESTful API 개발 경험 우대사항: 1. 클라우드 서비스(AWS, Azure 등) 사용 경험 2. CI/CD 파이프라인 구축 경험 ')

# 초기 세션 상태 설정
if 'show_questions' not in st.session_state:
    st.session_state.show_questions = False
if 'show_answer_input' not in st.session_state:
    st.session_state.show_answer_input = True
if 'recomendq' not in st.session_state:
    st.session_state.recomendq = "기본 예상 질문" #초기값

#제목
st.write("자기소개서 업로드시 상세한 분석이 가능합니다.")

#uploader
uploaded_file = st.file_uploader("자기소개서를 PDF 또는 TXT 파일로 업로드 해주세요",type=['pdf', 'txt'])
st.write("___")

# Before the function definition
if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
else:
    file_extension = ""

def file_to_document(uploaded_file):
    # PDF 파일의 경우
    if file_extension == '.pdf':
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath,"wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages

    # TXT 파일의 경우
    elif file_extension == '.txt':
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath,"wb") as f:
            f.write(uploaded_file.getvalue())
        loader = TextLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages

    else:
        raise ValueError("PDF, TXT만 가능합니다..")

#업로드시 동작 코드
if st.button('자기소개서 기반 질문 생성'):
    with st.spinner('잠시만 기다려주세요...'):
        if uploaded_file is not None:
            pages = file_to_document(uploaded_file)

            #Split
            text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
                chunk_size = 100,
                chunk_overlap  = 20,
                length_function = len,
                is_separator_regex = False,
            )
            texts = text_splitter.split_documents(pages)

            #Embedding

            embeddings_model = OpenAIEmbeddings()
            
        
            #load it into Chroma
            data = Chroma.from_documents(texts,embeddings_model)

            # # summurize texts
            # chain = load_summarize_chain(chat_model, chain_type="stuff")
            # docs = chain.run(texts)

            #Stream 받아 줄 Hander 만들기
            from langchain.callbacks.base import BaseCallbackHandler
            class StreamHandler(BaseCallbackHandler):
                def __init__(self, container, initial_text=""):
                    self.container = container
                    self.text=initial_text
                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    self.text+=token
                    self.container.markdown(self.text)

            #자기소개서 요약

            st.header("자기소개서 요약")

            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)

            # Define prompt
            prompt_template = """아래 내용에 대한 2000 자 이내 요약을 한국어로 제공하세요:
            "{text}"
            요약:"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Define LLM chain
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", streaming=True, callbacks=[stream_hander])
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            summary=stuff_chain.run(texts)
                    

            personaq ="위 자기소개서 요약을 읽고 면접관 입장에서 지원자에 대한 질문을 만들어주세요"
            qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=data.as_retriever())
            result = qa_chain({"query" : summary + personaq})
            st.write(result["result"])
            st.write("[AI커뮤니케이션 코치 스픽스와 함께 스피치  모임 하기](https://open.kakao.com/o/ghKHP4If)")
        else:
            st.warning("자기소개서를 업로드해주세요")




col1, col2 = st.columns(2)
with col1:
    person = st.text_area('자기 소개', help='자기소개를 업로드하지 않으셨다면 여기에 간단하게 적어주세요.')

with col2:
    description = st.text_area('상황 설명', help='어떤 상황인지 설명해주세요')


if st.button('예상 질문 생성'):
    with st.spinner('질문 생성 중입니다...예상 10초?!'):
        st.session_state.recomendq = chat_model.predict(person +"은 제출된 자기소개서이다." + description + "인 상황을 기반으로 1분동안 답변할만한 상대방의 질문 1개와 예상 답변을 만들어줘")
        st.session_state.show_questions = True
        st.session_state.show_answer_input = True

# 예상 질문 표시
if st.session_state.show_questions:
    st.write('예상질문:', st.session_state.recomendq)

if st.session_state.show_answer_input:
    question = st.text_area('질문', value=st.session_state.recomendq if st.session_state.show_questions else "")
    st.text('답변을 입력하세요')
    answer = st.text_area('답변 입력')

    if st.button('분석 시작'):
        with st.spinner('답변 분석 중입니다...최대 1분?!'):
            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)
            chat_model = ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True, callbacks=[stream_hander])
            result = chat_model.predict("이용자가" +person + "와" + description + "을 바탕으로" + question + "에 대해" + answer + "으로 답변했습니다. 이 답변에 대해 명확성, 구조화, 적절한 길이, 문법과 언어 사용, 감정의 표현, 컨텍스트 이해, 그리고 응용 및 예시 사용의 관점에서 분석해주세요." +  "위 답변에 대해서 답변 개선 안을 보여주세요.")
            st.write('위 질문에 대한 모범 답변은?', result)
else:
    st.write('모범답변을 받으려면 클릭')


if st.button("리셋"):
    st.session_state.show_questions = False
    st.session_state.show_answer_input = True
    st.session_state.recomendq = ""


import openai
import streamlit as st


st.title("💬 모의 면접하기")
st.caption("🚀 스픽스 모의 면접관입니다.")

# 사용자로부터 person과 description을 입력받는 코드가 필요합니다.
person = st.text_input("자기 소개")
description = st.text_input("상황 설명")

if st.button('입력된 내용 기반 모의 면접 시작'):
    interveiwer = person + description + "위 내용을 참고해서 전문 면접관 역할을 수행한다. 답변은 한글로 한다. 이제 '안녕하세요. 면접을 시작하겠습니다.'라는 말로 면접을 바로 시작한다."
    st.session_state["messages"] = [{"role": "user", "content": interveiwer}]
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "user", "content": "전문 면접관 역할을 수행한다. 답변은 한글로 한다. 이제 '안녕하세요. 면접을 시작하겠습니다.'라는 말로 면접을 바로 시작한다."}]

# 메시지 출력
for message in st.session_state.get("messages", []):
    st.chat_message(message["role"]).write(message["content"])

# 사용자로부터 입력을 받습니다.
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 챗봇의 응답을 생성합니다.

    response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    


# if prompt := st.chat_input():
    
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     st.chat_message("assistant").write(msg.content)