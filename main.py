import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from openai import responses

# 환경변수 로드
load_dotenv()

# query = 'What is pinecone in machine learning? 한국어로 답변하세요'
query = '이재명의 슬로건과 정치적 배경 등 아는대로 알려주세요'


def basic_rag():
    # 1. 벡터스토어 초기화(index name, embeddigs)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv('INDEX_NAME'))
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv('INDEX_NAME_PRESIDENT'))

    # 2. langchain-ai/retrieval-qa-chat 프롬프트 로드
    prompt = hub.pull('langchain-ai/retrieval-qa-chat')  # input_variables=['context', 'input']
    print(f'prompt: \n {prompt}')

    # 3. 체인 설정
    # create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # 3-1. 체인 생성 (초기화)
    llm = ChatOpenAI(model='gpt-4o-mini')

    # 유사도 검색
    search_docs = vectorstore.similarity_search(query=query, k=3)
    # 검색된 문서들을 하나로 합칠 객체 정의
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    # 3-2. 체인 실행 (이때 문서 합쳐짐)
    # ⭐️방법 1
    response_stuff = combine_docs_chain.invoke({
        "context": search_docs, # 이때 여러개의 문서 함쳐짐
        "input": query
    })

    print(f'🔥🔥최종답변 combine_docs_chain: {response_stuff}') # ⭐⭐문서에 fit한 답변 - str

    # ⭐️방법 2
    # 4. 완전한 rag체인 생성 및 질문

    # 검색기 준비
    retrieval = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever=retrieval, combine_docs_chain=combine_docs_chain)
    response_retrieval = retrieval_chain.invoke({
        'input': query # 질문만 넘기면 끝
    })

    print(f'🔥🔥최종답변 retrieval_chain: {response_retrieval.get("answer")}')  # ⭐⭐문서에 fit한 답변 - dict(input, context, answer)



def lcel_rag():
    print('...')
    # 새로운 프롬프트 템플릿
    # template = """다음 컨텍스트를 사용해서 마지막 질문에 한글로 답하세요.
    # 답을 모르면 모른다고 하고, 답을 지어내려고 하지 마세요.
    # 최대 세 문장으로 답하고 가능한 한 간결하게 유지하세요.
    # 답변 마지막에는 항상 "질문해 주셔서 감사합니다!"라고 말하세요.
    #
    # {context}
    #
    # 질문: {question}
    #
    # 도움이 되는 답변:(한글로 친절하게 답변)"""

def search_without_rag():
    prompt = PromptTemplate.from_template(query)
    llm = ChatOpenAI(model='gpt-4o-mini')
    chain = prompt | llm
    response = chain.invoke(input={})
    print(f'🔥🔥최종 답변(rag없이): {response}') # ⭐일반적 답변



if __name__ == '__main__':
    # print('rag없이 기본 질문하기..')
    # search_without_rag()

    print('basic_rag 검색하기..')
    basic_rag()

    print('LCEL방식으로 검색하기..')
    # basic_rag()


