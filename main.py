import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# 환경변수 로드
load_dotenv()


def basic_rag():
    # 1. 벡터스토어 초기화(index name, embeddigs)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vetorstor = PineconeVectorStore(embedding=embeddings, index_name=os.getenv('INDEX_NAME'))

    # 2. langchain-ai/retrieval-qa-chat 프롬프트 로드
    prompt = hub.pull('langchain-ai/retrieval-qa-chat')  # input_variables=['context', 'input']
    print(f'prompt: \n {prompt}')

    # 3. 체인 설정
    # create_stuff_documents_chain: 검색된 문서들을 하나로 합쳐서 LLM에게 전달하는 체인을 만드는 함수
    # create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # 3-1. 체인 생성 (초기화)
    llm = ChatOpenAI(model='gpt-4o-mini')

    # 🔥🔥🔥🔥여기부터
    # combined_docs = create_stuff_documents_chain(llm=llm)

    # 3-2. 체인 실행 (이때 문서 합쳐짐)

    # 4. 완전한 rag체인 생성 및 질문

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


if __name__ == '__main__':
    print('basic_rag 검색하기..')
    basic_rag()

    print('LCEL방식으로 검색하기..')
    # basic_rag()


