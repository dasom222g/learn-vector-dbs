import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from openai import vector_stores

load_dotenv()

if __name__ == '__main__':
    print('hello')
    # =============================== data injection ===============================

    # 1. 랭체인 문서 불러오기
    file_path = '/Users/dasom/somi/github.com/dasom222g/learn-vector-dbs/medium_blog.txt'
    loader = TextLoader(file_path=file_path) # 문서 로더 객체
    document = loader.load() # 랭체인 문서

    # 2. 청킹하기
    chunk_size = 1000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=["\n\n", "\n", " ", ""], chunk_overlap=0) # 텍스트 분할 객체
    chunked_docs = text_splitter.split_documents(document) # 청킹한 데이터
    print(chunked_docs)

    # for i, data in enumerate(chunked_docs):
    #     print(f'{i + 1}번째 청크 사이즈: {len(data.page_content)}')

    # 3. 임베딩 모델의 정보를 가진 임베딩 객체 가져오기
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=os.getenv('OPENAI_API_KEY')) # api_key 옵션 명시 안해줘도 자동인식

    # 4-1. 벡터스토어 생성 및 저장
    vector_store = PineconeVectorStore.from_documents(documents=chunked_docs, embedding=embeddings, index_name=os.getenv('INDEX_NAME'))

    # 4-2. 기존 스토어에 추가
    # vector_store.add_documents(documents=chunked_docs)