import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# API 키 설정
load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

# 네이버 쇼핑 API 데이터 가져오기
def get_naver_shopping_data(query, display=100):
    url = f"https://openapi.naver.com/v1/search/shop.json?query={query}&display=10&start=1&sort=sim"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json().get('items', [])

# 데이터 수집 및 전처리
queries = ["노트북", "스마트폰", "태블릿"]
all_data = []
for query in queries:
    all_data.extend(get_naver_shopping_data(query, display=100))

df = pd.DataFrame(all_data)
df['text'] = df.apply(lambda row: f"상품명: {row['title']}\n가격: {row['lprice']}원\n브랜드: {row.get('brand', 'N/A')}\n카테고리: {row.get('category1', '')}/{row.get('category2', '')}/{row.get('category3', '')}/{row.get('category4', '')}\n설명: {row.get('description', '')}", axis=1)

# 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(df['text'].tolist(), embeddings)

# LLM 모델 초기화
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# 대화 체인 생성
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 챗봇 함수
def chatbot():
    print("챗봇: 안녕하세요! 네이버 쇼핑 관련 질문에 답변해 드리겠습니다. 종료하려면 '종료'를 입력하세요.")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() == '종료':
            print("챗봇: 감사합니다. 좋은 하루 되세요!")
            break
        
        result = qa_chain({"question": user_input})
        print("챗봇:", result['answer'])

# 챗봇 실행
if __name__ == "__main__":
    chatbot()