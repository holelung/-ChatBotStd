import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from flask import Flask, render_template, request, Response
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.callbacks import StreamingStdOutCallbackHandler
import json

app = Flask(__name__)

# API 키 설정
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NAVER_CLIENT_ID = os.getenv('NAVER_CLIENT_ID')
NAVER_CLIENT_SECRET = os.getenv('NAVER_CLIENT_SECRET')

# 네이버 쇼핑 API 데이터 가져오기
def get_naver_shopping_data(query, display=30):
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json().get('items', [])

def validate_and_process_link(item):
    original_link = item['link']
    
    # 네이버 검색 결과 페이지 링크 생성
    search_query = quote(item['title'])
    alternative_link = f"https://search.shopping.naver.com/search/all?query={search_query}"
    return alternative_link

# 상품 정보 포맷팅
def format_product_info(items):
    formatted_items = []
    for item in items:
        link = validate_and_process_link(item)
        formatted_item = (
            f"상품명: {item['title']}\n"
            f"가격: {item['lprice']}원\n"
            f"브랜드: {item.get('brand', 'N/A')}\n"
            f"카테고리: {item.get('category1', '')}/{item.get('category2', '')}\n"
            f"링크: {link}\n"
        )
        formatted_items.append(formatted_item)
    return "\n".join(formatted_items)

# 커스텀 스트리밍 콜백 핸들러
class CustomStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        print(token, end="", flush=True)

# LLM 모델 초기화
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", streaming=True)

# 프롬프트 템플릿 설정
template = """
너는 네이버 쇼핑 도우미야. 사용자의 질문에 대해 제공된 상품 정보를 바탕으로 답변해줘. 
상품 정보에 없는 내용은 추측하지 말고, 정보가 부족하다고 말해줘.
링크를 제공할 때는 링크 상태를 확인하고, 유효하지 않은 경우 대체 링크나 검색 방법을 안내해줘.
여러 상품을 비교해서 설명해줘.

상품 정보:
{product_info}

대화 기록:
{history}

사용자: {human_input}
AI 도우미:
"""

prompt = ChatPromptTemplate.from_template(template)

# 메모리 설정
memory = ConversationBufferMemory(memory_key="history", input_key="human_input")

@app.route('/')
def home():
    return render_template('chat.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # 네이버 쇼핑 API로 상품 정보 가져오기
    items = get_naver_shopping_data(user_message)
    product_info = format_product_info(items)
    
    # 대화 기록 가져오기
    history = memory.load_memory_variables({})["history"]
    
    # 프롬프트 생성
    messages = prompt.format_messages(
        product_info=product_info,
        history=history,
        human_input=user_message
    )

    def generate():
        full_response = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                yield f"data: {json.dumps({'response': full_response})}\n\n"
        
        # 메모리 업데이트
        memory.save_context({"human_input": user_message}, {"output": full_response})

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)