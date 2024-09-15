# TrandNavigator

## ChatBot1 설명 및 작동과정
1. 초기 설정 및 라이브러리 임포트:
    - 필요한 라이브러리들을 임포트합니다.
    - `dotenv`를 사용해 환경 변수를 로드합니다.
2. API 키 설정:
    - `.env` 파일에서 OpenAI와 네이버 API 키를 가져옵니다.
3. 네이버 쇼핑 API 데이터 수집:
    - `get_naver_shopping_data` 함수는 네이버 쇼핑 API를 호출하여 상품 정보를 가져옵니다.
    - "노트북", "스마트폰", "태블릿"에 대한 정보를 각각 100개씩 수집합니다.
4. 데이터 전처리:
    - 수집된 데이터를 pandas DataFrame으로 변환합니다.
    - 각 상품 정보를 문자열 형태로 구조화하여 'text' 열에 저장합니다.
5. 벡터 저장소 생성:
    - OpenAIEmbeddings를 사용하여 각 상품 설명을 벡터로 변환합니다.
    - FAISS를 사용하여 이 벡터들의 인덱스를 생성합니다. 이는 빠른 유사성 검색을 가능하게 합니다.
6. 언어 모델 초기화:
    - ChatOpenAI 모델을 초기화합니다 (GPT-3.5-turbo 사용).
7. 대화 체인 생성:
    - ConversationBufferMemory를 사용하여 대화 기록을 저장합니다.
    - ConversationalRetrievalChain을 생성합니다. 이는 사용자 질문, 관련 상품 정보, 이전 대화 기록을 종합하여 응답을 생성합니다.
8. 챗봇 함수:
    - 사용자 입력을 받고, 'qa_chain'을 사용하여 응답을 생성합니다.
    - 사용자가 '종료'를 입력할 때까지 대화를 계속합니다.

### 작동 과정:

1. 사용자가 질문을 입력합니다.
2. 질문은 벡터로 변환되어 FAISS 인덱스에서 가장 관련성 높은 상품 정보를 검색합니다.
3. 검색된 정보, 사용자의 질문, 이전 대화 기록이 ChatGPT 모델에 입력됩니다.
4. 모델은 이 정보를 바탕으로 적절한 응답을 생성합니다.
5. 생성된 응답이 사용자에게 출력됩니다.

이 과정이 반복되면서 챗봇은 네이버 쇼핑 데이터를 기반으로 사용자와 대화를 나눕니다. 벡터 검색을 통해 관련 정보를 빠르게 찾고, ChatGPT의 언어 이해 능력을 활용하여 자연스러운 대화를 생성합니다.

## ChatBot2 설명 및 작동과정
1. 라이브러리 및 API 키 설정:
    - 필요한 라이브러리들을 임포트합니다.
    - `dotenv`를 사용하여 환경 변수에서 API 키를 로드합니다.
2. 네이버 쇼핑 API 함수:
    - `get_naver_shopping_data` 함수는 네이버 쇼핑 API를 호출하여 상품 정보를 가져옵니다.
    - 검색어와 표시할 상품 수를 파라미터로 받습니다.
3. 상품 정보 포맷팅:
    - `format_product_info` 함수는 API에서 받은 상품 정보를 읽기 쉬운 형식으로 변환합니다.
4. LLM 모델 초기화:
    - ChatOpenAI 모델을 초기화합니다. 여기서는 GPT-3.5-turbo를 사용합니다.
5. 프롬프트 템플릿:
    - 챗봇의 응답을 생성하기 위한 프롬프트 템플릿을 정의합니다.
    - 상품 정보, 대화 기록, 사용자 입력을 포함합니다.
6. 메모리 설정:
    - `ConversationBufferMemory`를 사용하여 대화 기록을 저장합니다.
7. RunnableSequence 설정:
    - `get_product_info` 함수는 사용자 입력을 바탕으로 상품 정보를 가져옵니다.
    - RunnableSequence를 사용하여 상품 정보 조회, 대화 기록 로드, 프롬프트 생성, LLM 실행, 결과 파싱 등의 과정을 연결합니다.
8. 챗봇 함수:
    - 사용자 입력을 받고, RunnableSequence를 실행하여 응답을 생성합니다.
    - 생성된 응답을 출력하고, 대화 내용을 메모리에 저장합니다.
9. 메인 실행:
    - 스크립트가 직접 실행될 때 챗봇 함수를 호출합니다.

이 코드의 핵심은 사용자 입력에 따라 실시간으로 네이버 쇼핑 API를 호출하고, 그 결과를 LLM에 제공하여 맥락에 맞는 응답을 생성하는 것입니다. 또한 대화 기록을 유지하여 이전 대화 내용을 고려한 응답을 생성할 수 있습니다.

이 구조는 유연성이 높아 다른 API나 데이터 소스로 쉽게 확장할 수 있으며, 프롬프트 템플릿을 수정하여 챗봇의 개성이나 응답 스타일을 쉽게 변경할 수 있습니다.

1. RunnableSequence 설정:

RunnableSequence는 여러 단계의 작업을 연결하여 하나의 실행 가능한 파이프라인을 만드는 LangChain의 기능입니다. 이를 사용하는 주요 이유는 다음과 같습니다:

a. 모듈성: 각 단계를 독립적으로 정의하고 조합할 수 있어 코드의 재사용성과 유지보수성이 향상됩니다.
b. 유연성: 필요에 따라 단계를 쉽게 추가, 제거, 수정할 수 있습니다.
c. 명확성: 복잡한 프로세스를 논리적 단계로 나누어 표현할 수 있어 코드의 가독성이 높아집니다.

코드에서 RunnableSequence 설정:

```python
def get_product_info(input_dict):
    query = input_dict["human_input"]
    items = get_naver_shopping_data(query)
    return format_product_info(items)

chain = (
    RunnablePassthrough.assign(
        product_info=RunnableLambda(get_product_info),
        history=RunnableLambda(lambda x: memory.load_memory_variables({})["history"])
    )
    | prompt
    | llm
    | StrOutputParser()
)

```

이 설정은 다음 단계로 구성됩니다:

1. 사용자 입력을 받아 상품 정보를 조회합니다 (`get_product_info`).
2. 대화 기록을 메모리에서 로드합니다.
3. 상품 정보와 대화 기록을 포함한 프롬프트를 생성합니다.
4. 생성된 프롬프트를 LLM에 입력하여 응답을 생성합니다.
5. 생성된 응답을 문자열로 파싱합니다.
6. 메모리 설정:

메모리 관리는 챗봇이 이전 대화 내용을 기억하고 맥락을 유지하는 데 중요합니다. 여기서는 `ConversationBufferMemory`를 사용합니다.

```python
memory = ConversationBufferMemory(memory_key="history", input_key="human_input")

```

이 설정의 상세 내용은 다음과 같습니다:

a. `memory_key="history"`: 메모리에서 대화 기록을 가져올 때 사용할 키를 지정합니다. 이 키는 프롬프트 템플릿에서 `{history}`로 참조됩니다.

b. `input_key="human_input"`: 사용자 입력을 저장할 때 사용할 키를 지정합니다. 이는 메모리에 저장될 때 사용자 입력을 식별하는 데 사용됩니다.

메모리 사용:

1. 대화 기록 로드:
    
    ```python
    history=RunnableLambda(lambda x: memory.load_memory_variables({})["history"])
    
    ```
    
    이 부분은 메모리에서 대화 기록을 로드하여 RunnableSequence에 제공합니다.
    
2. 대화 내용 저장:
    
    ```python
    memory.save_context({"human_input": user_input}, {"output": response})
    
    ```
    
    이 부분은 챗봇 함수 내에서 사용자 입력과 AI의 응답을 메모리에 저장합니다.
    

이러한 메모리 관리 방식을 통해 챗봇은 이전 대화 내용을 고려하여 더 자연스럽고 맥락에 맞는 응답을 생성할 수 있습니다. 예를 들어, 사용자가 이전에 언급한 제품이나 선호도에 대해 기억하고 이를 바탕으로 응답할 수 있습니다.

이렇게 구성된 RunnableSequence와 메모리 관리는 챗봇의 응답 생성 과정을 체계적으로 구조화하고, 대화의 연속성을 유지하는 데 도움을 줍니다.