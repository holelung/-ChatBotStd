import requests
import json

# 네이버 개발자 센터에서 발급받은 클라이언트 ID와 시크릿
client_id = "IRaU3KJ02B0myg8nQbKZ"
client_secret = "Ji13BSrbKG"
# 검색어 설정
query = "노트북"

# API 엔드포인트 URL
url = f"https://openapi.naver.com/v1/search/shop.json?query={query}&display=10&start=1&sort=sim"

# 헤더 설정
headers = {
    "X-Naver-Client-Id": client_id,
    "X-Naver-Client-Secret": client_secret
}

# API 요청 보내기
try:
    response = requests.get(url, headers=headers)
    
    # 응답 확인
    response.raise_for_status()  # 오류 발생 시 예외를 발생시킵니다.
    
    data = response.json()
    
    # 검색 결과 출력
    for item in data['items']:
        print(f"상품명: {item['title']}")
        print(f"가격: {item['lprice']}원")
        print(f"쇼핑몰: {item['mallName']}")
        print(f"링크: {item['link']}")
        print("---")
except requests.exceptions.RequestException as e:
    print(f"API 요청 중 오류 발생: {e}")
    print(f"상태 코드: {response.status_code}")
    print(f"응답 내용: {response.text}")
    print(f"요청 URL: {url}")
    print(f"요청 헤더: {headers}")