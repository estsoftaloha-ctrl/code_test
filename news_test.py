import requests  # 인터넷 통신을 하기위한 라이브러리
from langchain_community.tools import DuckDuckGoSearchRun  # 구글 검색 같은 기능 가져오기 - 최신 뉴스, 실시간 정보

# -----------------------------------
# 1. 뉴스 검색
# -----------------------------------

search = DuckDuckGoSearchRun() # 검색 툴 초기화

news_result = search.run("AI avatar release funding latest news") # 검색어로 AI Avatar 관련 최신 뉴스 검색

print("검색 결과:") 
print(news_result)

# -----------------------------------
# 2. Llama에게 전달할 프롬프트
# -----------------------------------
# prompt = f""" 문자열 안에 변수 넣기

prompt = f""" 
너는 AI 산업 분석가야.

반드시 한국어로 답변해.

아래 AI Avatar 뉴스를 분석해서:

1. 핵심 뉴스

2. 관련 기업

3. 기술 트렌드

4. 시장 변화

5. 왜 중요한지

한국어로 10줄 요약해줘.

뉴스:
{news_result}
"""

# -----------------------------------
# 3. Ollama API 호출
# -----------------------------------

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "exaone3.5",
        "prompt": prompt,
        "stream": False
    }
)

# -----------------------------------
# 4. 결과 출력
# -----------------------------------

result = response.json()

print("\nLlama 뉴스 요약:")
print(result["response"])