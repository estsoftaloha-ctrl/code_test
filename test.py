import datetime
import requests
from langchain_community.tools import DuckDuckGoSearchRun

# -----------------------------------
# 1. 뉴스 검색 기능 준비
# -----------------------------------

search = DuckDuckGoSearchRun()

# -----------------------------------
# 2. 현재 시간 가져오기
# -----------------------------------

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------
# 3. 뉴스 검색
# -----------------------------------

news_result = search.run("한국 AI 최신 뉴스")

print("검색 결과:")
print(news_result)

# -----------------------------------
# 4. AI에게 전달할 프롬프트 작성
# -----------------------------------

prompt = f"""
너는 사용자 질문에 답변하는 AI 챗봇이야.

현재 시간은 {current_time} 이야.

반드시 한국어로만 답변해.

아래 뉴스 내용을:
1. 핵심 뉴스
2. 관련 기업
3. 기술 트렌드
4. 왜 중요한지

기준으로 10줄 이내로 요약해줘.

뉴스:
{news_result}
"""

# -----------------------------------
# 5. EXAONE 모델 호출
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
# 6. 결과 변환
# -----------------------------------

result = response.json()

# -----------------------------------
# 7. 결과 출력
# -----------------------------------

print("\nEXAONE 뉴스 요약:")
print(result["response"])