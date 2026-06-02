import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------------
# 1. 삼성전자 주식 데이터 가져오기
# -----------------------------------

stock = yf.download("000660.KS", start="2026-01-01", end="2026-05-22")

# -----------------------------------
# 2. 종가 데이터만 사용
# -----------------------------------

df = stock[["Close"]].copy()

# 날짜 번호 생성
df["Day"] = range(len(df))

# -----------------------------------
# 3. 머신러닝 학습 데이터 준비
# -----------------------------------

X = df[["Day"]]
y = df["Close"]

# -----------------------------------
# 4. 선형 회귀 모델 생성
# -----------------------------------

model = LinearRegression()

# -----------------------------------
# 5. AI 학습
# -----------------------------------

model.fit(X, y)

# -----------------------------------
# 6. 다음 날 주가 예측
# -----------------------------------

next_day = [[len(df)]]

predicted_price = model.predict(next_day)

# 숫자만 추출
predicted_value = predicted_price[0][0]

print(f"하이닉스 다음 날 예상 종가: {predicted_value:.2f} 원")

# -----------------------------------
# 7. 그래프 출력
# -----------------------------------

plt.figure(figsize=(10, 5))

# 실제 데이터
plt.scatter(df["Day"], df["Close"])

# AI 예측 선
plt.plot(df["Day"], model.predict(X))

plt.xlabel("Day")
plt.ylabel("Close Price")

plt.title("Hynix Stock Prediction")

plt.show()