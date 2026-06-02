import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# 1. 예제 주가 데이터 만들기
# -----------------------------

data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "price": [100, 105, 108, 115, 120, 125, 130]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. 학습 데이터 준비
# -----------------------------

X = df[["day"]]
y = df["price"]

# -----------------------------
# 3. AI 모델 만들기
# -----------------------------

model = LinearRegression()

# -----------------------------
# 4. AI 학습시키기
# -----------------------------

model.fit(X, y)

# -----------------------------
# 5. 미래 주가 예측
# -----------------------------

future_day = [[8]]

predicted_price = model.predict(future_day)

print(f"8일차 예상 주가: {predicted_price[0]:.2f}")

# -----------------------------
# 6. 그래프로 보기
# -----------------------------

plt.scatter(df["day"], df["price"])

plt.plot(df["day"], model.predict(X))

plt.xlabel("Day")
plt.ylabel("Price")

plt.show()