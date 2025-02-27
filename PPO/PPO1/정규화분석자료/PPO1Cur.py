#################################
# PPO1번의 현재 정규화 패턴
#################################

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# sample.csv 파일 읽기
df = pd.read_csv("sample.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# ✅ Feature Columns 정의
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]

# ✅ 거래량 로그 변환
df["volume"] = np.log1p(df["volume"])

# ✅ 표준화 적용
scaler = StandardScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

# ✅ 캔들차트 스타일 지정
style = mpf.make_mpf_style(base_mpf_style="charles", rc={"figure.figsize": (12, 10)})

# ✅ 캔들차트 및 추가적인 차트 설정
fig, axes = plt.subplots(4, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1, 1]})

# 📌 1) 캔들차트 (정규화된 가격)
mpf.plot(
    df,
    type="candle",
    ax=axes[0], 
    volume=False,
    style=style
)
axes[0].set_title("Candlestick Chart (Standardized Price)")

# 📌 2) 로그수익률 변환된 가격 데이터 (선 그래프)
for col in ["open", "high", "low", "close"]:
    df[f"log_return_{col}"] = np.log(df[col] / df[col].shift(1))
axes[1].plot(df.index, df["log_return_close"], label="Log Return (Close)", color="orange")
axes[1].set_title("Log Returns (Close)")
axes[1].set_ylabel("Log Return")
axes[1].grid()

# 📌 3) 원본 거래량 (바 차트)
axes[2].bar(df.index, np.expm1(df["volume"]), width=0.0005, color="gray", alpha=0.7, align="center")
axes[2].set_title("Original Volume (Reversed Log)")
axes[2].set_ylabel("Volume")
axes[2].grid()

# 📌 4) 표준화된 거래량 (선 그래프)
axes[3].plot(df.index, df["volume"], label="Standardized Volume", color="blue")
axes[3].set_title("Standardized Volume")
axes[3].set_ylabel("Standardized Volume")
axes[3].grid()

# ✅ 그래프 출력
plt.tight_layout()
plt.show()
