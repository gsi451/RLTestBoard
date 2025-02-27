#################################
# PPO1번에서 개선하고자 하는 정규화 패턴
#################################

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# sample.csv 파일 읽기
df = pd.read_csv("sample.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# 로그수익률 변환 (첫 번째 값은 NaN이 됨)
for col in ["open", "high", "low", "close"]:
    df[f"log_return_{col}"] = np.log(df[col] / df[col].shift(1))

# 거래량 로그 변환
df["log_volume"] = np.log1p(df["volume"])

# 거래량 표준화 (로그 변환된 값 기준)
scaler = StandardScaler()
df["log_scaled_volume"] = scaler.fit_transform(df[["log_volume"]])

# 로그수익률을 적용한 새로운 가격 데이터 생성 (기준점 대비 상대 변화율을 누적)
df["adj_close"] = df["close"].iloc[0] * np.exp(df["log_return_close"].cumsum())
df["adj_open"] = df["open"].iloc[0] * np.exp(df["log_return_open"].cumsum())
df["adj_high"] = df["high"].iloc[0] * np.exp(df["log_return_high"].cumsum())
df["adj_low"] = df["low"].iloc[0] * np.exp(df["log_return_low"].cumsum())

# 새로운 데이터프레임 생성 (변환된 가격 데이터 사용)
ohlc_df = df[["adj_open", "adj_high", "adj_low", "adj_close", "volume"]].dropna()
ohlc_df.columns = ["open", "high", "low", "close", "volume"]  # 캔들차트 호환을 위해 이름 변경

# 캔들차트 스타일 지정
style = mpf.make_mpf_style(base_mpf_style="charles", rc={"figure.figsize": (10, 8)})

# 📌 4개 차트 (캔들차트, 원본 거래량, 로그 변환된 거래량, 로그수익률 변환된 가격) 시각화
fig, axes = plt.subplots(4, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1, 1]})

# 캔들차트 (원본 가격 데이터)
mpf.plot(
    ohlc_df,
    type="candle",
    ax=axes[0],
    volume=False,
    style=style
)
axes[0].set_title("Candlestick Chart (Original Price Data)")

# 로그수익률 변환된 가격 데이터 (선 그래프)
axes[1].plot(df.index, df["log_return_close"], label="Log Return (Close)", color="orange")
axes[1].set_title("Log Returns (Close)")
axes[1].set_ylabel("Log Return")
axes[1].grid()

# 원본 거래량 (바 차트)
axes[2].bar(df.index, df["volume"], width=0.0005, color="gray", alpha=0.7, align="center")
axes[2].set_title("Original Volume")
axes[2].set_ylabel("Volume")
axes[2].grid()

# 로그 변환 및 표준화된 거래량 (선 그래프)
axes[3].plot(df.index, df["log_scaled_volume"], label="Scaled Log Volume", color="blue")
axes[3].set_title("Log Scaled Volume")
axes[3].set_ylabel("Standardized Volume")
axes[3].grid()

# 레이아웃 조정 및 그래프 출력
plt.tight_layout()
plt.show()
