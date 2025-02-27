import pandas as pd
import mplfinance as mpf

# sample.csv 파일 읽기
df = pd.read_csv("sample.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# 캔들차트 스타일 지정
style = mpf.make_mpf_style(base_mpf_style="charles", rc={"figure.figsize": (10, 6)})

# 캔들차트 그리기 (거래량 포함)
mpf.plot(
    df,
    type="candle",  # 캔들차트
    volume=True,    # 거래량 바 차트 추가
    style=style,    # 스타일 적용
    title="OHLC Candlestick Chart with Volume",
    ylabel="Price",
    ylabel_lower="Volume",
    datetime_format="%H:%M",
    tight_layout=True
)
