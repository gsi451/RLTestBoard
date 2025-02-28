import pandas as pd
import mplfinance as mpf
import time

# CSV 파일 읽기
data = pd.read_csv('sujip.csv', parse_dates=['timestamp'])

# 최근 144개 데이터 선택
latest_data = data.tail(144)

# 인덱스를 날짜로 설정
latest_data.set_index('timestamp', inplace=True)

# 캔들 차트 그리기
mpf.plot(latest_data, type='candle', volume=True, style='charles', title='Candlestick Chart', ylabel='Price', ylabel_lower='Volume')
