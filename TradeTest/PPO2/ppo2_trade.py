"""
1분봉의 값을 실시간 가져오는 코드입니다.
이 코드는 sujip.csv 파일에 계속해서 파일을 기록합니다.

최초 실행시 1440개의 값을 먼저 가져옵니다.
이후 현재 시간을 토대로 해서 1분에 한번씩 1분봉을 가져오게 되어 있습니다.
1분 늦은 봉의 값을 가져옵니다.

이 프로그램은 계속해서 실행되도록 하고 1분봉을 계속해서 수집합니다.
이후 거래를 하는 다른 파이썬 프로그램이 이 파일을 읽어서 거래를 진행 하게 되어 있습니다.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

#########################################
# ① 과거(역사) 데이터 수집
#########################################

# 파일 경로 설정
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 기존 방식
output_file = "sujip.csv"  # 현재 실행 위치에 직접 저장

# Bitget API 설정
url = "https://api.bitget.com/api/v2/spot/market/history-candles"
symbol = "FARTCOINUSDT"
granularity = "1min"

# 시작 시간 설정
now = datetime.now()
if os.path.exists(output_file):
    # 기존 파일이 있는 경우 마지막 데이터의 시간 확인
    df_existing = pd.read_csv(output_file)
    last_timestamp = pd.to_datetime(df_existing['timestamp'].iloc[-1])
    start_time = last_timestamp + timedelta(minutes=1)
else:
    # 파일이 없는 경우 24시간 전부터 시작
    start_time = (now - timedelta(days=1)).replace(second=0, microsecond=0)

# 종료 시간은 현재 시간의 직전 분
end_time = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
print(f"데이터 수집 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 초기 데이터 수집
current_start = start_time
while current_start < end_time:
    current_end = min(current_start + timedelta(hours=1), end_time)
    start_timestamp = int(current_start.timestamp() * 1000)
    end_timestamp = int(current_end.timestamp() * 1000)
    
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "startTime": str(start_timestamp),
        "endTime": str(end_timestamp),
        "limit": "60"  # 한 시간에 60개의 1분 캔들
    }
    
    response = requests.get(url, params=params)
    print(f"API 응답 상태 코드: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        if not data['data']:
            print("받아온 데이터가 비어있습니다.")
            break
        
        print(f"데이터 수신 완료: {len(data['data'])}개의 캔들")
        cleaned_data = [row[:6] for row in data['data']]
        df = pd.DataFrame(cleaned_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') / 1000
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        
        mode = 'a' if os.path.exists(output_file) else 'w'
        header = not os.path.exists(output_file)
        df.to_csv(output_file, mode=mode, header=header)
        print(f"데이터 저장 완료: {current_start.strftime('%Y-%m-%d %H:%M')} ~ {current_end.strftime('%Y-%m-%d %H:%M')}")
        
        current_start = current_end
        time.sleep(0.5)  # API 요청 간격 조절
    else:
        print(f"API 요청 실패: {response.text}")
        break

# 데이터 수집 완료 후 빈 구간 체크 및 채우기
print("빈 구간 체크 중...")
if os.path.exists(output_file):
    df_check = pd.read_csv(output_file)
    df_check['timestamp'] = pd.to_datetime(df_check['timestamp'])
    # 중복된 타임스탬프 제거
    df_check = df_check[~df_check['timestamp'].duplicated()]
    df_check.set_index('timestamp', inplace=True)
    
    # 전체 시간 범위 생성 (1분 간격)
    # 여기서 end_time을 역사 데이터 수집 시 사용한 종료 시각으로 맞추거나,
    # 실시간 수집과 중복되지 않도록 주의해야 함
    full_range = pd.date_range(start=start_time, end=end_time, freq='1min')
    missing_times = full_range.difference(df_check.index)
    
    if len(missing_times) > 0:
        print(f"빈 구간 발견: {len(missing_times)}개의 1분봉 누락")
        
        for missing_time in missing_times:
            start_timestamp = int(missing_time.timestamp() * 1000)
            end_timestamp = int((missing_time + timedelta(minutes=1)).timestamp() * 1000)
            
            params = {
                "symbol": symbol,
                "granularity": granularity,
                "startTime": str(start_timestamp),
                "endTime": str(end_timestamp),
                "limit": "1"
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    cleaned_data = [row[:6] for row in data['data']]
                    df = pd.DataFrame(cleaned_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') / 1000
                    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
                    df.set_index('timestamp', inplace=True)
                    df = df.astype(float)
                    
                    df.to_csv(output_file, mode='a', header=False)
                    print(f"빈 구간 데이터 추가: {missing_time}")
            
            time.sleep(0.5)  # API 요청 간격 조절
        
        print("빈 구간 채우기 완료")
    else:
        print("빈 구간이 없습니다.")




print("모든 데이터 수집 완료. 실시간 데이터 수집을 시작합니다...")

#########################################
# ② 실시간 1분봉 데이터 지속적 수집
#########################################

# 실시간 데이터 수집
BEST_REQUEST_TIME = 5  # 초 단위

print("실시간 데이터 수집 시작...")
while True:
    now = datetime.now()
    current_minute = now.replace(second=0, microsecond=0)
    next_minute = current_minute + timedelta(minutes=1)
    
    # 현재 분의 데이터 확인
    if not os.path.exists(output_file):
        print("데이터 파일이 없습니다.")
        break
        
    df_check = pd.read_csv(output_file)
    df_check['timestamp'] = pd.to_datetime(df_check['timestamp'])
    last_timestamp = df_check['timestamp'].iloc[-1]
    
    # 다음 수집 시간까지 대기
    while True:
        now = datetime.now()
        if now.second == BEST_REQUEST_TIME:
            break
        
        # 남은 시간 표시
        next_collection = now.replace(second=BEST_REQUEST_TIME)
        if now.second > BEST_REQUEST_TIME:
            next_collection += timedelta(minutes=1)
        
        time_left = (next_collection - now).total_seconds()
        print(f"\r다음 데이터 수집까지 {time_left:.1f}초 남음... (현재 시간: {now.strftime('%H:%M:%S')})", end='')
        time.sleep(0.1)
    
    print("\n데이터 수집 시작...")
    
    # 새로운 분봉 데이터 수집
    now = datetime.now()
    target_minute = now.replace(second=0, microsecond=0)  # 현재 분의 데이터를 가져오도록 수정
    
    # 마지막 저장된 데이터와 target_minute이 다른 경우에만 수집
    if last_timestamp != target_minute:
        start_timestamp = int(target_minute.timestamp() * 1000)
        end_timestamp = int((target_minute + timedelta(minutes=1)).timestamp() * 1000)
        
        params = {
            "symbol": symbol,
            "granularity": granularity,
            "startTime": str(start_timestamp),
            "endTime": str(end_timestamp),
            "limit": "1"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['data']:
                cleaned_data = [row[:6] for row in data['data']]
                df = pd.DataFrame(cleaned_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') / 1000
                df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                df.to_csv(output_file, mode='a', header=False)
                print(f"실시간 데이터 저장 완료: {target_minute.strftime('%Y-%m-%d %H:%M')}")
    else:
        print(f"이미 {target_minute.strftime('%Y-%m-%d %H:%M')} 데이터가 존재합니다.")
    
    time.sleep(1)
