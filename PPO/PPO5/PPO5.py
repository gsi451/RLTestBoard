import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.callbacks import BaseCallback
import torch
from torch.utils.tensorboard import SummaryWriter
import mplfinance as mpf  # 캔들차트 스타일 지정용 (필요시 사용)
from sklearn.preprocessing import RobustScaler

############################
# ppo 강화학습
# History
"""
PPO1 - 기본모델
 |
PPO2 - 표준화 오류 수정
 | 
 |- PPO3 - 표준화를 RobustScaler 로 사용
 |- PPO4 - 표준화를 MinMaxScaler 로 사용
 |- PPO5 - 표준화를 PPO3 + 파생변수 로 사용
"""

# PPO2
# PPO1에서 표준화 부분의 요류가 있어서 수정함

# 원본 CSV 파일에서 데이터를 읽고 날짜에 따라 train/validation/test 데이터로 분할합니다.
# 각 데이터셋에 대해 가격의 로그수익률를 계산하고, 누적합으로 조정 가격을 생성하며, 거래량은 로그 변환 후 훈련 데이터를 기준으로 표준화합니다.
# 최종적으로 조정된 가격(시가, 고가, 저가, 종가)과 표준화된 거래량을 FEATURE_COLUMNS에 맞게 재설정하여 강화학습 환경에 입력합니다.
# 나머지 강화학습 환경, 하이퍼파라미터, 학습 및 검증/테스트 과정은 기존 코드와 동일하게 사용합니다.

# PPO3 (학습완료) - 결과가 좋아 보임
# volume_scaler = StandardScaler() 이 부분을 대신해서
# volume_scaler = RobustScaler() 를 사용해서 테스트 한다.

# PPO4 (학습완료) - 결과가 처참함
# volume_scaler = StandardScaler() 이 부분을 대신해서
# volume_scaler = MinMaxScaler() 를 사용해서 테스트 한다.

# PPO5 (학습중)
# PPO3 의 학습 코드에서 거래량 부분을 로그변환후 파생변수를 합한다.
# 파생 변수 계산: volume_ratio를 계산하기 위해 이동 평균을 구한 후, 로그변환 거래량과의 비율을 계산합니다.
# 스케일링: volume_ratio에 대해 RobustScaler를 사용하여 scaled_volume_ratio를 생성합니다.
# 최종 데이터프레임 생성: finalize_df 함수에서 조정 가격과 scaled_volume_ratio를 선택하여 강화학습 환경에 입력될 데이터셋을 구성합니다.
############################

###############################################
# 데이터 불러오기 및 분할 (원래 코드 그대로 사용)
###############################################

df = pd.read_csv('g_1min_m_add_2024112500_2025022313.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 원하는 날짜 구간으로 분할
train_df = df[(df['timestamp'] >= '2024-11-26') & (df['timestamp'] <= '2025-01-15')].copy()
val_df   = df[(df['timestamp'] >= '2025-01-16') & (df['timestamp'] <= '2025-01-31')].copy()
test_df  = df[(df['timestamp'] >= '2025-02-01') & (df['timestamp'] <= '2025-02-22')].copy()

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

###############################################
# 데이터 전처리: 로그수익률, 조정 가격 및 거래량 표준화
###############################################

def preprocess_df(df):
    # 가격 컬럼별 로그수익률 계산 (첫 행은 NaN)
    for col in ["open", "high", "low", "close"]:
        df[f"log_return_{col}"] = np.log(df[col] / df[col].shift(1))
    
    # 누적 로그수익률을 통해 조정 가격 계산
    df["adj_open"]  = df["open"].iloc[0]  * np.exp(df["log_return_open"].cumsum())
    df["adj_high"]  = df["high"].iloc[0]  * np.exp(df["log_return_high"].cumsum())
    df["adj_low"]   = df["low"].iloc[0]   * np.exp(df["log_return_low"].cumsum())
    df["adj_close"] = df["close"].iloc[0] * np.exp(df["log_return_close"].cumsum())
    
    # 거래량 로그 변환
    df["log_volume"] = np.log1p(df["volume"])
    
    # 파생 변수: 이동 평균 및 볼륨 비율 (window=14 예시)
    df["rolling_mean_volume"] = df["log_volume"].rolling(window=14).mean()
    df["volume_ratio"] = df["log_volume"] / df["rolling_mean_volume"]
    
    return df

# 각 분할 데이터프레임에 동일하게 전처리 적용
train_df = preprocess_df(train_df)
val_df = preprocess_df(val_df)
test_df = preprocess_df(test_df)

# 로그수익률 계산으로 발생하는 NaN 제거 및 인덱스 재설정
train_df = train_df.dropna().reset_index(drop=True)
val_df   = val_df.dropna().reset_index(drop=True)
test_df  = test_df.dropna().reset_index(drop=True)

# 거래량 표준화 (훈련 데이터 기준으로 fit 후, 검증/테스트 데이터에 transform 적용)
# 기존 StandardScaler 대신 RobustScaler 사용
# volume_scaler = StandardScaler()
volume_scaler = RobustScaler()
train_df["scaled_volume_ratio"] = volume_scaler.fit_transform(train_df[["volume_ratio"]])
val_df["scaled_volume_ratio"] = volume_scaler.transform(val_df[["volume_ratio"]])
test_df["scaled_volume_ratio"] = volume_scaler.transform(test_df[["volume_ratio"]])

# 조정된 가격 데이터를 원래의 컬럼 이름으로 재설정하고, 거래량은 표준화된 값으로 대체
def finalize_df(df):
    df_final = df[["adj_open", "adj_high", "adj_low", "adj_close", "scaled_volume_ratio"]].copy()
    df_final.columns = ["open", "high", "low", "close", "volume"]
    return df_final

train_df = finalize_df(train_df)
val_df   = finalize_df(val_df)
test_df  = finalize_df(test_df)

# 강화학습 환경에서 사용될 피처 컬럼
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

###############################################
# 강화학습 환경(TradingEnv)
###############################################

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, lookback=1440, initial_balance=10000.0, 
                 buy_fee_rate=0.00002, sell_fee_rate=0.00003, slippage=0.0005,
                 max_holding_duration=180):  # 3시간 = 180분
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.lookback = lookback  # 하루치 데이터를 관측값으로 사용
        self.max_step = len(df) - 1
        self.current_step = self.lookback
        
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.holding = 0.0  # 비율 기반
        self.net_worth = self.balance
        
        self.buy_fee_rate = buy_fee_rate
        self.sell_fee_rate = sell_fee_rate
        self.slippage = slippage  # 매수/매도 가격 변동 반영
        
        # 보유 기간 제한 변수
        self.max_holding_duration = max_holding_duration
        self.holding_duration = 0
        
        # 에피소드 누적 수익 추적 (실제 금액 단위)
        self.episode_profit = 0.0
        
        # (lookback, len(FEATURE_COLUMNS))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(lookback, len(FEATURE_COLUMNS)), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def _get_current_price(self):
        return self.df.loc[self.current_step, 'close']
    
    def _get_obs(self):
        obs = self.df.loc[self.current_step - self.lookback:self.current_step-1, FEATURE_COLUMNS].values
        return obs.astype(np.float32)
    
    def _update_net_worth(self, current_price):
        self.net_worth = self.balance + (self.holding * current_price)

    def step(self, action):
        prev_net_worth = self.net_worth
        current_price = self._get_current_price()
        
        trade_amount = action[0]  # 0~1 비율
        
        # 거래 타입 초기화 (매수, 매도, 홀드)
        trade_type = "hold"
        if trade_amount > 0.5:  # 매수
            trade_type = "buy"
            amount = (trade_amount - 0.5) * 2  # 0.5~1.0 -> 0~1 변환
            cost = amount * current_price * (1 + self.buy_fee_rate + self.slippage)
            if self.balance >= cost:
                self.balance -= cost
                self.holding += amount
        elif trade_amount < 0.5:  # 매도
            trade_type = "sell"
            amount = (0.5 - trade_amount) * 2  # 0~0.5 -> 0~1 변환
            proceeds = amount * current_price * (1 - self.sell_fee_rate - self.slippage)
            if self.holding >= amount:
                self.balance += proceeds
                self.holding -= amount

        # 보유 상태에 따른 보유 시간 업데이트
        if self.holding > 0:
            self.holding_duration += 1
        else:
            self.holding_duration = 0

        # 보유 기간 초과 시 강제 매도 처리 후 에피소드 종료
        if self.holding_duration >= self.max_holding_duration:
            # 보유한 모든 포지션을 현재 가격에 강제 매도
            if self.holding > 0:
                amount = self.holding
                proceeds = amount * current_price * (1 - self.sell_fee_rate - self.slippage)
                self.balance += proceeds
                self.holding = 0.0
            done = True
        else:
            self.current_step += 1
            done = self.current_step >= self.max_step
        
        self._update_net_worth(current_price)
        
        # 실제 금액 차이를 reward로 사용 (예: -100, +1000 등)
        reward = (self.net_worth - prev_net_worth)
        
        # 에피소드 누적 수익 업데이트
        self.episode_profit += reward
        
        obs = self._get_obs()
        
        info = {
            'step': self.current_step, 
            'balance': self.balance, 
            'holding': self.holding, 
            'net_worth': self.net_worth, 
            'holding_duration': self.holding_duration,
            'trade_type': trade_type,       # "buy", "sell", 또는 "hold"
            'reward': reward                # 해당 step의 보상값 (실제 금액 차이)
        }
        # 에피소드 종료 시 누적 수익을 info에 추가
        if done:
            info['episode_profit'] = self.episode_profit
        return obs, reward, done, info

    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.holding = 0.0
        self.net_worth = self.balance
        self.holding_duration = 0
        self.episode_profit = 0.0
        return self._get_obs()

###############################################
# 학습, 검증, 테스트 환경 준비
###############################################

train_env = DummyVecEnv([lambda: TradingEnv(train_df)])
val_env   = DummyVecEnv([lambda: TradingEnv(val_df)])
test_env  = DummyVecEnv([lambda: TradingEnv(test_df)])

###############################################
# 텐서보드에 커스텀 metric 기록하는 Callback
###############################################

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        
        if infos:
            info = infos[0]
            # 현재 step의 실제 보상 (금액)
            step_reward = rewards[0] if rewards else 0.0
            self.logger.record("custom/step_reward", step_reward)
            # 현재 net_worth 기록
            net_worth = info.get("net_worth", 0)
            self.logger.record("custom/net_worth", net_worth)
            # 거래 타입 indicator 기록
            trade_type = info.get("trade_type", "hold")
            buy_flag = 1 if trade_type == "buy" else 0
            sell_flag = 1 if trade_type == "sell" else 0
            hold_flag = 1 if trade_type == "hold" else 0
            self.logger.record("custom/buy_flag", buy_flag)
            self.logger.record("custom/sell_flag", sell_flag)
            self.logger.record("custom/hold_flag", hold_flag)
            # 에피소드 종료 시 누적 수익 기록
            if dones and dones[0] and "episode_profit" in info:
                ep_profit = info["episode_profit"]
                self.logger.record("custom/episode_profit", ep_profit)
            
            # 1000 스텝마다 모델 파라미터 히스토그램 기록
            if self.n_calls % 1000 == 0:
                for name, param in self.model.policy.named_parameters():
                    for output_format in self.logger.output_formats:
                        if hasattr(output_format, "writer"):
                            output_format.writer.add_histogram(
                                f"parameters/{name}",
                                param.detach().cpu().numpy(),
                                self.num_timesteps
                            )
        return True

###############################################
# PPO 모델 설정 및 학습
###############################################

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="../tensorboard_log/",
    n_steps=3200,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.02,
    learning_rate=5e-5,
    clip_range=0.15
)

# 학습 시 콜백을 통해 커스텀 metric 기록
model.learn(total_timesteps=20000000, callback=TensorboardCallback())
print("----- 학습 완료 -----")

# 학습된 모델 저장
model.save("ppo_trading_model_ppo5")
print("모델 저장 완료!")

###############################################
# 검증 (Validation) & 테스트 (Test) 시뮬레이션
###############################################

for env, name in zip([val_env, test_env], ["Validation", "Test"]):
    print(f"\n===== {name} 시뮬레이션 =====")
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    print(f"{name} 종료. 마지막 info:", info)