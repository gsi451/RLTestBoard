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
 |- PPO6 - PPO5에서 시가, 고가, 저가 부분을 제외한 종가 부분만 사용하도록 수정(예외로 체크) - 결과 안좋음
 |       |- PPO3에서 볼린저 밴드 부분을 사용하도록 수정(예외로 체크)
 |- PPO7 - PPO6에서 Ma를 추가해서 사용
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

# PPO5 (학습완료) - 결과가 좋아 보임
# PPO3 의 학습 코드에서 거래량 부분을 로그변환후 파생변수를 합한다.
# 파생 변수 계산: volume_ratio를 계산하기 위해 이동 평균을 구한 후, 로그변환 거래량과의 비율을 계산합니다.
# 스케일링: volume_ratio에 대해 RobustScaler를 사용하여 scaled_volume_ratio를 생성합니다.
# 최종 데이터프레임 생성: finalize_df 함수에서 조정 가격과 scaled_volume_ratio를 선택하여 강화학습 환경에 입력될 데이터셋을 구성합니다.

# PPO6 (학습완료) - 결과가 좋아 보임
# PPO5에서 시가, 고가, 저가 부분을 제외한다.
# 종가, 거래량 만 사용해서 학습해보자. 5번과 어떻게 다른지 확인해보자. - 결과 안좋음
# PPO3에서 볼린저 밴드 상하 부분 사용해서 학습해보자. 5번과 어덯게 다른지 확인해보자.

# PPO7 (학습완료) - (지금까지의 결과중에서 제일 좋음)
# PPO6에서 MA5,MA20,MA60를 추가해서 결과를 보자
# ** 이 모델을 실전 테스트를 하면서 나온 하나의 문제점이 1매수 로직을 추가하지 않아서 이 부분을 추가해서 PPO8을 돌려보고 다시 판단하자.

# PPO8 (학습완료) - (결과는 좋지 않지만 다음 변경할 부분들에 대한 기준이 생김)
# 지금까지의 모델에서 1매수 후 추가 매수는 없도록 하는 로직을 추가한다.

"""
위의 모든 학습 내용을 RLTestboard에 기록하고 
결과를 공유하도록 한다.
이후 다음 과제를 새로운 번호를 부여해서 진행 하도록 하자.

1. PPO 자체 하이퍼파라미터 튜닝
learning_rate 조정
현재 5e-5로 설정되어 있는데, 지나치게 작으면 학습 속도가 너무 느려지거나 수렴이 어려울 수 있습니다. 반대로 너무 크면 불안정해질 수 있으니, 예컨대 1e-4 ~ 3e-4 정도 범위를 탐색해보는 것도 좋습니다.
n_steps(rollout 길이) / batch_size 재설정
현재 n_steps=3200인데, 너무 크면 학습 과정에서 Gradient가 안정적으로 전파되기 어려울 수도 있고, 반대로 너무 작으면 샘플 효율이 떨어질 수 있습니다. 예를 들어 1024, 2048, 4096 등 다양한 값을 시도해볼 수 있습니다.
batch_size도 (기본값이 64인 경우가 많음) 함께 변경해가며 최적 조합을 찾는 것이 좋습니다.
ent_coef 조정
ent_coef=0.02로 설정되어 있는데, 이는 탐험(Exploration)을 어느 정도 유도하기 위한 항목입니다. 너무 크면 액션이 랜덤에 가깝게 되고, 너무 작으면 수렴이 너무 빠르면서 국소해(local optima)에 갇힐 수 있습니다. 0.005 ~ 0.01 정도로도 실험해볼 수 있습니다.
clip_range 조정
현재 clip_range=0.15인데, PPO에서 이 값을 너무 높이면 학습이 불안정해지고, 너무 낮으면 업데이트가 제한되어 수렴이 더딜 수 있습니다. 일반적으로 0.1 ~ 0.2 범위를 시도해보고 모델의 성능 변화를 살펴보시면 됩니다.

2. 네트워크 구조(Policy Network) 변경
MLP 레이어 구조 변경
기본 MlpPolicy는 보통 6464 혹은 256256 정도로 이뤄져 있습니다. 좀 더 깊은 네트워크(예: 12812864)나 넓은 레이어(예: 256~256) 등을 시도해볼 수 있습니다.
policy_kwargs 인자를 사용하여 net_arch를 직접 지정해볼 수도 있습니다.
python
복사
편집
model = PPO(
    "MlpPolicy",
    ...,
    policy_kwargs={"net_arch": [256, 256]}
)
RNN/LSTM/GRU Policy 적용
시계열 정보를 좀 더 잘 학습하기 위해 RecurrentPPO(SB3 >= 2.0 버전)나 LSTMPolicy를 사용해보는 방법도 있습니다. 금융 시계열에 대해서는 과거 패턴을 좀 더 효과적으로 학습하는 경우가 있으니 고려해보셔도 좋습니다.

3. 보상 함수(Reward) 설계 및 변형
현재 보상은 “순자산의 증감분(net_worth 차이)”를 그대로 쓰고 있습니다.
보상에 위험관리(리스크) 개념을 포함하면, 변동성 대비 수익을 높게 보는 방향(예: 샤프비율 비슷한 형태)을 시도해볼 수 있습니다.
짧은 구간에서 굉장히 큰 손실이 나는 트레이드를 막기 위해 Drawdown(혹은 MDD)에 대한 페널티를 주는 방법도 가능합니다.
보유 시간이 길수록 소정의 페널티를 준다든지(혹은 스케줄링) 해서 “너무 오래 보유하지 않도록” 유도할 수도 있습니다.

4. 액션 스페이스 혹은 매매 로직 수정
현재 액션 스페이스가 [0,1] 범위의 연속 값 하나를 두고, 이를 0.5 기준으로 buy/sell을 판단하고 있습니다.
(1매수 로직) 여러 번의 부분 매수를 허용하지 않는 구조나, 반대로 여러 번의 부분 매수를 허용하는 구조 모두 각각 장단점이 있습니다.
(기존 화이트 라인) 여러 차례 분할매수가 가능한 경우 변동성 장악력이 높아질 수 있지만, 보유 비중이 늘어나면서 리스크도 같이 증가합니다.
(오렌지 라인) 1회만 진입 후 매수 불가로 제한하면, 매매 빈도가 줄어들어 “견실하지만 적은” 수익으로 이어질 가능성이 있습니다.
Discrete action(매수/매도/홀드 3가지 분리)으로 바꾸거나, Buy/Sell을 -1~1 범위의 범위를 더 명확히 해석하는 연속 액션으로 잡는 등 구조적인 변경을 해볼 수도 있습니다.
Buy/Sell에 대해 클리핑(clipping)을 적용한다거나, trade_amount 자체에 대한 예측이 아닌 ‘매수/매도 확률’을 따로 두는 방법도 있습니다.

5. 데이터 전처리 / 특성(Feature) 보강
현재는 RobustScaler를 쓰고 MA와 Bollinger Band를 추가했습니다.
추가로 시가/고가/저가를 별도로 쓸지, 거래량 파생변수를 더 넣을지 등 여러 케이스에서 미세조정이 가능해 보입니다.
다양한 기술 지표(MACD, RSI, Stochastic 등)를 추가하여 피처로 삼아볼 수도 있고,
반대로 불필요하다고 보이는 지표는 제거해 Feature 수를 줄여서(모델이 더 쉽게 학습) 시도해볼 수도 있습니다.

6. 최대 보유 기간(max_holding_duration) 설정
현재 180분(3시간)으로 설정하셨는데, 만약 짧은 스캘핑이나 데이트레이딩 스타일이라면 이 시간을 줄이거나 늘려보는 것도 결과에 영향을 줄 수 있습니다.
“어차피 180분 뒤엔 강제 청산”이라는 규칙 때문에, 모델이 지나치게 ‘짧은 기간 스윙’에만 집중할 가능성도 있습니다.
-> 이 부분은 삭제처리를 하고 중간중간 “과도한 손실”에 대한 페널티와 혹은 최대 Drawdown을 줄이는 방향으로의 보상 등을 고려할 수 있습니다.

7. 튜닝 프로세스 자동화
하이퍼파라미터의 경우 한두 개씩 손으로 변경해보는 방법도 있지만, Optuna나 ray[tune] 같은 자동화 툴을 적용하면 훨씬 수월하게 최적값을 찾을 수 있습니다.
학습시간이 오래 걸리긴 하지만, 궁극적으로는 자동화 튜닝이 보다 체계적입니다.
결론
결국 강화학습의 성능을 끌어올리려면 “환경(보상/액션/제약)”과 “모델(네트워크 구조/하이퍼파라미터)” 두 축에서 동시에 탐색해야 합니다. 위에서 제안한 여러 방법들 중에서 하나씩(또는 여러 개씩) 시도해보시면서, Validation 데이터에서 성능 추이를 보며 어떤 설정이 가장 좋은 결과를 내는지 비교하시면 좋겠습니다.

우선순위로는,
learning rate, ent_coef, clip_range, n_steps 등 PPO 하이퍼파라미터를 여러 조합으로 간단하게 탐색
네트워크 구조(레이어 크기, RNN 적용 등) 변경
보상함수 변형 또는 데이터/피처 확장
이 순으로 천천히 시도하시는 것이 일반적으로 효율적인 접근입니다.
잘 튜닝해보시고, 좋은 결과 얻으시길 바랍니다!
"""
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
    for col in ["open", "high", "low", "close", "MA5", "MA20", "MA60", "Bollinger_Upper", "Bollinger_Lower"]:
        df[f"log_return_{col}"] = np.log(df[col] / df[col].shift(1))
    
    # 누적 로그수익률을 통해 조정 가격 계산
    df["adj_open"]  = df["open"].iloc[0]  * np.exp(df["log_return_open"].cumsum())
    df["adj_high"]  = df["high"].iloc[0]  * np.exp(df["log_return_high"].cumsum())
    df["adj_low"]   = df["low"].iloc[0]   * np.exp(df["log_return_low"].cumsum())
    df["adj_close"] = df["close"].iloc[0] * np.exp(df["log_return_close"].cumsum())

    df["adj_MA5"] = df["MA5"].iloc[0] * np.exp(df["log_return_MA5"].cumsum())
    df["adj_MA20"] = df["MA20"].iloc[0] * np.exp(df["log_return_MA20"].cumsum())
    df["adj_MA60"] = df["MA60"].iloc[0] * np.exp(df["log_return_MA60"].cumsum())

    df["adj_Bollinger_Upper"] = df["Bollinger_Upper"].iloc[0] * np.exp(df["log_return_Bollinger_Upper"].cumsum())
    df["adj_Bollinger_Lower"] = df["Bollinger_Lower"].iloc[0] * np.exp(df["log_return_Bollinger_Lower"].cumsum())
    
    # 거래량 로그 변환
    df["log_volume"] = np.log1p(df["volume"])

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
train_df["log_scaled_volume"] = volume_scaler.fit_transform(train_df[["log_volume"]])
val_df["log_scaled_volume"]   = volume_scaler.transform(val_df[["log_volume"]])
test_df["log_scaled_volume"]  = volume_scaler.transform(test_df[["log_volume"]])

# 조정된 가격 데이터를 원래의 컬럼 이름으로 재설정하고, 거래량은 표준화된 값으로 대체
def finalize_df(df):
    df_final = df[[
        "adj_open", "adj_high", "adj_low", "adj_close", 
        "adj_MA5", "adj_MA20", "adj_MA60",
        "adj_Bollinger_Upper", "adj_Bollinger_Lower", 
        "log_scaled_volume"]].copy()
    df_final.columns = [
        "open", "high", "low", "close", "volume", 
        "MA5", "MA20", "MA60",
        "Bollinger_Upper", "Bollinger_Lower"]
    return df_final

train_df = finalize_df(train_df)
val_df   = finalize_df(val_df)
test_df  = finalize_df(test_df)

# 강화학습 환경에서 사용될 피처 컬럼
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 
    'MA5', 'MA20', 'MA60',
    'Bollinger_Upper', 'Bollinger_Lower', 'volume']

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
        trade_type = "hold"
        
        # 매수: action이 0.5 초과일 때 -> 단, 이미 보유 중이면 매수 불가
        if trade_amount > 0.5:
            # 보유 중이 아니라면 매수 진행, 보유 중이면 매수 신호를 무시하고 hold 처리
            if self.holding == 0:
                trade_type = "buy"
                amount = (trade_amount - 0.5) * 2  # 0.5~1.0 -> 0~1 변환
                cost = amount * current_price * (1 + self.buy_fee_rate + self.slippage)
                if self.balance >= cost:
                    self.balance -= cost
                    self.holding += amount
            else:
                trade_type = "hold"  # 이미 매수 후 보유 중이면 추가 매수 불가
        
        # 매도: action이 0.5 미만일 때 진행
        elif trade_amount < 0.5:
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
model.save("ppo_trading_model_ppo7")
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