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

############################
# ppo 강화학습
# n_step 의 값이 3200에서 좋은 결과가 나왔다.
# 모델 파일을 저장하는 부분을 진행한다.
# ppo 2
# histograms 추가
############################

############################
# 데이터 불러오기 및 분할
############################

df = pd.read_csv('fartcoin_1min_data_241117_250212.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 원하는 날짜 구간으로 분할
train_df = df[(df['timestamp'] >= '2024-11-18') & (df['timestamp'] <= '2025-01-15')].copy()
val_df   = df[(df['timestamp'] >= '2025-01-16') & (df['timestamp'] <= '2025-01-31')].copy()
test_df  = df[(df['timestamp'] >= '2025-02-01') & (df['timestamp'] <= '2025-02-15')].copy()

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

FEATURE_COLUMNS = ['open','high','low','close','volume','RSI14']

###############################################################
# (옵션) 거래량 로그 변환 및 전체 피처 표준화 (Train->Val/Test)
###############################################################

# 거래량 로그 변환
for df_ in [train_df, val_df, test_df]:
    df_['volume'] = np.log1p(df_['volume'])

# 표준화
scaler = StandardScaler()
scaler.fit(train_df[FEATURE_COLUMNS])
for df_ in [train_df, val_df, test_df]:
    df_[FEATURE_COLUMNS] = scaler.transform(df_[FEATURE_COLUMNS])

################################
# 강화학습 환경(TradingEnv)
################################

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
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback, len(FEATURE_COLUMNS)), dtype=np.float32
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
                # logger.output_formats에서 TensorBoard writer를 찾습니다.
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
    tensorboard_log="./tensorboard_log/",
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
model.save("ppo_trading_model_ppo1")
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
