"""
增强版股票交易环境
包含以下改进:
1. 风险调整后的奖励函数(夏普比率)
2. 集成技术指标特征(MA, MACD, RSI, KDJ, Bollinger Bands)
3. 更好的状态空间设计
4. 交易成本考虑
"""
import random
import gym
from gym import spaces
import numpy as np
import pandas as pd

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000
TRANSACTION_FEE_PERCENT = 0.001  # 0.1% 交易费


class StockTradingEnvEnhanced(gym.Env):
    """增强版股票交易环境"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, risk_free_rate=0.03):
        super(StockTradingEnvEnhanced, self).__init__()
        self.df = df
        self.risk_free_rate = risk_free_rate / 252  # 日化无风险利率
        self.reward_range = (-np.inf, np.inf)

        # 动作空间: [动作类型(0-3), 交易比例(0-1)]
        # 0: 持有, 1: 买入, 2: 卖出, 3: 持有
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 计算观察空间维度
        # 基础特征(6) + 技术指标(~20) + 账户状态(6) = 32维
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(32,), dtype=np.float16)
        
        # 用于计算夏普比率的历史收益
        self.returns_history = []
        self.net_worth_history = []

    def _next_observation(self):
        """构建观察状态 - 包含价格、技术指标和账户状态"""
        frame = self.df.iloc[self.current_step]
        
        # 基础价格特征
        obs = [
            frame['open'] / MAX_SHARE_PRICE,
            frame['high'] / MAX_SHARE_PRICE,
            frame['low'] / MAX_SHARE_PRICE,
            frame['close'] / MAX_SHARE_PRICE,
            frame['volume'] / MAX_VOLUME,
            frame['pctChg'] / 100,
        ]
        
        # 技术指标特征
        # 移动平均线
        if 'ma5' in frame:
            obs.extend([
                frame['ma5'] / MAX_SHARE_PRICE if pd.notna(frame['ma5']) else 0,
                frame['ma10'] / MAX_SHARE_PRICE if pd.notna(frame['ma10']) else 0,
                frame['ma20'] / MAX_SHARE_PRICE if pd.notna(frame['ma20']) else 0,
                frame['ma60'] / MAX_SHARE_PRICE if pd.notna(frame['ma60']) else 0,
            ])
        else:
            obs.extend([0, 0, 0, 0])
        
        # MACD指标
        if 'macd' in frame:
            obs.extend([
                frame['macd'] / 100 if pd.notna(frame['macd']) else 0,
                frame['macd_signal'] / 100 if pd.notna(frame['macd_signal']) else 0,
                frame['macd_hist'] / 100 if pd.notna(frame['macd_hist']) else 0,
            ])
        else:
            obs.extend([0, 0, 0])
        
        # RSI指标
        if 'rsi' in frame:
            obs.append(frame['rsi'] / 100 if pd.notna(frame['rsi']) else 0.5)
        else:
            obs.append(0.5)
        
        # KDJ指标
        if 'kdj_k' in frame:
            obs.extend([
                frame['kdj_k'] / 100 if pd.notna(frame['kdj_k']) else 0.5,
                frame['kdj_d'] / 100 if pd.notna(frame['kdj_d']) else 0.5,
                frame['kdj_j'] / 100 if pd.notna(frame['kdj_j']) else 0.5,
            ])
        else:
            obs.extend([0.5, 0.5, 0.5])
        
        # 布林带
        if 'bb_upper' in frame:
            obs.extend([
                frame['bb_upper'] / MAX_SHARE_PRICE if pd.notna(frame['bb_upper']) else 0,
                frame['bb_middle'] / MAX_SHARE_PRICE if pd.notna(frame['bb_middle']) else 0,
                frame['bb_lower'] / MAX_SHARE_PRICE if pd.notna(frame['bb_lower']) else 0,
                frame['bb_width'] if pd.notna(frame['bb_width']) else 0,
            ])
        else:
            obs.extend([0, 0, 0, 0])
        
        # 成交量指标
        if 'volume_ratio' in frame:
            obs.extend([
                min(frame['volume_ratio'] / 5, 1) if pd.notna(frame['volume_ratio']) else 0.5,
            ])
        else:
            obs.append(0.5)
        
        # ATR波动率
        if 'atr' in frame:
            obs.append(frame['atr'] / MAX_SHARE_PRICE if pd.notna(frame['atr']) else 0)
        else:
            obs.append(0)
        
        # 账户状态特征
        obs.extend([
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE if self.shares_held > 0 else 0,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        
        # 确保维度正确
        obs = np.array(obs, dtype=np.float16)
        if len(obs) < 32:
            obs = np.pad(obs, (0, 32 - len(obs)), mode='constant')
        elif len(obs) > 32:
            obs = obs[:32]
        
        return obs

    def _take_action(self, action):
        """执行交易动作,考虑交易成本"""
        current_price = random.uniform(
            self.df.iloc[self.current_step]['open'], 
            self.df.iloc[self.current_step]['close'])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # 买入
            total_possible = int(self.balance / (current_price * (1 + TRANSACTION_FEE_PERCENT)))
            shares_bought = int(total_possible * amount)
            
            if shares_bought > 0:
                cost = shares_bought * current_price * (1 + TRANSACTION_FEE_PERCENT)
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price
                
                self.balance -= cost
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought

        elif action_type < 2:
            # 卖出
            shares_sold = int(self.shares_held * amount)
            
            if shares_sold > 0:
                revenue = shares_sold * current_price * (1 - TRANSACTION_FEE_PERCENT)
                self.balance += revenue
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += revenue

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def _calculate_reward(self, prev_net_worth):
        """
        计算风险调整后的奖励
        考虑:
        1. 净值变化
        2. 收益的波动率(夏普比率概念)
        3. 回撤惩罚
        """
        # 基础收益
        net_worth_change = self.net_worth - prev_net_worth
        return_rate = net_worth_change / prev_net_worth if prev_net_worth > 0 else 0
        
        # 记录历史
        self.returns_history.append(return_rate)
        self.net_worth_history.append(self.net_worth)
        
        # 保持历史记录在合理长度
        if len(self.returns_history) > 100:
            self.returns_history.pop(0)
            self.net_worth_history.pop(0)
        
        # 基础奖励
        reward = return_rate
        
        # 如果有足够的历史数据,计算夏普比率相关的奖励
        if len(self.returns_history) >= 10:
            mean_return = np.mean(self.returns_history)
            std_return = np.std(self.returns_history)
            
            # 避免除以零
            if std_return > 1e-6:
                # 简化的夏普比率 (日收益 - 无风险利率) / 波动率
                sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
                # 将夏普比率作为额外奖励/惩罚
                reward += sharpe_ratio * 0.01
            
            # 波动率惩罚 - 惩罚过高的波动
            volatility_penalty = std_return * 0.1
            reward -= volatility_penalty
        
        # 回撤惩罚
        if len(self.net_worth_history) >= 2:
            max_net_worth_so_far = max(self.net_worth_history)
            drawdown = (max_net_worth_so_far - self.net_worth) / max_net_worth_so_far
            if drawdown > 0.1:  # 回撤超过10%
                reward -= drawdown * 0.5
        
        # 破产惩罚
        if self.net_worth < INITIAL_ACCOUNT_BALANCE * 0.5:
            reward -= 1.0
        
        return reward

    def step(self, action):
        prev_net_worth = self.net_worth
        
        self._take_action(action)
        
        self.current_step += 1

        done = False
        if self.current_step >= len(self.df) - 1:
            done = True

        # 使用改进的奖励函数
        reward = self._calculate_reward(prev_net_worth)
        
        # 终止条件
        if self.net_worth <= 0:
            reward = -10
            done = True

        obs = self._next_observation()
        return obs, reward, done, {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held
        }

    def reset(self, new_df=None):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        # 重置历史记录
        self.returns_history = []
        self.net_worth_history = [INITIAL_ACCOUNT_BALANCE]

        if new_df is not None:
            self.df = new_df

        # 随机开始位置,但确保有足够的数据计算技术指标
        # 修复：确保不会越界
        min_start = min(60, len(self.df) - 10)  # 至少留10个交易日
        max_start = len(self.df) - 10  # 确保至少有10步可以交易
        
        # 如果数据太少，从头开始
        if max_start < min_start or max_start < 0:
            self.current_step = 0
        else:
            self.current_step = random.randint(min_start, max_start)

        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost: {self.cost_basis:.2f} (Total sales: {self.total_sales_value:.2f})')
        print(f'Net worth: {self.net_worth:.2f} (Max: {self.max_net_worth:.2f})')
        print(f'Profit: {profit:.2f} ({profit/INITIAL_ACCOUNT_BALANCE*100:.2f}%)')
        return profit
