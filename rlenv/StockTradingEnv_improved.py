import random
import gym
from gym import spaces
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnvImproved(gym.Env):
    """改进的股票交易环境"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnvImproved, self).__init__()
        self.df = df
        self.reward_range = (-np.inf, np.inf)

        # 动作空间: [动作类型(0-3), 交易比例(0-1)]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 观察空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float16)

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,
            self.df.loc[self.current_step, 'adjustflag'] / 10,
            self.df.loc[self.current_step, 'tradestatus'] / 1,
            self.df.loc[self.current_step, 'pctChg'] / 100,
            self.df.loc[self.current_step, 'peTTM'] / 1e4,
            self.df.loc[self.current_step, 'pbMRQ'] / 100,
            self.df.loc[self.current_step, 'psTTM'] / 100,
            self.df.loc[self.current_step, 'pctChg'] / 1e3,
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        return obs

    def _take_action(self, action):
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], 
            self.df.loc[self.current_step, "close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # 买入
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            
            if shares_bought > 0:
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price
                self.balance -= additional_cost
                self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought

        elif action_type < 2:
            # 卖出
            shares_sold = int(self.shares_held * amount)
            
            if shares_sold > 0:
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        prev_net_worth = self.net_worth
        
        self._take_action(action)
        
        self.current_step += 1

        done = False
        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0

        # 改进的奖励函数
        # 1. 基础奖励: 净值变化
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change / INITIAL_ACCOUNT_BALANCE
        
        # 2. 鼓励交易但不过度交易
        # 如果一直不交易,给予小的负奖励
        if self.shares_held == 0 and self.total_shares_sold == 0:
            reward -= 0.001  # 小惩罚,鼓励探索
        
        # 3. 如果净值严重下降,给予惩罚
        if self.net_worth < INITIAL_ACCOUNT_BALANCE * 0.5:
            reward -= 1.0
            done = True
        
        # 4. 如果净值为负或为0,结束episode
        if self.net_worth <= 0:
            reward = -10
            done = True

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self, new_df=None):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        if new_df is not None:
            self.df = new_df

        # 随机开始位置,增加训练多样性
        self.current_step = random.randint(0, len(self.df) - 100)

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
