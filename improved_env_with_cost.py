"""
改进版训练脚本 - 包含交易成本和风险控制
基于原项目优化
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
INITIAL_ACCOUNT_BALANCE = 10000

# 交易成本配置
COMMISSION_RATE = 0.0003  # 手续费 0.03%
STAMP_DUTY_RATE = 0.001   # 印花税 0.1% (仅卖出)
MIN_COMMISSION = 5        # 最低手续费 5元

# 风险控制配置
MAX_POSITION_RATIO = 0.8  # 最大仓位比例
MAX_SINGLE_LOSS_RATIO = 0.02  # 单笔最大亏损比例
STOP_LOSS_RATIO = 0.10  # 总止损比例


class ImprovedStockTradingEnv(gym.Env):
    """改进的股票交易环境,加入交易成本和风险控制"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(ImprovedStockTradingEnv, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        
        # 动作空间: [动作类型, 交易比例]
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        
        # 观测空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(21,), dtype=np.float16)  # 增加2个风险特征

    def _calculate_commission(self, transaction_amount, is_sell=False):
        """计算交易成本"""
        commission = transaction_amount * COMMISSION_RATE
        commission = max(commission, MIN_COMMISSION)  # 最低佣金
        
        if is_sell:
            stamp_duty = transaction_amount * STAMP_DUTY_RATE
            total_cost = commission + stamp_duty
        else:
            total_cost = commission
        
        return total_cost
    
    def _next_observation(self):
        """获取当前观测"""
        # 原有特征
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
            # 新增风险特征
            self.position_ratio,  # 当前仓位比例
            self.drawdown_ratio,  # 当前回撤比例
        ])
        return obs
