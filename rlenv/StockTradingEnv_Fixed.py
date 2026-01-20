"""
修复并改进的股票交易环境
包含以下改进:
1. 修复除零错误
2. 增加真实交易成本模拟
3. 添加风险控制机制
4. 支持GPU加速
"""
import random
import gym
from gym import spaces
import pandas as pd
import numpy as np

# 常量定义
MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
INITIAL_ACCOUNT_BALANCE = 10000

# 交易成本配置 (符合中国A股市场规则)
COMMISSION_RATE = 0.0003      # 手续费 0.03%
STAMP_DUTY_RATE = 0.001       # 印花税 0.1% (仅卖出)
TRANSFER_FEE_RATE = 0.00002   # 过户费 0.002%
MIN_COMMISSION = 5            # 最低手续费 5元

# 风险控制配置
MAX_POSITION_RATIO = 0.70     # 最大仓位比例 70%
MAX_SINGLE_LOSS_RATIO = 0.02  # 单笔最大亏损 2%
MAX_TOTAL_LOSS_RATIO = 0.20   # 最大总亏损 20%
MAX_CONSECUTIVE_LOSSES = 3    # 最大连续亏损次数


class StockTradingEnvFixed(gym.Env):
    """修复并改进的股票交易环境"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnvFixed, self).__init__()
        
        self.df = df.reset_index(drop=True)  # 重置索引,避免索引问题
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        
        # 动作空间: [动作类型, 交易比例]
        # 动作类型: 0-1=买入, 1-2=卖出, 2-3=持有
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([3, 1]), 
            dtype=np.float32)
        
        # 观测空间 (增加风险控制相关特征)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(22,), dtype=np.float32)

    def _calculate_transaction_cost(self, shares, price, is_sell=False):
        """
        计算交易成本
        Args:
            shares: 交易股数
            price: 交易价格
            is_sell: 是否为卖出操作
        Returns:
            总交易成本
        """
        if shares == 0:
            return 0
        
        transaction_amount = shares * price
        
        # 佣金 (最低5元)
        commission = max(transaction_amount * COMMISSION_RATE, MIN_COMMISSION)
        
        # 过户费 (沪深两市都收取)
        transfer_fee = transaction_amount * TRANSFER_FEE_RATE
        
        # 印花税 (仅卖出时收取)
        stamp_duty = transaction_amount * STAMP_DUTY_RATE if is_sell else 0
        
        total_cost = commission + transfer_fee + stamp_duty
        
        return total_cost

    def _check_risk_limits(self, action_type, amount, current_price):
        """
        检查风险限制
        Returns:
            是否允许交易, 调整后的交易比例
        """
        # 检查是否连续亏损过多
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False, 0  # 停止交易
        
        # 检查总亏损是否超限
        total_loss_ratio = (INITIAL_ACCOUNT_BALANCE - self.net_worth) / INITIAL_ACCOUNT_BALANCE
        if total_loss_ratio > MAX_TOTAL_LOSS_RATIO:
            return False, 0  # 停止交易
        
        # 买入操作的风险控制
        if action_type < 1:
            # 计算当前持仓市值
            current_position_value = self.shares_held * current_price
            total_assets = self.balance + current_position_value
            
            # 计算拟买入后的仓位比例
            max_buy_value = self.balance * amount
            projected_position = (current_position_value + max_buy_value) / total_assets
            
            # 限制最大仓位
            if projected_position > MAX_POSITION_RATIO:
                # 调整买入比例
                allowed_buy_value = MAX_POSITION_RATIO * total_assets - current_position_value
                if allowed_buy_value <= 0:
                    return False, 0
                adjusted_amount = allowed_buy_value / self.balance
                return True, min(adjusted_amount, 1.0)
        
        # 卖出操作的风险控制
        elif action_type < 2:
            # 检查单笔亏损是否过大
            if self.shares_held > 0 and self.cost_basis > 0:
                sell_price = current_price
                loss_ratio = (self.cost_basis - sell_price) / self.cost_basis
                
                if loss_ratio > MAX_SINGLE_LOSS_RATIO:
                    # 如果单笔亏损过大,减少卖出比例
                    adjusted_amount = amount * 0.5  # 减半卖出
                    return True, adjusted_amount
        
        return True, amount

    def _next_observation(self):
        """获取当前观测状态"""
        # 计算风险指标
        position_value = self.shares_held * self.df.loc[self.current_step, 'close']
        total_assets = self.balance + position_value
        position_ratio = position_value / total_assets if total_assets > 0 else 0
        
        # 计算回撤
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0
        
        obs = np.array([
            # 市场数据 (归一化)
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
            
            # 账户状态
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
            
            # 风险控制指标
            position_ratio,                          # 仓位比例
            drawdown,                                # 回撤比例
            self.consecutive_losses / MAX_CONSECUTIVE_LOSSES,  # 连续亏损次数
            self.total_transaction_cost / MAX_ACCOUNT_BALANCE,  # 累计交易成本
        ], dtype=np.float32)
        
        return obs

    def _take_action(self, action):
        """
        执行交易动作
        """
        # 使用收盘价作为交易价格 (更接近实际交易)
        current_price = self.df.loc[self.current_step, "close"]
        
        action_type = action[0]
        amount = action[1]
        
        # 记录交易前净值
        prev_net_worth = self.net_worth
        
        # 风险检查
        can_trade, adjusted_amount = self._check_risk_limits(action_type, amount, current_price)
        
        if not can_trade:
            # 不允许交易,直接返回
            pass
        elif action_type < 1:
            # 买入操作
            amount = adjusted_amount
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            
            if shares_bought > 0:
                # 计算交易成本
                transaction_cost = self._calculate_transaction_cost(
                    shares_bought, current_price, is_sell=False)
                
                # 计算总花费
                total_cost = shares_bought * current_price + transaction_cost
                
                # 确保有足够余额
                if total_cost <= self.balance:
                    prev_cost = self.cost_basis * self.shares_held
                    additional_cost = shares_bought * current_price
                    
                    self.balance -= total_cost
                    
                    # 【修复1: 除零错误】
                    if self.shares_held + shares_bought > 0:
                        self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
                    
                    self.shares_held += shares_bought
                    self.total_transaction_cost += transaction_cost
        
        elif action_type < 2:
            # 卖出操作
            amount = adjusted_amount
            shares_sold = int(self.shares_held * amount)
            
            if shares_sold > 0:
                # 计算交易成本
                transaction_cost = self._calculate_transaction_cost(
                    shares_sold, current_price, is_sell=True)
                
                # 【修复2: 交易成本】卖出收入扣除成本
                sell_revenue = shares_sold * current_price - transaction_cost
                
                self.balance += sell_revenue
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price
                self.total_transaction_cost += transaction_cost
                
                # 如果全部卖出,重置成本基础
                if self.shares_held == 0:
                    self.cost_basis = 0
        
        # 更新净值
        self.net_worth = self.balance + self.shares_held * current_price
        
        # 更新最大净值
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # 【修复4: 连续亏损控制】
        if self.net_worth < prev_net_worth:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def step(self, action):
        """执行一步"""
        # 执行动作
        self._take_action(action)
        
        # 检查是否结束
        done = False
        self.current_step += 1
        
        # 检查是否到达数据末尾
        if self.current_step >= len(self.df) - 1:
            self.current_step = 0  # 循环训练
            # done = True  # 如果不需要循环,取消注释
        
        # 计算奖励 (考虑交易成本)
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = profit / INITIAL_ACCOUNT_BALANCE  # 归一化奖励
        
        # 惩罚交易成本
        cost_penalty = self.total_transaction_cost / INITIAL_ACCOUNT_BALANCE
        reward -= cost_penalty * 0.5  # 交易成本的惩罚系数
        
        # 破产检查
        if self.net_worth <= INITIAL_ACCOUNT_BALANCE * 0.5:
            done = True
            reward -= 1  # 破产惩罚
        
        obs = self._next_observation()
        
        return obs, reward, done, {
            'net_worth': self.net_worth,
            'transaction_cost': self.total_transaction_cost
        }

    def reset(self, new_df=None):
        """重置环境"""
        # 重置账户状态
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        # 重置风险控制指标
        self.consecutive_losses = 0
        self.total_transaction_cost = 0
        
        # 更新数据集
        if new_df is not None:
            self.df = new_df.reset_index(drop=True)
        
        # 重置步数
        self.current_step = 0
        
        return self._next_observation()

    def render(self, mode='human', close=False):
        """渲染环境状态"""
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        profit_rate = (profit / INITIAL_ACCOUNT_BALANCE) * 100
        
        print('-' * 50)
        print(f'Step: {self.current_step}')
        print(f'Balance: ¥{self.balance:.2f}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Cost basis: ¥{self.cost_basis:.2f}')
        print(f'Net worth: ¥{self.net_worth:.2f} (Max: ¥{self.max_net_worth:.2f})')
        print(f'Profit: ¥{profit:.2f} ({profit_rate:.2f}%)')
        print(f'Transaction cost: ¥{self.total_transaction_cost:.2f}')
        print(f'Consecutive losses: {self.consecutive_losses}')
        print('-' * 50)
        
        return profit
