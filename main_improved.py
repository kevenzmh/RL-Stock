import os
import pickle
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from rlenv.StockTradingEnv_improved import StockTradingEnvImproved

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def stock_trade_improved(stock_file, train_steps=100000):
    """
    改进的股票交易训练函数
    
    Args:
        stock_file: 股票数据文件路径
        train_steps: 训练步数,默认10万步(原来是1万)
    """
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')
    
    print(f"训练数据: {len(df)} 条记录")
    print(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")

    # 创建训练环境
    env = DummyVecEnv([lambda: StockTradingEnvImproved(df)])

    # 确保log目录存在
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_improved')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n开始训练,总步数: {train_steps}")
    
    # 改进的PPO2参数
    model = PPO2(
        MlpPolicy, 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,      # 学习率
        n_steps=2048,               # 每次更新的步数
        nminibatches=32,            # minibatch数量
        noptepochs=10,              # 每次更新的训练轮数
        gamma=0.99,                 # 折扣因子
        lam=0.95,                   # GAE参数
        cliprange=0.2,              # PPO裁剪参数
        ent_coef=0.01,              # 熵系数,鼓励探索
    )
    
    model.learn(total_timesteps=train_steps)
    
    # 保存模型
    model_path = f'./models/ppo2_stock_{train_steps}.pkl'
    os.makedirs('./models', exist_ok=True)
    model.save(model_path)
    print(f"\n模型已保存至: {model_path}")

    # 测试
    print("\n开始测试...")
    df_test = pd.read_csv(stock_file.replace('train', 'test'))
    print(f"测试数据: {len(df_test)} 条记录")
    print(f"时间范围: {df_test['date'].min()} 至 {df_test['date'].max()}")

    env_test = DummyVecEnv([lambda: StockTradingEnvImproved(df_test)])
    obs = env_test.reset()
    
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        
        # 每10步打印一次
        if i % 10 == 0:
            profit = env_test.envs[0].net_worth - 10000
            day_profits.append(profit)
            print(f"Step {i}: Profit = {profit:.2f}")
        
        if done:
            break
    
    final_profit = env_test.envs[0].net_worth - 10000
    print(f"\n最终利润: {final_profit:.2f} ({final_profit/10000*100:.2f}%)")
    
    return day_profits


def find_file(path, name):
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_improved_strategy(stock_code='sh.000001', train_steps=100000):
    """
    测试改进的交易策略
    
    Args:
        stock_code: 股票代码
        train_steps: 训练步数,默认10万
    """
    print(f"{'='*50}")
    print(f"股票代码: {stock_code}")
    print(f"训练步数: {train_steps}")
    print(f"{'='*50}\n")
    
    stock_file = find_file('./stockdata/train', str(stock_code))
    
    if not stock_file:
        print(f"错误: 找不到股票 {stock_code} 的数据文件")
        return
    
    daily_profits = stock_trade_improved(stock_file, train_steps)
    
    # 绘制利润曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_profits, '-o', label=f'{stock_code} (训练{train_steps}步)', 
            marker='o', ms=6, alpha=0.7, mfc='orange')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='盈亏平衡线')
    plt.xlabel('测试步数', fontproperties=font, fontsize=12)
    plt.ylabel('利润 (元)', fontproperties=font, fontsize=12)
    plt.title(f'{stock_code} 强化学习交易策略收益曲线', fontproperties=font, fontsize=14)
    ax.legend(prop=font)
    
    # 保存图片
    plt.savefig(f'./img/{stock_code}_improved.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: ./img/{stock_code}_improved.png")
    plt.close()


if __name__ == '__main__':
    # 测试改进的策略,使用10万步训练
    test_improved_strategy('sh.000001', train_steps=100000)
    
    # 如果想要更好的效果,可以尝试更多训练步数:
    # test_improved_strategy('sh.000001', train_steps=200000)
