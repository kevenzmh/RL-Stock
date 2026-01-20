"""
快速训练测试 - 使用较小的训练步数
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from rlenv.StockTradingEnv0 import StockTradingEnv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
try:
    font = fm.FontProperties(fname='font/wqy-microhei.ttc')
except:
    font = None

plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("快速训练测试")
print("=" * 60)

# 1. 选择一个股票数据
stock_code = 'sh.600036'  # 招商银行
train_file = f'stockdata/train/{stock_code}.招商银行.csv'
test_file = f'stockdata/test/{stock_code}.招商银行.csv'

if not os.path.exists(train_file):
    print(f"❌ 训练文件不存在: {train_file}")
    # 尝试查找任意一个文件
    train_files = list(Path('stockdata/train').glob('sh.6*.csv'))
    if len(train_files) > 0:
        train_file = str(train_files[0])
        test_file = str(train_file).replace('train', 'test')
        print(f"✅ 使用替代文件: {Path(train_file).name}")

# 2. 加载和预处理数据
print(f"\n加载训练数据: {Path(train_file).name}")
df_train = pd.read_csv(train_file)
df_train = df_train.sort_values('date')

print(f"训练数据: {len(df_train)} 条记录")
print(f"时间范围: {df_train['date'].min()} 至 {df_train['date'].max()}")

# 检查缺失值
missing_count = df_train.isnull().sum().sum()
if missing_count > 0:
    print(f"⚠️  检测到 {missing_count} 个缺失值,进行填充...")
    df_train = df_train.fillna(method='ffill').fillna(0)

# 3. 创建训练环境
print("\n创建训练环境...")
env = DummyVecEnv([lambda: StockTradingEnv(df_train)])

# 4. 创建PPO模型 (使用较小的训练步数用于快速测试)
print("\n创建PPO2模型...")
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)

# 5. 开始训练 (只训练2000步用于快速测试)
train_steps = 2000
print(f"\n开始训练 (训练步数: {train_steps})...")
print("这可能需要几分钟时间...")

try:
    model.learn(total_timesteps=train_steps)
    print("✅ 训练完成!")
    
    # 6. 保存模型
    model_path = './models/quick_test_model'
    os.makedirs('./models', exist_ok=True)
    model.save(model_path)
    print(f"✅ 模型已保存: {model_path}")
    
except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()

# 7. 在测试集上评估
print("\n" + "=" * 60)
print("在测试集上评估模型")
print("=" * 60)

if os.path.exists(test_file):
    df_test = pd.read_csv(test_file)
    df_test = df_test.sort_values('date')
    
    # 填充缺失值
    df_test = df_test.fillna(method='ffill').fillna(0)
    
    print(f"\n测试数据: {len(df_test)} 条记录")
    print(f"时间范围: {df_test['date'].min()} 至 {df_test['date'].max()}")
    
    # 创建测试环境
    env_test = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    
    # 运行测试
    obs = env_test.reset()
    day_profits = []
    
    print("\n开始测试...")
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        
        # 获取当前收益
        profit = env_test.envs[0].net_worth - 10000
        day_profits.append(profit)
        
        if done[0]:
            print(f"测试在第 {i+1} 天结束")
            break
    
    # 8. 显示结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    final_profit = day_profits[-1] if len(day_profits) > 0 else 0
    max_profit = max(day_profits) if len(day_profits) > 0 else 0
    min_profit = min(day_profits) if len(day_profits) > 0 else 0
    
    print(f"\n初始资金: 10000")
    print(f"最终收益: {final_profit:.2f} ({final_profit/10000*100:.2f}%)")
    print(f"最大收益: {max_profit:.2f}")
    print(f"最小收益: {min_profit:.2f}")
    print(f"测试天数: {len(day_profits)}")
    
    # 9. 绘制收益曲线
    print("\n绘制收益曲线...")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(day_profits, '-o', marker='o', ms=4, alpha=0.7, mfc='orange')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Trading Days', fontsize=12)
        plt.ylabel('Profit (CNY)', fontsize=12)
        plt.title(f'Quick Test - Profit Curve', fontsize=14)
        
        # 保存图片
        output_path = './img/quick_test_profit.png'
        os.makedirs('./img', exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✅ 收益曲线已保存: {output_path}")
        
    except Exception as e:
        print(f"⚠️  绘图失败: {e}")
    
    env_test.close()
else:
    print(f"❌ 测试文件不存在: {test_file}")

env.close()

print("\n" + "=" * 60)
print("快速测试完成!")
print("=" * 60)
