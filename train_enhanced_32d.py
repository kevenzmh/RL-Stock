"""
快速训练32维增强模型
使用 StockTradingEnvEnhanced (包含技术指标)
"""
import os
import sys
import pandas as pd
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced
from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor

print("="*70)
print("🚀 快速训练32维增强模型")
print("="*70)

# 参数配置
STOCK_FILE = 'stockdata/train/sh.600036.招商银行.csv'  # 招商银行
TOTAL_STEPS = 50000  # 50k步，约30-40分钟
MODEL_NAME = 'models/ppo2_enhanced_32d'

print(f"\n配置:")
print(f"  股票: {STOCK_FILE}")
print(f"  训练步数: {TOTAL_STEPS:,}")
print(f"  模型保存: {MODEL_NAME}")

# 1. 加载数据
print("\n" + "-"*70)
print("【步骤1/5】加载训练数据")
print("-"*70)

df = pd.read_csv(STOCK_FILE)
df = df.sort_values('date').reset_index(drop=True)
print(f"✓ 原始数据: {len(df)} 条")

# 2. 添加技术指标
print("\n" + "-"*70)
print("【步骤2/5】添加技术指标")
print("-"*70)

df = add_all_technical_indicators(df)
print(f"✓ 添加指标后: {len(df)} 条, {len(df.columns)} 列")

# 3. 数据预处理
print("\n" + "-"*70)
print("【步骤3/5】数据预处理")
print("-"*70)

preprocessor = DataPreprocessor(method='robust')
df = preprocessor.preprocess_pipeline(
    df,
    fit=True,
    handle_missing=True,
    handle_outliers_flag=True,
    normalize=False
)
print(f"✓ 预处理完成: {len(df)} 条")

# 确保有足够的数据
if len(df) < 100:
    print(f"✗ 数据太少 ({len(df)} 条)，需要至少100条")
    sys.exit(1)

# 4. 创建环境和模型
print("\n" + "-"*70)
print("【步骤4/5】创建环境和模型")
print("-"*70)

print("创建训练环境...")
env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df)])

print("初始化PPO2模型...")
print("  观察空间: 32维")
print("  策略网络: MlpPolicy [64, 64]")
print("  学习率: 0.00025")

model = PPO2(
    MlpPolicy, 
    env, 
    verbose=1,
    tensorboard_log='./log_enhanced_32d/',
    learning_rate=0.00025,
    n_steps=2048,
    nminibatches=32,
    lam=0.95,
    gamma=0.99,
    noptepochs=10,
    ent_coef=0.01,
    cliprange=0.2
)

print("✓ 模型创建成功")

# 5. 开始训练
print("\n" + "-"*70)
print("【步骤5/5】开始训练")
print("-"*70)

print(f"\n开始训练 {TOTAL_STEPS:,} 步...")
print("预计时间: 30-40分钟（CPU模式）")
print("\n提示: 可以随时按 Ctrl+C 停止训练并保存模型")
print("-"*70)

try:
    model.learn(total_timesteps=TOTAL_STEPS)
    print("\n✓ 训练完成!")
    
except KeyboardInterrupt:
    print("\n\n⚠ 训练被中断，保存当前模型...")

# 6. 保存模型
print("\n" + "-"*70)
print("保存模型")
print("-"*70)

model.save(MODEL_NAME)
print(f"✓ 模型已保存到: {MODEL_NAME}.zip")

# 7. 快速测试
print("\n" + "-"*70)
print("快速测试模型")
print("-"*70)

# 创建测试环境
test_file = 'stockdata/test/sh.600036.csv'
if os.path.exists(test_file):
    print(f"加载测试数据: {test_file}")
    
    df_test = pd.read_csv(test_file)
    df_test = df_test.sort_values('date').reset_index(drop=True)
    df_test = df_test.tail(60)  # 只用最近60天
    
    # 添加指标和预处理
    df_test = add_all_technical_indicators(df_test)
    df_test = preprocessor.preprocess_pipeline(
        df_test, 
        fit=False,
        handle_missing=True,
        handle_outliers_flag=True,
        normalize=False
    )
    
    print(f"测试数据: {len(df_test)} 条")
    
    # 创建测试环境
    test_env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df_test)])
    
    # 运行测试
    obs = test_env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < len(df_test) - 10:
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
        total_reward += rewards[0]
        steps += 1
    
    final_worth = info[0]['net_worth']
    profit = final_worth - 10000
    profit_pct = (profit / 10000) * 100
    
    print(f"\n测试结果:")
    print(f"  交易步数: {steps}")
    print(f"  最终资产: {final_worth:.2f} 元")
    print(f"  总利润: {profit:.2f} 元")
    print(f"  收益率: {profit_pct:.2f}%")
    
    if profit > 0:
        print(f"  ✅ 盈利!")
    else:
        print(f"  ⚠️ 亏损")

else:
    print(f"⚠ 测试文件不存在: {test_file}")

# 完成
print("\n" + "="*70)
print("✅ 训练完成!")
print("="*70)

print(f"\n模型文件: {MODEL_NAME}.zip")
print(f"观察空间: 32维 (包含技术指标)")
print(f"环境类型: StockTradingEnvEnhanced")

print("\n下一步:")
print("  1. 模型已自动保存")
print("  2. 运行测试: python quick_test.py")
print("  3. 开始选股: python simple_selector.py")

print("\n💡 提示:")
print("  - 训练更多步数可以获得更好效果")
print("  - 推荐: 100,000 - 200,000 步")
print("  - 使用: python train_enhanced_32d.py 100000")
