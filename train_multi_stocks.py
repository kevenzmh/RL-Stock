"""
多股票训练 - 32维增强模型
使用多只不同类型股票训练，提高泛化能力
"""
import os
import sys
import pandas as pd
import numpy as np
import random
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced
from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor

print("="*70)
print("🚀 多股票训练 - 32维增强模型")
print("="*70)

# 训练配置
TOTAL_STEPS = 100000  # 10万步（多股票需要更多步数）
MODEL_NAME = 'models/ppo2_enhanced_32d_multi'

# 选择多样化的股票池
TRAINING_STOCKS = [
    # 银行股（价值股）
    'sh.600036',  # 招商银行
    'sh.601398',  # 工商银行
    
    # 科技股（成长股）
    'sz.300750',  # 宁德时代
    'sz.000063',  # 中兴通讯
    
    # 消费股
    'sh.600519',  # 贵州茅台
    'sz.000858',  # 五粮液
    
    # 医药股
    'sh.600276',  # 恒瑞医药
    'sz.000538',  # 云南白药
    
    # 地产股
    'sz.000002',  # 万科A
    'sz.000001',  # 平安银行
]

print(f"\n配置:")
print(f"  训练股票数: {len(TRAINING_STOCKS)} 只")
print(f"  训练步数: {TOTAL_STEPS:,}")
print(f"  模型保存: {MODEL_NAME}")

# 1. 加载和准备所有股票数据
print("\n" + "-"*70)
print("【步骤1/5】加载多只股票训练数据")
print("-"*70)

all_dfs = []
preprocessor = DataPreprocessor(method='robust')

for i, stock_code in enumerate(TRAINING_STOCKS, 1):
    stock_file = f'stockdata/train/{stock_code}.csv'
    
    if not os.path.exists(stock_file):
        print(f"[{i}/{len(TRAINING_STOCKS)}] ⚠ {stock_code}: 文件不存在，跳过")
        continue
    
    try:
        df = pd.read_csv(stock_file)
        df = df.sort_values('date').reset_index(drop=True)
        
        # 添加股票代码标识
        df['stock_code'] = stock_code
        
        print(f"[{i}/{len(TRAINING_STOCKS)}] ✓ {stock_code}: {len(df)} 条")
        all_dfs.append(df)
    
    except Exception as e:
        print(f"[{i}/{len(TRAINING_STOCKS)}] ✗ {stock_code}: {e}")

if len(all_dfs) == 0:
    print("\n✗ 没有成功加载任何股票数据！")
    sys.exit(1)

print(f"\n✓ 成功加载 {len(all_dfs)} 只股票")
total_rows = sum(len(df) for df in all_dfs)
print(f"✓ 总数据量: {total_rows:,} 条")

# 2. 添加技术指标
print("\n" + "-"*70)
print("【步骤2/5】添加技术指标")
print("-"*70)

processed_dfs = []
for i, df in enumerate(all_dfs, 1):
    stock_code = df['stock_code'].iloc[0]
    try:
        df = add_all_technical_indicators(df)
        processed_dfs.append(df)
        print(f"[{i}/{len(all_dfs)}] ✓ {stock_code}: {len(df.columns)} 列")
    except Exception as e:
        print(f"[{i}/{len(all_dfs)}] ✗ {stock_code}: {e}")

all_dfs = processed_dfs

# 3. 数据预处理
print("\n" + "-"*70)
print("【步骤3/5】数据预处理")
print("-"*70)

# 对每只股票单独预处理
processed_dfs = []
for i, df in enumerate(all_dfs, 1):
    stock_code = df['stock_code'].iloc[0]
    try:
        df_processed = preprocessor.preprocess_pipeline(
            df.copy(),
            fit=(i == 1),  # 第一只股票拟合，其他使用相同参数
            handle_missing=True,
            handle_outliers_flag=True,
            normalize=False
        )
        processed_dfs.append(df_processed)
        print(f"[{i}/{len(all_dfs)}] ✓ {stock_code}: {len(df_processed)} 条")
    except Exception as e:
        print(f"[{i}/{len(all_dfs)}] ✗ {stock_code}: {e}")

all_dfs = processed_dfs

# 4. 创建多环境训练
print("\n" + "-"*70)
print("【步骤4/5】创建训练环境")
print("-"*70)

print(f"创建 {len(all_dfs)} 个并行环境...")

# 为每只股票创建环境
def make_env(df):
    def _init():
        return StockTradingEnvEnhanced(df)
    return _init

# 创建多个环境（每只股票一个）
env_fns = [make_env(df) for df in all_dfs]
env = DummyVecEnv(env_fns)

print(f"✓ 环境创建成功: {len(all_dfs)} 个并行环境")
print(f"  观察空间: 32维")
print(f"  动作空间: Box(2,)")

# 5. 创建和训练模型
print("\n" + "-"*70)
print("【步骤5/5】训练模型")
print("-"*70)

print("初始化PPO2模型...")
model = PPO2(
    MlpPolicy, 
    env, 
    verbose=1,
    tensorboard_log='./log_multi_32d/',
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

print(f"\n开始训练 {TOTAL_STEPS:,} 步...")
print(f"训练股票: {len(all_dfs)} 只")
print(f"预计时间: 1-1.5小时")
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

# 7. 测试模型
print("\n" + "-"*70)
print("测试模型")
print("-"*70)

# 在测试集上测试几只股票
test_stocks = ['sh.600036', 'sz.300750', 'sh.600519']
test_results = []

for stock_code in test_stocks:
    test_file = f'stockdata/test/{stock_code}.csv'
    
    if not os.path.exists(test_file):
        continue
    
    try:
        df_test = pd.read_csv(test_file)
        df_test = df_test.sort_values('date').reset_index(drop=True)
        df_test = df_test.tail(60)
        
        df_test = add_all_technical_indicators(df_test)
        df_test = preprocessor.preprocess_pipeline(
            df_test,
            fit=False,
            handle_missing=True,
            handle_outliers_flag=True,
            normalize=False
        )
        
        test_env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df_test)])
        
        obs = test_env.reset()
        done = False
        steps = 0
        
        while not done and steps < len(df_test) - 10:
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            steps += 1
        
        final_worth = info[0]['net_worth']
        profit = final_worth - 10000
        profit_pct = (profit / 10000) * 100
        
        test_results.append({
            'stock': stock_code,
            'profit': profit,
            'return': profit_pct
        })
        
        print(f"  {stock_code}: {profit_pct:>6.2f}% ({profit:>8.2f}元)")
    
    except Exception as e:
        print(f"  {stock_code}: ✗ {e}")

# 统计
if len(test_results) > 0:
    avg_return = np.mean([r['return'] for r in test_results])
    win_rate = len([r for r in test_results if r['return'] > 0]) / len(test_results) * 100
    
    print(f"\n  平均收益: {avg_return:.2f}%")
    print(f"  胜率: {win_rate:.1f}%")

# 完成
print("\n" + "="*70)
print("✅ 多股票训练完成!")
print("="*70)

print(f"\n模型文件: {MODEL_NAME}.zip")
print(f"训练股票: {len(all_dfs)} 只")
print(f"训练数据: {total_rows:,} 条")
print(f"观察空间: 32维")

print("\n下一步:")
print("  1. 运行测试: python quick_test.py")
print("  2. 开始选股: python simple_selector.py")

print("\n💡 优势:")
print("  ✓ 多样化训练 → 更好的泛化能力")
print("  ✓ 不同行业 → 适应各种股票")
print("  ✓ 更多数据 → 更稳定的模型")
