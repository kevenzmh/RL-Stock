"""
改进的训练脚本
改进点:
1. 修复除零错误
2. 增加交易成本模拟
3. 延长训练时间至10万步
4. 支持GPU加速
5. 添加风险控制机制
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# 导入修复后的环境
from rlenv.StockTradingEnv_Fixed import StockTradingEnvFixed

# 设置中文字体
try:
    font = fm.FontProperties(fname='font/wqy-microhei.ttc')
except:
    font = None
    print("Warning: Chinese font not found, using default font")

plt.rcParams['axes.unicode_minus'] = False


def setup_gpu():
    """
    配置GPU加速
    需要安装: tensorflow-gpu==1.14.0 (对应CUDA 10.0)
    """
    import tensorflow as tf
    
    # 检查GPU是否可用
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # 设置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"Found {len(gpus)} GPU(s), GPU acceleration enabled!")
            print(f"GPU devices: {[gpu.name for gpu in gpus]}")
            
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPU found, using CPU")
        return False


def stock_trade_improved(stock_file, total_timesteps=100000, use_gpu=True):
    """
    改进的股票交易训练函数
    
    Args:
        stock_file: 训练数据文件路径
        total_timesteps: 训练步数 (默认10万步)
        use_gpu: 是否使用GPU加速
    
    Returns:
        day_profits: 测试集每日利润
        model: 训练好的模型
    """
    # GPU设置
    if use_gpu:
        gpu_available = setup_gpu()
        if not gpu_available:
            print("Warning: GPU not available, falling back to CPU")
    
    # 读取训练数据
    df_train = pd.read_csv(stock_file)
    df_train = df_train.sort_values('date').reset_index(drop=True)
    
    print(f"Training data: {len(df_train)} rows")
    print(f"Date range: {df_train['date'].min()} to {df_train['date'].max()}")
    
    # 创建环境
    env = DummyVecEnv([lambda: StockTradingEnvFixed(df_train)])
    
    # 确保日志目录存在
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_improved')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting training with {total_timesteps:,} timesteps")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")
    
    # 创建PPO2模型
    # 使用更大的网络和更好的超参数
    model = PPO2(
        MlpPolicy, 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,      # 学习率
        n_steps=2048,            # 每次更新的步数
        nminibatches=32,         # 小批次数量
        noptepochs=10,           # 优化轮数
        gamma=0.99,              # 折扣因子
        ent_coef=0.01,          # 熵系数 (鼓励探索)
        cliprange=0.2,          # PPO裁剪范围
        policy_kwargs=dict(
            net_arch=[256, 256]  # 更大的网络
        )
    )
    
    # 训练模型
    print("Training started...")
    model.learn(total_timesteps=total_timesteps)
    print("Training completed!")
    
    # 保存模型
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'ppo2_stock_{total_timesteps}.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # 在测试集上评估
    test_file = stock_file.replace('train', 'test')
    
    if not os.path.exists(test_file):
        print(f"Warning: Test file not found: {test_file}")
        return [], model
    
    df_test = pd.read_csv(test_file)
    df_test = df_test.sort_values('date').reset_index(drop=True)
    
    print(f"\nTesting on {len(df_test)} rows")
    print(f"Date range: {df_test['date'].min()} to {df_test['date'].max()}")
    
    env_test = DummyVecEnv([lambda: StockTradingEnvFixed(df_test)])
    
    # 测试
    day_profits = []
    obs = env_test.reset()
    
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        
        # 获取利润
        profit = env_test.envs[0].net_worth - 10000
        day_profits.append(profit)
        
        if done[0]:
            break
    
    # 输出测试结果统计
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"Final profit: ¥{day_profits[-1]:.2f}")
    print(f"Max profit: ¥{max(day_profits):.2f}")
    print(f"Min profit: ¥{min(day_profits):.2f}")
    print(f"Return rate: {(day_profits[-1] / 10000 * 100):.2f}%")
    print(f"{'='*60}\n")
    
    return day_profits, model


def test_a_stock_improved(stock_code, total_timesteps=100000, use_gpu=True):
    """
    测试单个股票
    
    Args:
        stock_code: 股票代码
        total_timesteps: 训练步数
        use_gpu: 是否使用GPU
    """
    stock_file = find_file('./stockdata/train', str(stock_code))
    
    if not stock_file:
        print(f"Stock file not found for code: {stock_code}")
        return
    
    print(f"Training stock: {stock_code}")
    print(f"File: {stock_file}")
    
    # 训练和测试
    daily_profits, model = stock_trade_improved(
        stock_file, 
        total_timesteps=total_timesteps,
        use_gpu=use_gpu
    )
    
    if len(daily_profits) == 0:
        print("No test results available")
        return
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 利润曲线
    ax1.plot(daily_profits, '-', label=f'{stock_code} Profit', 
             linewidth=2, color='#2E86AB')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Profit (¥)', fontsize=12)
    ax1.set_title(f'Stock {stock_code} - Trading Profit (Training Steps: {total_timesteps:,})', 
                  fontsize=14, fontweight='bold')
    if font:
        ax1.legend(prop=font, fontsize=10)
    else:
        ax1.legend(fontsize=10)
    
    # 收益率曲线
    returns = [(p / 10000 * 100) for p in daily_profits]
    ax2.plot(returns, '-', label=f'{stock_code} Return Rate', 
             linewidth=2, color='#A23B72')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Return Rate (%)', fontsize=12)
    ax2.set_title('Return Rate Over Time', fontsize=14, fontweight='bold')
    if font:
        ax2.legend(prop=font, fontsize=10)
    else:
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    img_dir = './img'
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f'{stock_code}_improved_{total_timesteps}.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {img_path}")
    
    # 显示图片
    # plt.show()


def find_file(path, name):
    """查找文件"""
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)
    return None


def multi_stock_trade_improved(start_code=600000, max_num=100, total_timesteps=100000):
    """
    批量训练多个股票
    
    Args:
        start_code: 起始股票代码
        max_num: 训练股票数量
        total_timesteps: 每个股票的训练步数
    """
    group_result = []
    successful = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"Batch Training: {max_num} stocks")
    print(f"Code range: {start_code} - {start_code + max_num}")
    print(f"Training steps per stock: {total_timesteps:,}")
    print(f"{'='*60}\n")
    
    for i, code in enumerate(range(start_code, start_code + max_num)):
        print(f"\n[{i+1}/{max_num}] Processing stock: {code}")
        
        stock_file = find_file('./stockdata/train', str(code))
        
        if stock_file:
            try:
                profits, model = stock_trade_improved(
                    stock_file, 
                    total_timesteps=total_timesteps
                )
                group_result.append({
                    'code': code,
                    'profits': profits,
                    'final_profit': profits[-1] if len(profits) > 0 else 0
                })
                successful += 1
                print(f"✓ Success: {code}")
            except Exception as err:
                print(f"✗ Failed: {code} - Error: {err}")
                failed += 1
        else:
            print(f"✗ File not found: {code}")
            failed += 1
    
    # 保存结果
    result_file = f'batch_result_{start_code}_{start_code + max_num}_{total_timesteps}.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(group_result, f)
    
    print(f"\n{'='*60}")
    print(f"Batch Training Completed!")
    print(f"Successful: {successful}/{max_num}")
    print(f"Failed: {failed}/{max_num}")
    print(f"Results saved to: {result_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    """
    使用说明:
    
    1. 单股票训练 (推荐用于测试):
       python main_fixed.py
    
    2. 批量训练:
       取消注释 multi_stock_trade_improved() 行
    
    3. GPU加速:
       需要安装: pip install tensorflow-gpu==1.14.0
       CUDA版本: 10.0
       cuDNN版本: 7.6
    
    4. 训练步数调整:
       - 快速测试: 10000 步
       - 标准训练: 100000 步 (默认)
       - 深度训练: 500000 步
    """
    
    # 单股票训练示例
    test_a_stock_improved(
        stock_code='sh.000001',  # 上证指数
        total_timesteps=100000,   # 10万步训练
        use_gpu=True              # 使用GPU加速
    )
    
    # 批量训练示例 (取消注释以使用)
    # multi_stock_trade_improved(
    #     start_code=600000,
    #     max_num=10,
    #     total_timesteps=100000
    # )
