"""
SAC算法实现 - 替代PPO2
Soft Actor-Critic: 更高的样本效率和更好的性能

使用方法:
    python main_sac.py

特点:
1. Off-policy算法,样本效率更高
2. 最大熵强化学习,自动平衡探索-利用
3. 双Q网络,避免Q值过估计
4. 对超参数更鲁棒
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced
from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor
from utils.enhanced_evaluator import EnhancedEvaluator

# 设置中文字体
font = fm.FontProperties(fname='font/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def prepare_data(stock_file):
    """准备数据"""
    print(f"\n加载数据: {stock_file}")
    df = pd.read_csv(stock_file)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 添加技术指标
    df = add_all_technical_indicators(df)
    
    # 数据预处理
    preprocessor = DataPreprocessor(method='robust')
    df = preprocessor.handle_missing_values(df, strategy='interpolate')
    
    print(f"数据准备完成: {len(df)} 条记录")
    return df


def train_sac_model(df_train, train_steps=100000, log_name='sac'):
    """
    训练SAC模型
    
    Args:
        df_train: 训练数据
        train_steps: 训练步数
        log_name: 日志名称
    """
    print("\n" + "="*60)
    print("SAC模型训练")
    print("="*60)
    print(f"训练步数: {train_steps:,}")
    
    # 创建环境
    env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df_train)])
    
    # 日志和模型保存目录
    log_dir = os.path.join(os.path.dirname(__file__), 'log_sac')
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建回调函数
    # 1. 定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix='sac_checkpoint'
    )
    
    # 2. 评估回调(可选)
    eval_env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df_train)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # 创建SAC模型
    print("\n创建SAC模型...")
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,      # 大replay buffer,充分利用历史经验
        learning_starts=10000,       # 预填充buffer后才开始训练
        batch_size=256,             # 较大的batch size提高稳定性
        tau=0.005,                  # 软更新系数
        gamma=0.99,                 # 折扣因子
        train_freq=1,               # 每步都训练
        gradient_steps=1,           # 每次训练1步
        ent_coef='auto',           # 自动调节熵系数(重要!)
        target_update_interval=1,   # 每步都更新target网络
        target_entropy='auto',      # 自动目标熵
        use_sde=False,             # 不使用状态依赖探索
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        tensorboard_log=log_dir,
        verbose=1,
        device='auto'              # 自动选择GPU/CPU
    )
    
    print(f"\n策略网络结构:")
    print(f"  观察空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.shape}")
    print(f"  Buffer大小: {model.buffer_size:,}")
    print(f"  学习率: {model.learning_rate}")
    
    # 训练
    print("\n开始训练...")
    print("="*60)
    model.learn(
        total_timesteps=train_steps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )
    
    # 保存最终模型
    model_path = os.path.join(model_dir, f'sac_final_{train_steps}.zip')
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")
    
    return model


def evaluate_model(model, df_test, stock_code):
    """
    评估模型
    
    Args:
        model: 训练好的模型
        df_test: 测试数据
        stock_code: 股票代码
    """
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    evaluator = EnhancedEvaluator(initial_balance=10000)
    
    # 创建测试环境
    env = StockTradingEnvEnhanced(df_test)
    obs = env.reset()
    
    # 运行测试
    portfolio_values = [10000]
    actions_log = []
    
    done = False
    step = 0
    print("\n测试中...")
    
    while not done and step < len(df_test) - 1:
        # SAC使用确定性策略进行评估
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        portfolio_values.append(info['net_worth'])
        actions_log.append(action)
        step += 1
        
        if step % 20 == 0:
            print(f"  Step {step}/{len(df_test)}: Net Worth = ¥{info['net_worth']:,.2f}")
    
    # 计算指标
    metrics = evaluator._calculate_metrics(portfolio_values, len(df_test))
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"\n最终资产:     ¥{portfolio_values[-1]:,.2f}")
    print(f"总收益率:     {metrics['total_return']*100:.2f}%")
    print(f"年化收益率:   {metrics['annualized_return']*100:.2f}%")
    print(f"夏普比率:     {metrics['sharpe_ratio']:.3f}")
    print(f"最大回撤:     {metrics['max_drawdown']*100:.2f}%")
    print(f"卡玛比率:     {metrics['calmar_ratio']:.3f}")
    print(f"波动率:       {metrics['volatility']*100:.2f}%")
    
    # 绘制净值曲线
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, linewidth=2, color='#2E86AB', label='SAC策略')
    plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='初始资金')
    plt.grid(True, alpha=0.3)
    plt.xlabel('交易日', fontproperties=font, fontsize=12)
    plt.ylabel('资产净值 (元)', fontproperties=font, fontsize=12)
    plt.title(f'{stock_code} SAC策略回测', fontproperties=font, fontsize=14, fontweight='bold')
    plt.legend(prop=font, fontsize=10)
    plt.tight_layout()
    
    img_path = f'./img/{stock_code}_sac.png'
    os.makedirs('./img', exist_ok=True)
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {img_path}")
    plt.close()
    
    return metrics


def compare_with_ppo(stock_code, train_steps=100000):
    """
    对比SAC和PPO的性能
    """
    print("\n" + "="*60)
    print("SAC vs PPO 性能对比")
    print("="*60)
    
    # 数据准备
    train_file = f'./stockdata/train/{stock_code}.csv'
    test_file = f'./stockdata/test/{stock_code}.csv'
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"错误: 找不到股票 {stock_code} 的数据文件")
        return
    
    df_train = prepare_data(train_file)
    df_test = prepare_data(test_file)
    
    # 训练SAC
    print("\n[1/2] 训练SAC模型...")
    sac_model = train_sac_model(df_train, train_steps=train_steps, log_name=stock_code)
    
    # 评估SAC
    print("\n[2/2] 评估SAC模型...")
    sac_metrics = evaluate_model(sac_model, df_test, stock_code)
    
    # 如果有PPO模型,进行对比
    ppo_model_path = f'./models/ppo2_enhanced_{train_steps}.pkl'
    if os.path.exists(ppo_model_path):
        print("\n加载PPO模型进行对比...")
        from stable_baselines import PPO2
        ppo_model = PPO2.load(ppo_model_path)
        
        # 评估PPO
        env = StockTradingEnvEnhanced(df_test)
        obs = env.reset()
        
        ppo_values = [10000]
        done = False
        step = 0
        
        while not done and step < len(df_test) - 1:
            action, _ = ppo_model.predict(obs)
            obs, reward, done, info = env.step(action)
            ppo_values.append(info['net_worth'])
            step += 1
        
        evaluator = EnhancedEvaluator()
        ppo_metrics = evaluator._calculate_metrics(ppo_values, len(df_test))
        
        # 对比表
        print("\n" + "="*60)
        print("性能对比")
        print("="*60)
        print(f"{'指标':<20} {'SAC':>15} {'PPO':>15} {'提升':>15}")
        print("-"*65)
        
        def compare_metric(name, sac_val, ppo_val, is_percent=True, higher_better=True):
            if is_percent:
                sac_str = f"{sac_val*100:.2f}%"
                ppo_str = f"{ppo_val*100:.2f}%"
            else:
                sac_str = f"{sac_val:.3f}"
                ppo_str = f"{ppo_val:.3f}"
            
            if ppo_val != 0:
                improvement = (sac_val - ppo_val) / abs(ppo_val) * 100
                if not higher_better:
                    improvement = -improvement
                imp_str = f"{improvement:+.2f}%"
            else:
                imp_str = "N/A"
            
            print(f"{name:<20} {sac_str:>15} {ppo_str:>15} {imp_str:>15}")
        
        compare_metric("总收益率", sac_metrics['total_return'], ppo_metrics['total_return'])
        compare_metric("年化收益率", sac_metrics['annualized_return'], ppo_metrics['annualized_return'])
        compare_metric("夏普比率", sac_metrics['sharpe_ratio'], ppo_metrics['sharpe_ratio'], False)
        compare_metric("最大回撤", sac_metrics['max_drawdown'], ppo_metrics['max_drawdown'], higher_better=False)
        compare_metric("卡玛比率", sac_metrics['calmar_ratio'], ppo_metrics['calmar_ratio'], False)
        compare_metric("波动率", sac_metrics['volatility'], ppo_metrics['volatility'], higher_better=False)
        
        print("="*65)
    else:
        print("\n未找到PPO模型,跳过对比")
    
    print("\n完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC算法训练和评估')
    parser.add_argument('--stock', type=str, default='sh.000001', help='股票代码')
    parser.add_argument('--steps', type=int, default=100000, help='训练步数')
    parser.add_argument('--compare', action='store_true', help='与PPO对比')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_ppo(args.stock, args.steps)
    else:
        # 单独训练和评估
        train_file = f'./stockdata/train/{args.stock}.csv'
        test_file = f'./stockdata/test/{args.stock}.csv'
        
        df_train = prepare_data(train_file)
        df_test = prepare_data(test_file)
        
        model = train_sac_model(df_train, args.steps, args.stock)
        evaluate_model(model, df_test, args.stock)
