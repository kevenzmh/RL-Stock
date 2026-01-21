"""
增强版主程序
整合所有改进:
1. 技术指标特征
2. 改进的奖励函数(夏普比率)
3. 增强的数据预处理
4. 完整的评估体系(滚动窗口、不同市场环境、Monte Carlo)
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced
from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor
from utils.enhanced_evaluator import EnhancedEvaluator

# 设置中文字体
font = fm.FontProperties(fname='font/wqy-microhei.ttc')
plt.rcParams['axes.unicode_minus'] = False


def prepare_data_with_indicators(stock_file, method='robust'):
    """
    准备数据:加载、添加技术指标、预处理
    
    Args:
        stock_file: 股票数据文件路径
        method: 标准化方法
    
    Returns:
        处理后的DataFrame
    """
    print("\n" + "="*60)
    print("数据准备")
    print("="*60)
    
    # 1. 加载数据
    df = pd.read_csv(stock_file)
    df = df.sort_values('date').reset_index(drop=True)
    print(f"原始数据: {len(df)} 条记录")
    print(f"时间范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
    
    # 2. 添加技术指标
    print("\n添加技术指标...")
    df = add_all_technical_indicators(df)
    print("技术指标添加完成!")
    print(f"  总特征数: {len(df.columns)}")
    
    # 3. 数据预处理
    print("\n数据预处理...")
    preprocessor = DataPreprocessor(method=method)
    
    # 处理缺失值(技术指标可能产生NaN)
    df = preprocessor.handle_missing_values(df, strategy='interpolate')
    
    # 处理异常值
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    df = preprocessor.handle_outliers(df, columns=price_cols, method='clip')
    
    print(f"\n预处理后数据: {len(df)} 条记录")
    print(f"包含特征: {', '.join(df.columns[:10])}...")
    
    return df


def train_enhanced_model(df_train, train_steps=100000, log_name='enhanced'):
    """
    训练增强版模型
    
    Args:
        df_train: 训练数据
        train_steps: 训练步数
        log_name: 日志名称
    
    Returns:
        训练好的模型
    """
    print("\n" + "="*60)
    print("模型训练")
    print("="*60)
    print(f"训练步数: {train_steps}")
    
    # 创建环境
    env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df_train)])
    
    # 日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_enhanced')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建模型 - 使用优化的超参数
    model = PPO2(
        MlpPolicy, 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,      # 学习率
        n_steps=2048,               # 每次更新的步数
        nminibatches=32,            # minibatch数量
        noptepochs=10,              # 优化轮数
        gamma=0.99,                 # 折扣因子
        lam=0.95,                   # GAE lambda
        cliprange=0.2,              # PPO裁剪参数
        ent_coef=0.01,              # 熵系数(鼓励探索)
        vf_coef=0.5,                # 价值函数系数
        max_grad_norm=0.5,          # 梯度裁剪
    )
    
    print("\n开始训练...")
    model.learn(total_timesteps=train_steps)
    
    # 保存模型
    model_path = f'./models/ppo2_enhanced_{train_steps}.pkl'
    os.makedirs('./models', exist_ok=True)
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")
    
    return model


def comprehensive_evaluation(model, df_test, df_full, stock_code):
    """
    全面评估
    
    Args:
        model: 训练好的模型
        df_test: 测试数据
        df_full: 完整数据(用于滚动窗口测试)
        stock_code: 股票代码
    
    Returns:
        评估结果
    """
    print("\n" + "="*60)
    print("全面评估")
    print("="*60)
    
    evaluator = EnhancedEvaluator(initial_balance=10000)
    
    # 1. 基础测试
    print("\n1. 基础测试")
    print("-"*60)
    env_test = StockTradingEnvEnhanced(df_test)
    obs = env_test.reset()
    
    portfolio_values = [10000]
    actions_log = []
    
    done = False
    step = 0
    while not done and step < len(df_test) - 1:
        action, _ = model.predict(obs)
        obs, reward, done, info = env_test.step(action)
        
        portfolio_values.append(info['net_worth'])
        actions_log.append(action)
        step += 1
        
        if step % 50 == 0:
            print(f"  Step {step}/{len(df_test)}: Net Worth = ¥{info['net_worth']:,.2f}")
    
    # 计算基础指标
    basic_metrics = evaluator._calculate_metrics(portfolio_values, len(df_test))
    
    print(f"\n基础测试结果:")
    print(f"  最终资产: ¥{portfolio_values[-1]:,.2f}")
    print(f"  总收益率: {basic_metrics['total_return']*100:.2f}%")
    print(f"  夏普比率: {basic_metrics['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {basic_metrics['max_drawdown']*100:.2f}%")
    print(f"  卡玛比率: {basic_metrics['calmar_ratio']:.3f}")
    
    # 2. Monte Carlo 模拟
    print("\n2. Monte Carlo 模拟")
    print("-"*60)
    mc_results, mc_returns, mc_sharpes, mc_drawdowns = evaluator.monte_carlo_simulation(
        model, StockTradingEnvEnhanced, df_test, 
        n_simulations=50, random_start=True
    )
    
    # 3. 滚动窗口测试
    if len(df_full) >= 400:  # 确保有足够的数据
        print("\n3. 滚动窗口测试")
        print("-"*60)
        wf_results = evaluator.walk_forward_test(
            model, StockTradingEnvEnhanced, df_full,
            train_window=252,  # 1年训练
            test_window=63,    # 3个月测试
            step_size=21       # 每月滚动
        )
    else:
        print("\n数据量不足,跳过滚动窗口测试")
        wf_results = None
    
    # 4. 不同市场环境测试
    print("\n4. 不同市场环境测试")
    print("-"*60)
    market_results = evaluator.test_different_market_conditions(
        model, StockTradingEnvEnhanced, df_full
    )
    
    return {
        'basic': basic_metrics,
        'portfolio_values': portfolio_values,
        'monte_carlo': mc_results,
        'mc_returns': mc_returns,
        'mc_sharpes': mc_sharpes,
        'mc_drawdowns': mc_drawdowns,
        'walk_forward': wf_results,
        'market_conditions': market_results,
    }


def plot_comprehensive_results(results, stock_code, save_dir='./img'):
    """
    绘制综合评估结果
    
    Args:
        results: 评估结果字典
        stock_code: 股票代码
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 净值曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 净值曲线
    ax = axes[0, 0]
    ax.plot(results['portfolio_values'], linewidth=2, color='#2E86AB')
    ax.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='初始资金')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('交易日', fontproperties=font, fontsize=10)
    ax.set_ylabel('资产净值 (元)', fontproperties=font, fontsize=10)
    ax.set_title('资产净值曲线', fontproperties=font, fontsize=12, fontweight='bold')
    ax.legend(prop=font)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x:,.0f}'))
    
    # Monte Carlo 收益率分布
    ax = axes[0, 1]
    ax.hist(np.array(results['mc_returns'])*100, bins=30, 
            color='#A23B72', alpha=0.7, edgecolor='black')
    ax.axvline(x=results['basic']['total_return']*100, 
               color='red', linestyle='--', linewidth=2, label='实际收益')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('收益率 (%)', fontproperties=font, fontsize=10)
    ax.set_ylabel('频数', fontproperties=font, fontsize=10)
    ax.set_title('Monte Carlo 收益率分布', fontproperties=font, fontsize=12, fontweight='bold')
    ax.legend(prop=font)
    
    # Monte Carlo 夏普比率分布
    ax = axes[1, 0]
    ax.hist(results['mc_sharpes'], bins=30, 
            color='#F18F01', alpha=0.7, edgecolor='black')
    ax.axvline(x=results['basic']['sharpe_ratio'], 
               color='red', linestyle='--', linewidth=2, label='实际夏普比率')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('夏普比率', fontproperties=font, fontsize=10)
    ax.set_ylabel('频数', fontproperties=font, fontsize=10)
    ax.set_title('Monte Carlo 夏普比率分布', fontproperties=font, fontsize=12, fontweight='bold')
    ax.legend(prop=font)
    
    # 最大回撤分布
    ax = axes[1, 1]
    ax.hist(np.array(results['mc_drawdowns'])*100, bins=30, 
            color='#C73E1D', alpha=0.7, edgecolor='black')
    ax.axvline(x=results['basic']['max_drawdown']*100, 
               color='red', linestyle='--', linewidth=2, label='实际最大回撤')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('最大回撤 (%)', fontproperties=font, fontsize=10)
    ax.set_ylabel('频数', fontproperties=font, fontsize=10)
    ax.set_title('Monte Carlo 最大回撤分布', fontproperties=font, fontsize=12, fontweight='bold')
    ax.legend(prop=font)
    
    plt.suptitle(f'{stock_code} 策略综合评估', 
                 fontproperties=font, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{stock_code}_comprehensive.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n综合评估图表已保存: {save_path}")
    plt.close()
    
    # 2. 如果有滚动窗口结果,绘制
    if results['walk_forward'] is not None and len(results['walk_forward']) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        wf_results = results['walk_forward']
        windows = [r['window'] for r in wf_results]
        returns = [r['total_return']*100 for r in wf_results]
        sharpes = [r['sharpe_ratio'] for r in wf_results]
        drawdowns = [r['max_drawdown']*100 for r in wf_results]
        
        # 收益率
        ax = axes[0]
        ax.bar(windows, returns, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('窗口编号', fontproperties=font, fontsize=10)
        ax.set_ylabel('收益率 (%)', fontproperties=font, fontsize=10)
        ax.set_title('滚动窗口收益率', fontproperties=font, fontsize=12, fontweight='bold')
        
        # 夏普比率
        ax = axes[1]
        ax.bar(windows, sharpes, color='#F18F01', alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('窗口编号', fontproperties=font, fontsize=10)
        ax.set_ylabel('夏普比率', fontproperties=font, fontsize=10)
        ax.set_title('滚动窗口夏普比率', fontproperties=font, fontsize=12, fontweight='bold')
        
        # 最大回撤
        ax = axes[2]
        ax.bar(windows, drawdowns, color='#C73E1D', alpha=0.7, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('窗口编号', fontproperties=font, fontsize=10)
        ax.set_ylabel('最大回撤 (%)', fontproperties=font, fontsize=10)
        ax.set_title('滚动窗口最大回撤', fontproperties=font, fontsize=12, fontweight='bold')
        
        plt.suptitle(f'{stock_code} 滚动窗口测试结果', 
                     fontproperties=font, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{stock_code}_walk_forward.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"滚动窗口图表已保存: {save_path}")
        plt.close()


def find_file(path, name):
    """查找文件"""
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def run_complete_pipeline(stock_code='sh.000001', train_steps=100000):
    """
    运行完整的训练和评估流水线
    
    Args:
        stock_code: 股票代码
        train_steps: 训练步数
    """
    print("\n" + "="*60)
    print("增强版RL股票交易系统")
    print("="*60)
    print(f"股票代码: {stock_code}")
    print(f"训练步数: {train_steps}")
    print("="*60)
    
    # 1. 查找数据文件
    train_file = find_file('./stockdata/train', stock_code)
    test_file = find_file('./stockdata/test', stock_code)
    
    if not train_file or not test_file:
        print(f"错误: 找不到股票 {stock_code} 的数据文件")
        return
    
    print(f"\n训练数据: {train_file}")
    print(f"测试数据: {test_file}")
    
    # 2. 准备数据
    df_train = prepare_data_with_indicators(train_file, method='robust')
    df_test = prepare_data_with_indicators(test_file, method='robust')
    
    # 合并用于滚动窗口测试
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full = df_full.sort_values('date').reset_index(drop=True)
    
    # 3. 训练模型
    model = train_enhanced_model(df_train, train_steps=train_steps, log_name=stock_code)
    
    # 4. 全面评估
    results = comprehensive_evaluation(model, df_test, df_full, stock_code)
    
    # 5. 绘制结果
    plot_comprehensive_results(results, stock_code)
    
    # 6. 保存完整报告
    print("\n" + "="*60)
    print("最终评估报告")
    print("="*60)
    
    print("\n【基础指标】")
    print(f"  总收益率:     {results['basic']['total_return']*100:>8.2f}%")
    print(f"  年化收益率:   {results['basic']['annualized_return']*100:>8.2f}%")
    print(f"  夏普比率:     {results['basic']['sharpe_ratio']:>8.3f}")
    print(f"  最大回撤:     {results['basic']['max_drawdown']*100:>8.2f}%")
    print(f"  卡玛比率:     {results['basic']['calmar_ratio']:>8.3f}")
    print(f"  波动率:       {results['basic']['volatility']*100:>8.2f}%")
    
    print("\n【Monte Carlo 模拟 - 50次】")
    mc = results['monte_carlo']
    print(f"  平均收益率:   {mc['returns']['mean']*100:>8.2f}%")
    print(f"  收益率标准差: {mc['returns']['std']*100:>8.2f}%")
    print(f"  胜率:         {mc['win_rate']*100:>8.2f}%")
    print(f"  平均夏普比率: {mc['sharpe_ratio']['mean']:>8.3f}")
    print(f"  平均最大回撤: {mc['max_drawdown']['mean']*100:>8.2f}%")
    
    if results['walk_forward'] is not None:
        wf_returns = [r['total_return'] for r in results['walk_forward']]
        wf_sharpes = [r['sharpe_ratio'] for r in results['walk_forward']]
        
        print("\n【滚动窗口测试】")
        print(f"  窗口数量:     {len(results['walk_forward'])}")
        print(f"  平均收益率:   {np.mean(wf_returns)*100:>8.2f}%")
        print(f"  胜率:         {sum(1 for r in wf_returns if r > 0)/len(wf_returns)*100:>8.2f}%")
        print(f"  平均夏普比率: {np.mean(wf_sharpes):>8.3f}")
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == '__main__':
    # 运行完整流水线
    # 可以调整训练步数以获得更好的效果
    run_complete_pipeline('sh.000001', train_steps=100000)
    
    # 如果要测试其他股票:
    # run_complete_pipeline('sz.300677', train_steps=100000)
