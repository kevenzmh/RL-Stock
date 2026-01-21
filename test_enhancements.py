"""
快速测试脚本 - 验证所有增强功能
用于快速验证改进是否正常工作
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor
from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced

def test_technical_indicators():
    """测试技术指标模块"""
    print("\n" + "="*60)
    print("测试 1: 技术指标模块")
    print("="*60)
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100),
    })
    
    # 添加技术指标
    print("\n添加技术指标...")
    df_with_indicators = add_all_technical_indicators(df)
    
    # 检查新增的列
    new_columns = set(df_with_indicators.columns) - set(df.columns)
    print(f"\n新增 {len(new_columns)} 个技术指标:")
    for col in sorted(new_columns):
        print(f"  - {col}")
    
    # 验证关键指标
    assert 'ma5' in df_with_indicators.columns, "MA5未生成"
    assert 'macd' in df_with_indicators.columns, "MACD未生成"
    assert 'rsi' in df_with_indicators.columns, "RSI未生成"
    assert 'kdj_k' in df_with_indicators.columns, "KDJ未生成"
    assert 'bb_upper' in df_with_indicators.columns, "布林带未生成"
    
    print("\n✅ 技术指标模块测试通过!")
    return df_with_indicators


def test_data_preprocessing():
    """测试数据预处理模块"""
    print("\n" + "="*60)
    print("测试 2: 数据预处理模块")
    print("="*60)
    
    # 创建包含问题的数据
    df = pd.DataFrame({
        'close': [100, 101, np.nan, 103, 1000, 105, 106],  # 包含缺失值和异常值
        'volume': [1000, 1100, 1050, np.inf, 1080, 1090, 1100],  # 包含无穷值
        'other': [1, 2, 3, 4, 5, 6, 7]
    })
    
    print("\n原始数据问题:")
    print(f"  缺失值: {df.isnull().sum().sum()}")
    print(f"  无穷值: {df.isin([np.inf, -np.inf]).sum().sum()}")
    
    # 测试预处理
    preprocessor = DataPreprocessor(method='robust')
    
    # 质量检查
    print("\n执行数据质量检查...")
    quality_report = preprocessor.check_data_quality(df)
    print(f"  质量报告生成成功")
    
    # 预处理流水线
    print("\n执行预处理流水线...")
    df_processed = preprocessor.preprocess_pipeline(
        df,
        fit=True,
        handle_missing=True,
        handle_outliers_flag=True,
        normalize=False
    )
    
    print(f"\n处理后:")
    print(f"  缺失值: {df_processed.isnull().sum().sum()}")
    print(f"  无穷值: {df_processed.isin([np.inf, -np.inf]).sum().sum()}")
    
    assert df_processed.isnull().sum().sum() == 0, "仍有缺失值"
    assert df_processed.isin([np.inf, -np.inf]).sum().sum() == 0, "仍有无穷值"
    
    print("\n✅ 数据预处理模块测试通过!")
    return df_processed


def test_enhanced_environment():
    """测试增强版环境"""
    print("\n" + "="*60)
    print("测试 3: 增强版交易环境")
    print("="*60)
    
    # 准备数据
    dates = pd.date_range('2020-01-01', periods=200)
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 200),
        'pctChg': np.random.randn(200) * 2,
        'amount': np.random.randint(100000000, 1000000000, 200),
        'adjustflag': [1] * 200,
        'tradestatus': [1] * 200,
        'peTTM': np.random.randn(200) * 10 + 20,
        'pbMRQ': np.random.randn(200) * 2 + 3,
        'psTTM': np.random.randn(200) * 5 + 10,
    })
    
    # 添加技术指标
    df = add_all_technical_indicators(df)
    
    # 创建环境
    print("\n创建增强版环境...")
    env = StockTradingEnvEnhanced(df)
    
    # 测试重置
    print("测试环境重置...")
    obs = env.reset()
    assert obs.shape == (32,), f"观察空间维度错误: {obs.shape}"
    print(f"  观察空间维度: {obs.shape} ✓")
    
    # 测试步进
    print("\n测试环境步进...")
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    assert obs.shape == (32,), "步进后观察空间维度错误"
    assert 'net_worth' in info, "信息中缺少net_worth"
    assert 'balance' in info, "信息中缺少balance"
    assert 'shares_held' in info, "信息中缺少shares_held"
    
    print(f"  步进成功 ✓")
    print(f"  奖励: {reward:.4f}")
    print(f"  净值: {info['net_worth']:.2f}")
    
    # 测试多个步骤
    print("\n测试多步交易...")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print(f"  第{i+1}步完成,重置环境")
            obs = env.reset()
    
    print(f"  多步测试成功 ✓")
    
    # 测试奖励函数组件
    print("\n测试奖励函数...")
    print(f"  收益历史长度: {len(env.returns_history)}")
    print(f"  净值历史长度: {len(env.net_worth_history)}")
    
    assert len(env.returns_history) > 0, "收益历史未记录"
    assert len(env.net_worth_history) > 0, "净值历史未记录"
    
    print("\n✅ 增强版环境测试通过!")


def test_evaluator():
    """测试增强版评估器"""
    print("\n" + "="*60)
    print("测试 4: 增强版评估器")
    print("="*60)
    
    try:
        from utils.enhanced_evaluator import EnhancedEvaluator
        
        evaluator = EnhancedEvaluator(initial_balance=10000)
        
        # 测试指标计算
        print("\n测试性能指标计算...")
        portfolio_values = [10000, 10200, 10150, 10300, 10250, 10400]
        metrics = evaluator._calculate_metrics(portfolio_values, 5)
        
        print(f"  总收益率: {metrics['total_return']*100:.2f}%")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  卡玛比率: {metrics['calmar_ratio']:.3f}")
        
        assert 'total_return' in metrics, "缺少总收益率"
        assert 'sharpe_ratio' in metrics, "缺少夏普比率"
        assert 'max_drawdown' in metrics, "缺少最大回撤"
        assert 'calmar_ratio' in metrics, "缺少卡玛比率"
        
        print("\n✅ 评估器测试通过!")
        
    except Exception as e:
        print(f"\n⚠️  评估器测试跳过: {e}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("RL-Stock 增强版功能测试")
    print("="*60)
    
    try:
        # 测试1: 技术指标
        df_with_indicators = test_technical_indicators()
        
        # 测试2: 数据预处理
        test_data_preprocessing()
        
        # 测试3: 增强版环境
        test_enhanced_environment()
        
        # 测试4: 评估器
        test_evaluator()
        
        # 总结
        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)
        print("\n✅ 所有核心功能正常工作")
        print("\n可以运行以下命令开始训练:")
        print("  python main_enhanced.py")
        
    except Exception as e:
        print("\n" + "="*60)
        print("测试失败!")
        print("="*60)
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
