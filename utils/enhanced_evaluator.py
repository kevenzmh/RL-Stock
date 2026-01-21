"""
增强版评估工具
包含:
1. 滚动窗口测试 (Walk-forward Analysis)
2. 不同市场环境测试
3. Monte Carlo 模拟
4. 完整的性能指标计算
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class EnhancedEvaluator:
    """增强版策略评估器"""
    
    def __init__(self, initial_balance=10000, risk_free_rate=0.03):
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.results = []
        
    def walk_forward_test(self, model, env_class, df_full, 
                         train_window=252, test_window=63, 
                         step_size=21):
        """
        滚动窗口测试 (Walk-forward Analysis)
        
        Args:
            model: 训练好的模型
            env_class: 环境类
            df_full: 完整数据集
            train_window: 训练窗口大小(天)
            test_window: 测试窗口大小(天)
            step_size: 滚动步长(天)
        
        Returns:
            测试结果列表
        """
        print("\n" + "="*60)
        print("滚动窗口测试 (Walk-forward Analysis)")
        print("="*60)
        print(f"训练窗口: {train_window}天")
        print(f"测试窗口: {test_window}天")
        print(f"滚动步长: {step_size}天")
        
        results = []
        total_windows = (len(df_full) - train_window - test_window) // step_size
        
        for i in range(0, len(df_full) - train_window - test_window, step_size):
            window_num = i // step_size + 1
            print(f"\n窗口 {window_num}/{total_windows}")
            
            # 训练集
            train_start = i
            train_end = i + train_window
            df_train = df_full.iloc[train_start:train_end].reset_index(drop=True)
            
            # 测试集
            test_start = train_end
            test_end = test_start + test_window
            df_test = df_full.iloc[test_start:test_end].reset_index(drop=True)
            
            print(f"  训练期: {df_train['date'].iloc[0]} 至 {df_train['date'].iloc[-1]}")
            print(f"  测试期: {df_test['date'].iloc[0]} 至 {df_test['date'].iloc[-1]}")
            
            # 在测试集上评估
            test_env = env_class(df_test)
            obs = test_env.reset()
            
            portfolio_values = [self.initial_balance]
            actions_taken = []
            
            done = False
            step = 0
            while not done and step < len(df_test) - 1:
                action, _ = model.predict(obs)
                obs, reward, done, info = test_env.step(action)
                
                portfolio_values.append(info['net_worth'])
                actions_taken.append(action)
                step += 1
            
            # 计算该窗口的性能指标
            window_metrics = self._calculate_metrics(
                portfolio_values, 
                test_window
            )
            window_metrics['window'] = window_num
            window_metrics['train_start'] = df_train['date'].iloc[0]
            window_metrics['train_end'] = df_train['date'].iloc[-1]
            window_metrics['test_start'] = df_test['date'].iloc[0]
            window_metrics['test_end'] = df_test['date'].iloc[-1]
            
            results.append(window_metrics)
            
            print(f"  收益率: {window_metrics['total_return']*100:.2f}%")
            print(f"  夏普比率: {window_metrics['sharpe_ratio']:.3f}")
            print(f"  最大回撤: {window_metrics['max_drawdown']*100:.2f}%")
        
        # 汇总统计
        self._print_walk_forward_summary(results)
        
        return results
    
    def test_different_market_conditions(self, model, env_class, df_full):
        """
        测试不同市场环境下的表现
        分为: 牛市、熊市、震荡市
        
        Args:
            model: 训练好的模型
            env_class: 环境类
            df_full: 完整数据集
        
        Returns:
            不同市场条件下的结果
        """
        print("\n" + "="*60)
        print("不同市场环境测试")
        print("="*60)
        
        # 计算市场趋势
        df_full['returns'] = df_full['close'].pct_change()
        df_full['cumulative_return'] = (1 + df_full['returns']).cumprod() - 1
        
        # 定义市场类型 (简化版本)
        # 牛市: 累计收益率 > 10%
        # 熊市: 累计收益率 < -10%
        # 震荡: 其他
        
        market_conditions = {
            'bull': df_full[df_full['cumulative_return'] > 0.10],
            'bear': df_full[df_full['cumulative_return'] < -0.10],
            'sideways': df_full[(df_full['cumulative_return'] >= -0.10) & 
                               (df_full['cumulative_return'] <= 0.10)]
        }
        
        results = {}
        
        for market_type, df_market in market_conditions.items():
            if len(df_market) < 50:  # 数据太少,跳过
                print(f"\n{market_type.upper()}市场数据不足,跳过")
                continue
                
            print(f"\n{market_type.upper()}市场测试")
            print(f"  数据点数: {len(df_market)}")
            
            # 重置索引
            df_market = df_market.reset_index(drop=True)
            
            # 测试
            env = env_class(df_market)
            obs = env.reset()
            
            portfolio_values = [self.initial_balance]
            done = False
            step = 0
            
            while not done and step < len(df_market) - 1:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                portfolio_values.append(info['net_worth'])
                step += 1
            
            # 计算指标
            metrics = self._calculate_metrics(portfolio_values, len(df_market))
            metrics['market_type'] = market_type
            metrics['sample_size'] = len(df_market)
            
            results[market_type] = metrics
            
            print(f"  收益率: {metrics['total_return']*100:.2f}%")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.3f}")
            print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
            print(f"  卡玛比率: {metrics['calmar_ratio']:.3f}")
        
        return results
    
    def monte_carlo_simulation(self, model, env_class, df_test, 
                               n_simulations=100, random_start=True):
        """
        Monte Carlo 模拟
        通过多次随机起始点运行来评估策略稳定性
        
        Args:
            model: 训练好的模型
            env_class: 环境类
            df_test: 测试数据
            n_simulations: 模拟次数
            random_start: 是否使用随机起始点
        
        Returns:
            模拟结果
        """
        print("\n" + "="*60)
        print(f"Monte Carlo 模拟 (运行 {n_simulations} 次)")
        print("="*60)
        
        all_returns = []
        all_sharpe_ratios = []
        all_max_drawdowns = []
        all_final_values = []
        
        for sim in range(n_simulations):
            if (sim + 1) % 10 == 0:
                print(f"进度: {sim + 1}/{n_simulations}")
            
            # 创建环境
            env = env_class(df_test)
            
            # 如果使用随机起始,修改环境的起始点
            if random_start:
                min_start = 60
                max_start = max(min_start, len(df_test) - 100)
                env.current_step = np.random.randint(min_start, max_start)
            
            obs = env.reset()
            
            portfolio_values = [self.initial_balance]
            done = False
            step = 0
            max_steps = len(df_test) - env.current_step - 1
            
            while not done and step < max_steps:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                portfolio_values.append(info['net_worth'])
                step += 1
            
            # 计算该次模拟的指标
            if len(portfolio_values) > 1:
                metrics = self._calculate_metrics(portfolio_values, step)
                all_returns.append(metrics['total_return'])
                all_sharpe_ratios.append(metrics['sharpe_ratio'])
                all_max_drawdowns.append(metrics['max_drawdown'])
                all_final_values.append(portfolio_values[-1])
        
        # 统计结果
        results = {
            'n_simulations': n_simulations,
            'returns': {
                'mean': np.mean(all_returns),
                'std': np.std(all_returns),
                'min': np.min(all_returns),
                'max': np.max(all_returns),
                'percentile_5': np.percentile(all_returns, 5),
                'percentile_25': np.percentile(all_returns, 25),
                'percentile_50': np.percentile(all_returns, 50),
                'percentile_75': np.percentile(all_returns, 75),
                'percentile_95': np.percentile(all_returns, 95),
            },
            'sharpe_ratio': {
                'mean': np.mean(all_sharpe_ratios),
                'std': np.std(all_sharpe_ratios),
                'min': np.min(all_sharpe_ratios),
                'max': np.max(all_sharpe_ratios),
            },
            'max_drawdown': {
                'mean': np.mean(all_max_drawdowns),
                'std': np.std(all_max_drawdowns),
                'min': np.min(all_max_drawdowns),
                'max': np.max(all_max_drawdowns),
            },
            'final_value': {
                'mean': np.mean(all_final_values),
                'std': np.std(all_final_values),
                'min': np.min(all_final_values),
                'max': np.max(all_final_values),
            },
            'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns),
        }
        
        self._print_monte_carlo_summary(results)
        
        return results, all_returns, all_sharpe_ratios, all_max_drawdowns
    
    def _calculate_metrics(self, portfolio_values, days):
        """计算性能指标"""
        # 计算收益率序列
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        if len(returns) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'volatility': 0,
            }
        
        # 总收益率
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        
        # 年化收益率
        years = days / 252
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # 波动率
        volatility = np.std(returns) * np.sqrt(252)
        
        # 夏普比率
        if volatility > 1e-6:
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # 卡玛比率
        if max_drawdown > 1e-6:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
        }
    
    def _print_walk_forward_summary(self, results):
        """打印滚动窗口测试汇总"""
        print("\n" + "="*60)
        print("滚动窗口测试汇总")
        print("="*60)
        
        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        
        print(f"\n总窗口数: {len(results)}")
        print(f"\n收益率统计:")
        print(f"  平均: {np.mean(returns)*100:.2f}%")
        print(f"  标准差: {np.std(returns)*100:.2f}%")
        print(f"  最小: {np.min(returns)*100:.2f}%")
        print(f"  最大: {np.max(returns)*100:.2f}%")
        print(f"  胜率: {sum(1 for r in returns if r > 0) / len(returns)*100:.2f}%")
        
        print(f"\n夏普比率统计:")
        print(f"  平均: {np.mean(sharpes):.3f}")
        print(f"  标准差: {np.std(sharpes):.3f}")
        print(f"  最小: {np.min(sharpes):.3f}")
        print(f"  最大: {np.max(sharpes):.3f}")
        
        print(f"\n最大回撤统计:")
        print(f"  平均: {np.mean(drawdowns)*100:.2f}%")
        print(f"  标准差: {np.std(drawdowns)*100:.2f}%")
        print(f"  最小: {np.min(drawdowns)*100:.2f}%")
        print(f"  最大: {np.max(drawdowns)*100:.2f}%")
    
    def _print_monte_carlo_summary(self, results):
        """打印Monte Carlo模拟汇总"""
        print("\n" + "="*60)
        print("Monte Carlo 模拟结果")
        print("="*60)
        
        print(f"\n模拟次数: {results['n_simulations']}")
        print(f"胜率: {results['win_rate']*100:.2f}%")
        
        print(f"\n收益率分布:")
        print(f"  均值: {results['returns']['mean']*100:.2f}%")
        print(f"  标准差: {results['returns']['std']*100:.2f}%")
        print(f"  最小值: {results['returns']['min']*100:.2f}%")
        print(f"  最大值: {results['returns']['max']*100:.2f}%")
        print(f"  5%分位: {results['returns']['percentile_5']*100:.2f}%")
        print(f"  25%分位: {results['returns']['percentile_25']*100:.2f}%")
        print(f"  50%分位(中位数): {results['returns']['percentile_50']*100:.2f}%")
        print(f"  75%分位: {results['returns']['percentile_75']*100:.2f}%")
        print(f"  95%分位: {results['returns']['percentile_95']*100:.2f}%")
        
        print(f"\n夏普比率:")
        print(f"  均值: {results['sharpe_ratio']['mean']:.3f}")
        print(f"  标准差: {results['sharpe_ratio']['std']:.3f}")
        
        print(f"\n最大回撤:")
        print(f"  均值: {results['max_drawdown']['mean']*100:.2f}%")
        print(f"  标准差: {results['max_drawdown']['std']*100:.2f}%")
        
        print(f"\n最终资产:")
        print(f"  均值: ¥{results['final_value']['mean']:,.2f}")
        print(f"  标准差: ¥{results['final_value']['std']:,.2f}")
