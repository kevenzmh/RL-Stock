"""
评估指标模块
包含各种交易策略评估指标
"""
import numpy as np
import pandas as pd
from typing import List, Dict


class PerformanceMetrics:
    """性能评估指标类"""
    
    def __init__(self, initial_balance=10000, risk_free_rate=0.03):
        """
        Args:
            initial_balance: 初始资金
            risk_free_rate: 无风险利率 (年化)
        """
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.portfolio_values = []
        self.returns = []
        self.trades = []
        
    def add_portfolio_value(self, value: float):
        """添加组合价值记录"""
        self.portfolio_values.append(value)
        
        if len(self.portfolio_values) > 1:
            ret = (value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(ret)
    
    def calculate_total_return(self) -> float:
        """计算总收益率"""
        if not self.portfolio_values:
            return 0.0
        return (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance
    
    def calculate_annualized_return(self, days: int) -> float:
        """
        计算年化收益率
        
        Args:
            days: 交易天数
        
        Returns:
            年化收益率
        """
        total_return = self.calculate_total_return()
        years = days / 252  # 假设一年252个交易日
        if years > 0:
            return (1 + total_return) ** (1 / years) - 1
        return 0.0
    
    def calculate_volatility(self, annualized=True) -> float:
        """
        计算波动率
        
        Args:
            annualized: 是否年化
        
        Returns:
            波动率
        """
        if len(self.returns) < 2:
            return 0.0
        
        vol = np.std(self.returns)
        if annualized:
            vol = vol * np.sqrt(252)
        return vol
    
    def calculate_sharpe_ratio(self, days: int) -> float:
        """
        计算夏普比率
        
        Args:
            days: 交易天数
        
        Returns:
            夏普比率
        """
        if len(self.returns) < 2:
            return 0.0
        
        annualized_return = self.calculate_annualized_return(days)
        volatility = self.calculate_volatility(annualized=True)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (annualized_return - self.risk_free_rate) / volatility
        return sharpe
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        计算最大回撤
        
        Returns:
            包含最大回撤、回撤开始和结束位置的字典
        """
        if not self.portfolio_values:
            return {'max_drawdown': 0.0, 'start_idx': 0, 'end_idx': 0}
        
        values = np.array(self.portfolio_values)
        
        # 计算累计最大值
        running_max = np.maximum.accumulate(values)
        
        # 计算回撤
        drawdown = (values - running_max) / running_max
        
        # 找到最大回撤
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # 找到回撤开始位置
        start_idx = np.argmax(values[:max_dd_idx+1])
        
        return {
            'max_drawdown': abs(max_dd),
            'start_idx': int(start_idx),
            'end_idx': int(max_dd_idx),
            'peak_value': float(values[start_idx]),
            'trough_value': float(values[max_dd_idx])
        }
    
    def calculate_calmar_ratio(self, days: int) -> float:
        """
        计算卡玛比率 (Calmar Ratio)
        年化收益率 / 最大回撤
        
        Args:
            days: 交易天数
        
        Returns:
            卡玛比率
        """
        annualized_return = self.calculate_annualized_return(days)
        max_dd_info = self.calculate_max_drawdown()
        max_dd = max_dd_info['max_drawdown']
        
        if max_dd == 0:
            return 0.0
        
        return annualized_return / max_dd
    
    def calculate_sortino_ratio(self, days: int) -> float:
        """
        计算索提诺比率
        只考虑下行波动率
        
        Args:
            days: 交易天数
        
        Returns:
            索提诺比率
        """
        if len(self.returns) < 2:
            return 0.0
        
        annualized_return = self.calculate_annualized_return(days)
        
        # 只计算负收益的标准差
        downside_returns = [r for r in self.returns if r < 0]
        if not downside_returns:
            return float('inf') if annualized_return > 0 else 0.0
        
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        
        if downside_vol == 0:
            return 0.0
        
        return (annualized_return - self.risk_free_rate) / downside_vol
    
    def calculate_win_rate(self) -> Dict[str, float]:
        """
        计算胜率相关指标
        
        Returns:
            包含胜率、平均盈利、平均亏损的字典
        """
        if len(self.returns) == 0:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0
            }
        
        wins = [r for r in self.returns if r > 0]
        losses = [r for r in self.returns if r < 0]
        
        win_rate = len(wins) / len(self.returns) if self.returns else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'total_trades': len(self.returns)
        }
    
    def get_summary(self, days: int) -> Dict:
        """
        获取完整的性能总结
        
        Args:
            days: 交易天数
        
        Returns:
            包含所有指标的字典
        """
        max_dd_info = self.calculate_max_drawdown()
        win_rate_info = self.calculate_win_rate()
        
        summary = {
            # 收益指标
            'total_return': self.calculate_total_return(),
            'annualized_return': self.calculate_annualized_return(days),
            'final_value': self.portfolio_values[-1] if self.portfolio_values else 0,
            
            # 风险指标
            'volatility': self.calculate_volatility(annualized=True),
            'max_drawdown': max_dd_info['max_drawdown'],
            'max_dd_duration': max_dd_info['end_idx'] - max_dd_info['start_idx'],
            
            # 风险调整收益
            'sharpe_ratio': self.calculate_sharpe_ratio(days),
            'calmar_ratio': self.calculate_calmar_ratio(days),
            'sortino_ratio': self.calculate_sortino_ratio(days),
            
            # 交易统计
            'win_rate': win_rate_info['win_rate'],
            'avg_win': win_rate_info['avg_win'],
            'avg_loss': win_rate_info['avg_loss'],
            'win_loss_ratio': win_rate_info['win_loss_ratio'],
            'total_trades': win_rate_info['total_trades'],
        }
        
        return summary
    
    def print_summary(self, days: int):
        """打印性能总结"""
        summary = self.get_summary(days)
        
        print("\n" + "=" * 60)
        print("策略性能评估报告".center(60))
        print("=" * 60)
        
        print("\n【收益指标】")
        print(f"  总收益率:     {summary['total_return']*100:>8.2f}%")
        print(f"  年化收益率:   {summary['annualized_return']*100:>8.2f}%")
        print(f"  最终资产:     ¥{summary['final_value']:>10,.2f}")
        
        print("\n【风险指标】")
        print(f"  年化波动率:   {summary['volatility']*100:>8.2f}%")
        print(f"  最大回撤:     {summary['max_drawdown']*100:>8.2f}%")
        print(f"  回撤持续:     {summary['max_dd_duration']:>8d} 天")
        
        print("\n【风险调整收益】")
        print(f"  夏普比率:     {summary['sharpe_ratio']:>8.3f}")
        print(f"  卡玛比率:     {summary['calmar_ratio']:>8.3f}")
        print(f"  索提诺比率:   {summary['sortino_ratio']:>8.3f}")
        
        print("\n【交易统计】")
        print(f"  总交易次数:   {summary['total_trades']:>8d}")
        print(f"  胜率:         {summary['win_rate']*100:>8.2f}%")
        print(f"  平均盈利:     {summary['avg_win']*100:>8.2f}%")
        print(f"  平均亏损:     {summary['avg_loss']*100:>8.2f}%")
        print(f"  盈亏比:       {summary['win_loss_ratio']:>8.2f}")
        
        print("\n" + "=" * 60)
