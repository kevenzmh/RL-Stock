"""
智能模型推理引擎
自动识别19维或32维模型
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlenv.StockTradingEnv0 import StockTradingEnv  # 19维
from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced  # 32维
from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor


class ModelInference:
    """智能模型推理引擎"""
    
    def __init__(self, model_path=None):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path or self._find_best_model()
        self.model = None
        self.preprocessor = DataPreprocessor(method='robust')
        self.initial_balance = 10000
        self.model_dim = None  # 模型维度（19或32）
        
        self._load_model()
        self._detect_model_dimension()
    
    def _find_best_model(self):
        """自动查找最佳模型"""
        models_dir = 'models'
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"模型目录不存在: {models_dir}")
        
        # 优先级顺序
        priority_models = [
            'ppo2_enhanced_32d.zip',       # 32维增强模型（最新）
            'ppo2_enhanced_100000.pkl',    # 旧版
            'quick_test_model.zip',        # 19维测试模型
            'ppo2_stock_100000.pkl',       # 旧版
        ]
        
        for model_name in priority_models:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                print(f"✓ 找到模型: {model_path}")
                return model_path
        
        raise FileNotFoundError("未找到可用的模型文件")
    
    def _load_model(self):
        """加载模型 - 支持多种格式"""
        print(f"\n加载模型: {self.model_path}")
        
        try:
            print("尝试 PPO2.load()...")
            self.model = PPO2.load(self.model_path)
            print("✓ 模型加载成功")
            return
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            raise Exception(f"模型加载失败: {e}")
    
    def _detect_model_dimension(self):
        """检测模型的观察空间维度"""
        try:
            obs_space = self.model.observation_space
            if hasattr(obs_space, 'shape'):
                self.model_dim = obs_space.shape[0]
                print(f"✓ 检测到模型维度: {self.model_dim}维")
                
                if self.model_dim == 19:
                    print("  → 使用原始环境 (StockTradingEnv)")
                elif self.model_dim == 32:
                    print("  → 使用增强环境 (StockTradingEnvEnhanced)")
                else:
                    print(f"  ⚠ 未知维度: {self.model_dim}")
            else:
                print("⚠ 无法检测模型维度，默认使用19维")
                self.model_dim = 19
        except Exception as e:
            print(f"⚠ 检测维度失败: {e}，默认使用19维")
            self.model_dim = 19
    
    def preprocess_data(self, df, fit=False):
        """
        预处理数据
        
        Args:
            df: 原始数据
            fit: 是否拟合预处理器
        
        Returns:
            处理后的DataFrame
        """
        # 根据模型维度决定是否添加技术指标
        if self.model_dim == 32:
            # 32维：添加技术指标
            df = add_all_technical_indicators(df)
        else:
            # 19维：确保有必需的基础列
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                            'tradestatus', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                for col in missing_cols:
                    df[col] = 0
            
            if 'adjustflag' not in df.columns:
                df['adjustflag'] = 3
        
        # 数据预处理
        df = self.preprocessor.preprocess_pipeline(
            df,
            fit=fit,
            handle_missing=True,
            handle_outliers_flag=False,
            normalize=False
        )
        
        df = df.reset_index(drop=True)
        return df
    
    def predict_single_stock(self, df, return_details=False):
        """
        预测单只股票
        
        Args:
            df: 股票数据 (已预处理)
            return_details: 是否返回详细信息
        
        Returns:
            dict: 预测结果
        """
        # 根据模型维度选择环境
        if self.model_dim == 32:
            env = DummyVecEnv([lambda: StockTradingEnvEnhanced(df)])
        else:
            env = DummyVecEnv([lambda: StockTradingEnv(df)])
        
        # 重置环境
        obs = env.reset()
        
        # 模拟交易
        done = False
        total_profit = 0
        max_profit = 0
        min_profit = 0
        actions_history = []
        rewards_history = []
        
        step = 0
        while not done:
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = env.step(action)
            
            actions_history.append(action[0])
            rewards_history.append(rewards[0])
            
            # 更新利润统计
            current_profit = info[0].get('net_worth', self.initial_balance) - self.initial_balance
            total_profit = current_profit
            max_profit = max(max_profit, current_profit)
            min_profit = min(min_profit, current_profit)
            
            step += 1
        
        # 计算综合得分
        final_net_worth = info[0].get('net_worth', self.initial_balance)
        total_return = (final_net_worth - self.initial_balance) / self.initial_balance
        
        # 计算波动率
        returns = np.diff([self.initial_balance] + [self.initial_balance + p for p in rewards_history])
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # 计算夏普比率
        sharpe = total_return / (volatility + 1e-6) if volatility > 0 else 0
        
        # 综合评分
        score = self._calculate_score(total_return, sharpe, max_profit, min_profit)
        
        result = {
            'score': score,
            'total_return': total_return * 100,
            'total_profit': total_profit,
            'max_profit': max_profit,
            'min_profit': min_profit,
            'sharpe_ratio': sharpe,
            'volatility': volatility,
            'final_net_worth': final_net_worth,
            'recommendation': self._get_recommendation(score, total_return)
        }
        
        if return_details:
            result['actions'] = actions_history
            result['rewards'] = rewards_history
            result['steps'] = step
        
        return result
    
    def _calculate_score(self, total_return, sharpe, max_profit, min_profit):
        """计算综合评分 (0-100)"""
        return_score = min(50, max(0, total_return * 1000))
        sharpe_score = min(30, max(0, sharpe * 10))
        
        if max_profit > 0:
            max_drawdown = (max_profit - min_profit) / max_profit
        else:
            max_drawdown = 0
        drawdown_score = max(0, 20 * (1 - max_drawdown))
        
        total_score = return_score + sharpe_score + drawdown_score
        return round(total_score, 2)
    
    def _get_recommendation(self, score, total_return):
        """根据得分给出建议"""
        if score >= 70 and total_return > 0.05:
            return "强烈推荐"
        elif score >= 50 and total_return > 0:
            return "推荐"
        elif score >= 30:
            return "中性"
        else:
            return "不推荐"
    
    def predict_batch(self, stock_data_dict, verbose=True):
        """批量预测多只股票"""
        results = []
        total = len(stock_data_dict)
        
        for i, (stock_code, df) in enumerate(stock_data_dict.items(), 1):
            try:
                df_processed = self.preprocess_data(df.copy(), fit=False)
                prediction = self.predict_single_stock(df_processed)
                prediction['stock_code'] = stock_code
                results.append(prediction)
                
                if verbose:
                    print(f"[{i}/{total}] ✓ {stock_code}: 得分={prediction['score']:.2f}, "
                          f"收益={prediction['total_return']:.2f}%, {prediction['recommendation']}")
            
            except Exception as e:
                if verbose:
                    print(f"[{i}/{total}] ✗ {stock_code}: {str(e)}")
                continue
        
        if len(results) == 0:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
        
        return df_results
    
    def get_top_stocks(self, stock_data_dict, top_n=10, min_score=30):
        """获取Top N只股票"""
        results = self.predict_batch(stock_data_dict, verbose=True)
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results = results[results['score'] >= min_score]
        return results.head(top_n)


# 便捷函数
def predict_stock(stock_code, df):
    """预测单只股票的便捷函数"""
    engine = ModelInference()
    df_processed = engine.preprocess_data(df.copy())
    result = engine.predict_single_stock(df_processed, return_details=True)
    result['stock_code'] = stock_code
    return result


if __name__ == "__main__":
    print("="*60)
    print("智能模型推理引擎测试")
    print("="*60)
    
    test_file = 'stockdata/test/sh.600036.csv'
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        df = df.sort_values('date').reset_index(drop=True)
        df = df.tail(60)
        
        try:
            result = predict_stock('sh.600036', df)
            
            print(f"\n✓ 预测成功!")
            print(f"股票代码: {result['stock_code']}")
            print(f"综合得分: {result['score']:.2f}")
            print(f"收益率: {result['total_return']:.2f}%")
            print(f"推荐等级: {result['recommendation']}")
        
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ 测试文件不存在: {test_file}")
    
    print("\n" + "="*60)
