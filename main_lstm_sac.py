"""
LSTM + SAC实现 - 捕捉时间序列依赖
结合LSTM的时序建模能力和SAC的高效学习

使用方法:
    python main_lstm_sac.py --stock sh.000001 --steps 200000

特点:
1. LSTM特征提取器捕捉时间依赖
2. 多层LSTM处理复杂模式
3. 注意力机制(可选)
4. 与SAC算法结合
"""
import os
import sys
import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
from gym import spaces

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced, INITIAL_ACCOUNT_BALANCE
from utils.technical_indicators import add_all_technical_indicators
from utils.data_preprocessing import DataPreprocessor


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM特征提取器
    用于处理时间序列数据
    """
    def __init__(self, observation_space: gym.spaces.Box, 
                 features_dim: int = 256,
                 lstm_hidden: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        前向传播
        
        Args:
            observations: (batch_size, features)
        
        Returns:
            features: (batch_size, features_dim)
        """
        # LSTM需要 (batch, seq_len, features)
        # 这里seq_len=1,因为我们处理单个时间步
        observations = observations.unsqueeze(1)
        
        # LSTM前向
        lstm_out, (h_n, c_n) = self.lstm(observations)
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        
        # 全连接层
        features = self.linear(lstm_out)
        
        return features


class AttentionLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    带注意力机制的LSTM特征提取器
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 lstm_hidden: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(lstm_hidden)
        self.norm2 = nn.LayerNorm(lstm_hidden)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # LSTM处理
        observations = observations.unsqueeze(1)
        lstm_out, _ = self.lstm(observations)
        
        # 自注意力
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 残差连接 + Layer Norm
        lstm_out = self.norm1(lstm_out + attn_out)
        
        # 前馈网络
        lstm_out = lstm_out.squeeze(1)
        features = self.ffn(lstm_out)
        features = self.norm2(features)
        
        return features


class StockTradingEnvWithHistory(StockTradingEnvEnhanced):
    """
    支持历史窗口的环境
    用于LSTM模型
    """
    def __init__(self, df, window_size=10, **kwargs):
        super().__init__(df, **kwargs)
        self.window_size = window_size
        
        # 修改观察空间为 (window_size, features)
        single_obs_shape = super().observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(window_size, single_obs_shape),
            dtype=np.float16
        )
        
    def _next_observation(self):
        """返回历史窗口的观察"""
        obs_list = []
        
        # 确保有足够的历史
        start_idx = max(0, self.current_step - self.window_size + 1)
        
        for i in range(start_idx, self.current_step + 1):
            # 获取单个时间步的观察
            single_obs = self._get_single_observation(i)
            obs_list.append(single_obs)
        
        # 填充不足的部分(用第一个观察重复填充)
        while len(obs_list) < self.window_size:
            obs_list.insert(0, obs_list[0] if obs_list else np.zeros(32))
        
        return np.array(obs_list, dtype=np.float16)
    
    def _get_single_observation(self, step):
        """
        获取单个时间步的观察
        复用父类的逻辑
        """
        # 临时保存当前步
        original_step = self.current_step
        self.current_step = step
        
        # 调用父类方法
        obs = super()._next_observation()
        
        # 恢复当前步
        self.current_step = original_step
        
        return obs


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


def train_lstm_sac(df_train, train_steps=200000, 
                   use_attention=False, window_size=10):
    """
    训练LSTM+SAC模型
    
    Args:
        df_train: 训练数据
        train_steps: 训练步数
        use_attention: 是否使用注意力机制
        window_size: 历史窗口大小
    """
    print("\n" + "="*60)
    print("LSTM + SAC 模型训练")
    print("="*60)
    print(f"训练步数: {train_steps:,}")
    print(f"使用注意力: {use_attention}")
    print(f"窗口大小: {window_size}")
    
    # 创建环境(使用历史窗口)
    env = DummyVecEnv([
        lambda: StockTradingEnvWithHistory(df_train, window_size=window_size)
    ])
    
    # 日志和模型目录
    log_dir = './log_lstm_sac'
    model_dir = './models'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Checkpoint回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix='lstm_sac_checkpoint'
    )
    
    # 选择特征提取器
    if use_attention:
        print("\n使用注意力机制...")
        feature_extractor = AttentionLSTMFeatureExtractor
        extractor_kwargs = dict(
            features_dim=256,
            lstm_hidden=128,
            num_layers=2,
            num_heads=4,
            dropout=0.2
        )
    else:
        print("\n使用标准LSTM...")
        feature_extractor = LSTMFeatureExtractor
        extractor_kwargs = dict(
            features_dim=256,
            lstm_hidden=128,
            num_layers=2,
            dropout=0.2
        )
    
    # 策略网络配置
    policy_kwargs = dict(
        features_extractor_class=feature_extractor,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )
    
    # 创建SAC模型
    print("\n创建LSTM+SAC模型...")
    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=500_000,        # LSTM需要更多内存,减小buffer
        learning_starts=10000,
        batch_size=128,             # 减小batch size以适应LSTM
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        tensorboard_log=log_dir,
        verbose=1,
        device='auto'
    )
    
    print(f"\n模型配置:")
    print(f"  观察空间: {env.observation_space.shape}")
    print(f"  特征提取器: {feature_extractor.__name__}")
    print(f"  LSTM隐藏层: 128")
    print(f"  网络深度: 2层")
    
    # 训练
    print("\n开始训练...")
    print("="*60)
    model.learn(
        total_timesteps=train_steps,
        callback=checkpoint_callback,
        log_interval=10,
        progress_bar=True
    )
    
    # 保存模型
    model_name = 'lstm_attn_sac' if use_attention else 'lstm_sac'
    model_path = os.path.join(model_dir, f'{model_name}_final_{train_steps}.zip')
    model.save(model_path)
    print(f"\n模型已保存: {model_path}")
    
    return model


def evaluate_lstm_model(model, df_test, stock_code, window_size=10):
    """评估LSTM模型"""
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    # 创建测试环境
    env = StockTradingEnvWithHistory(df_test, window_size=window_size)
    obs = env.reset()
    
    # 运行测试
    portfolio_values = [INITIAL_ACCOUNT_BALANCE]
    actions_log = []
    
    done = False
    step = 0
    print("\n测试中...")
    
    while not done and step < len(df_test) - window_size:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        portfolio_values.append(info['net_worth'])
        actions_log.append(action)
        step += 1
        
        if step % 20 == 0:
            print(f"  Step {step}/{len(df_test)}: Net Worth = ¥{info['net_worth']:,.2f}")
    
    # 计算指标
    from utils.enhanced_evaluator import EnhancedEvaluator
    evaluator = EnhancedEvaluator()
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
    
    # 绘图
    import matplotlib.font_manager as fm
    font = fm.FontProperties(fname='font/wqy-microhei.ttc')
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, linewidth=2, color='#6A4C93', label='LSTM+SAC策略')
    plt.axhline(y=INITIAL_ACCOUNT_BALANCE, color='r', linestyle='--', 
                alpha=0.5, label='初始资金')
    plt.grid(True, alpha=0.3)
    plt.xlabel('交易日', fontproperties=font, fontsize=12)
    plt.ylabel('资产净值 (元)', fontproperties=font, fontsize=12)
    plt.title(f'{stock_code} LSTM+SAC策略回测', 
              fontproperties=font, fontsize=14, fontweight='bold')
    plt.legend(prop=font, fontsize=10)
    plt.tight_layout()
    
    img_path = f'./img/{stock_code}_lstm_sac.png'
    os.makedirs('./img', exist_ok=True)
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {img_path}")
    plt.close()
    
    return metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM+SAC训练和评估')
    parser.add_argument('--stock', type=str, default='sh.000001', help='股票代码')
    parser.add_argument('--steps', type=int, default=200000, help='训练步数')
    parser.add_argument('--window', type=int, default=10, help='历史窗口大小')
    parser.add_argument('--attention', action='store_true', help='使用注意力机制')
    
    args = parser.parse_args()
    
    # 数据准备
    train_file = f'./stockdata/train/{args.stock}.csv'
    test_file = f'./stockdata/test/{args.stock}.csv'
    
    if not os.path.exists(train_file):
        # 尝试查找文件
        for root, dirs, files in os.walk('./stockdata/train'):
            for f in files:
                if args.stock in f:
                    train_file = os.path.join(root, f)
                    break
    
    if not os.path.exists(test_file):
        for root, dirs, files in os.walk('./stockdata/test'):
            for f in files:
                if args.stock in f:
                    test_file = os.path.join(root, f)
                    break
    
    df_train = prepare_data(train_file)
    df_test = prepare_data(test_file)
    
    # 训练
    model = train_lstm_sac(
        df_train, 
        train_steps=args.steps,
        use_attention=args.attention,
        window_size=args.window
    )
    
    # 评估
    evaluate_lstm_model(model, df_test, args.stock, window_size=args.window)
    
    print("\n完成!")
