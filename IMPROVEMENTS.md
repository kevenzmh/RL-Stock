# 强化学习股票交易策略改进方案

## 📊 改进内容总结

### 1. 奖励函数优化 (最关键!)

**原问题**: 
- 原始代码中,如果利润<=0就给-100的惩罚
- 导致模型学习到"不交易"是最安全的策略

**改进方案**:
```python
# 1. 基于净值变化的连续奖励
net_worth_change = self.net_worth - prev_net_worth
reward = net_worth_change / INITIAL_ACCOUNT_BALANCE

# 2. 鼓励探索,避免一直不交易
if self.shares_held == 0 and self.total_shares_sold == 0:
    reward -= 0.001  # 小惩罚

# 3. 风险控制
if self.net_worth < INITIAL_ACCOUNT_BALANCE * 0.5:
    reward -= 1.0  # 亏损50%以上惩罚
    done = True
```

### 2. 训练步数增加

- **原始**: 10,000 步
- **改进**: 100,000 步 (10倍)
- **建议**: 可尝试 200,000 - 500,000 步

### 3. PPO2 超参数优化

```python
model = PPO2(
    MlpPolicy, env,
    learning_rate=0.0003,    # 学习率
    n_steps=2048,            # 每次更新步数
    batch_size=64,           # 批次大小  
    n_epochs=10,             # 训练轮数
    gamma=0.99,              # 折扣因子
    ent_coef=0.01,           # 熵系数,鼓励探索
    clip_range=0.2,          # PPO裁剪
)
```

### 4. 训练起始位置随机化

```python
# 增加训练多样性
self.current_step = random.randint(0, len(self.df) - 100)
```

### 5. 更好的监控和调试

- 添加详细的训练日志
- 保存训练好的模型
- 改进的可视化输出

## 🚀 使用方法

### 方式1: 使用默认参数(10万步)
```bash
cd D:\PycharmProjects\RL-Stock
conda activate rl-stock
python main_improved.py
```

### 方式2: 自定义训练步数
在 `main_improved.py` 中修改:
```python
# 20万步训练,效果更好但需要更长时间
test_improved_strategy('sh.000001', train_steps=200000)
```

### 方式3: 测试其他股票
```python
# 测试招商银行
test_improved_strategy('sh.600036', train_steps=100000)
```

## 📈 预期效果

| 训练步数 | 预期时间 | 预期效果 |
|---------|---------|---------|
| 10,000  | 1-2分钟 | 可能仍无交易 |
| 100,000 | 10-15分钟 | 开始有交易行为 |
| 200,000 | 20-30分钟 | 较好的策略 |
| 500,000 | 1小时+ | 最优策略 |

## 🔧 进一步优化建议

### 1. 特征工程
添加技术指标:
- MA5, MA10, MA20 (移动平均线)
- RSI (相对强弱指标)
- MACD (指数平滑移动平均线)
- 布林带

### 2. 多股票组合训练
同时训练多只股票,提高泛化能力

### 3. 使用A2C或SAC算法
尝试其他强化学习算法

### 4. 回测框架
使用 backtrader 进行更专业的回测

### 5. 风险管理
- 设置止损止盈
- 仓位管理
- 最大回撤控制

## 📝 文件说明

- `rlenv/StockTradingEnv_improved.py` - 改进的交易环境
- `main_improved.py` - 改进的训练脚本  
- `models/` - 保存训练好的模型
- `log_improved/` - TensorBoard日志

## 🎯 关键改进点

1. ✅ 奖励函数从二值改为连续
2. ✅ 增加探索鼓励机制
3. ✅ 训练步数增加10倍
4. ✅ 优化PPO2超参数
5. ✅ 添加模型保存和加载
6. ✅ 改进日志和可视化

现在运行改进版本,应该能看到AI agent开始主动交易了!
