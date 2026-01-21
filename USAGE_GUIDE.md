# RL-Stock 增强版使用指南

## 📦 安装依赖

### 方法1: 使用增强版requirements (推荐)

```bash
pip install -r requirements_enhanced.txt
```

### 方法2: 手动安装缺失的包

如果已经安装了基础依赖,只需添加:

```bash
pip install scikit-learn
```

---

## 🚀 快速开始

### 1. 验证安装

运行测试脚本确保所有功能正常:

```bash
python test_enhancements.py
```

**预期输出**:
```
==============================================================
RL-Stock 增强版功能测试
==============================================================

测试 1: 技术指标模块
...
✅ 技术指标模块测试通过!

测试 2: 数据预处理模块
...
✅ 数据预处理模块测试通过!

测试 3: 增强版交易环境
...
✅ 增强版环境测试通过!

测试 4: 增强版评估器
...
✅ 评估器测试通过!

所有测试完成!
✅ 所有核心功能正常工作
```

### 2. 运行完整训练和评估

```bash
python main_enhanced.py
```

这将执行:
- 数据加载和预处理
- 添加技术指标
- 模型训练 (100,000步)
- 基础测试
- Monte Carlo模拟 (50次)
- 滚动窗口测试
- 不同市场环境测试
- 生成可视化图表

**训练时间**: 约30-60分钟 (取决于硬件)

---

## 📊 输出文件

### 训练日志
- 位置: `log_enhanced/PPO2_1/`
- 查看: `tensorboard --logdir=./log_enhanced`

### 模型文件
- 位置: `models/ppo2_enhanced_100000.pkl`
- 用途: 可加载用于后续测试或部署

### 评估图表
1. **综合评估图** - `img/sh.000001_comprehensive.png`
   - 净值曲线
   - Monte Carlo收益率分布
   - 夏普比率分布
   - 最大回撤分布

2. **滚动窗口图** - `img/sh.000001_walk_forward.png`
   - 各窗口收益率
   - 各窗口夏普比率
   - 各窗口最大回撤

---

## 🎯 自定义训练

### 训练不同股票

编辑 `main_enhanced.py` 末尾:

```python
if __name__ == '__main__':
    # 修改股票代码
    run_complete_pipeline('sz.300677', train_steps=100000)
```

### 调整训练步数

```python
# 快速测试 (10分钟)
run_complete_pipeline('sh.000001', train_steps=50000)

# 标准训练 (30分钟)
run_complete_pipeline('sh.000001', train_steps=100000)

# 高质量训练 (1-2小时)
run_complete_pipeline('sh.000001', train_steps=200000)
```

### 修改评估参数

编辑 `main_enhanced.py` 中的 `comprehensive_evaluation` 函数:

```python
# Monte Carlo 模拟次数
mc_results, mc_returns, mc_sharpes, mc_drawdowns = evaluator.monte_carlo_simulation(
    model, StockTradingEnvEnhanced, df_test, 
    n_simulations=100,  # 增加到100次
    random_start=True
)

# 滚动窗口参数
wf_results = evaluator.walk_forward_test(
    model, StockTradingEnvEnhanced, df_full,
    train_window=126,  # 半年训练
    test_window=21,    # 1个月测试
    step_size=7        # 每周滚动
)
```

---

## 🔧 高级配置

### 修改奖励函数权重

编辑 `rlenv/StockTradingEnv_enhanced.py` 中的 `_calculate_reward` 方法:

```python
# 调整夏普比率权重
reward += sharpe_ratio * 0.02  # 原来是0.01

# 调整波动率惩罚
volatility_penalty = std_return * 0.2  # 原来是0.1
reward -= volatility_penalty

# 调整回撤惩罚阈值
if drawdown > 0.05:  # 原来是0.1 (10%)
    reward -= drawdown * 1.0  # 原来是0.5
```

### 调整交易成本

编辑 `rlenv/StockTradingEnv_enhanced.py`:

```python
TRANSACTION_FEE_PERCENT = 0.0005  # 0.05% (原来是0.1%)
```

### 修改PPO超参数

编辑 `main_enhanced.py` 中的 `train_enhanced_model` 函数:

```python
model = PPO2(
    MlpPolicy, 
    env, 
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0001,      # 降低学习率
    n_steps=4096,              # 增加步数
    nminibatches=64,           # 增加minibatch
    noptepochs=20,             # 增加优化轮数
    gamma=0.995,               # 调整折扣因子
    lam=0.98,                  # 调整GAE参数
    cliprange=0.1,             # 减小裁剪范围
    ent_coef=0.005,            # 减小熵系数
)
```

---

## 📈 性能优化建议

### 1. 数据准备优化

```python
# 使用更鲁棒的标准化方法
df_train = prepare_data_with_indicators(train_file, method='robust')
```

### 2. 训练加速

- 使用GPU版本的TensorFlow
- 减少Monte Carlo模拟次数 (开发阶段)
- 跳过滚动窗口测试 (开发阶段)

### 3. 内存优化

如果遇到内存问题:

```python
# 减少滚动窗口数量
wf_results = evaluator.walk_forward_test(
    model, StockTradingEnvEnhanced, df_full,
    train_window=252,
    test_window=63,
    step_size=63  # 增大步长,减少窗口数
)
```

---

## 🐛 常见问题

### Q1: 缺少sklearn模块

**解决方案**:
```bash
pip install scikit-learn
```

### Q2: TensorFlow GPU版本问题

**解决方案**:
如果没有GPU或CUDA配置问题,可以使用CPU版本:
```bash
pip uninstall tensorflow-gpu
pip install tensorflow==1.15.0
```

### Q3: 内存不足

**解决方案**:
- 减少训练步数
- 减少Monte Carlo模拟次数
- 跳过滚动窗口测试

### Q4: 训练过慢

**解决方案**:
- 使用GPU版本
- 减少n_steps参数
- 减少noptepochs参数

### Q5: 找不到股票数据文件

**解决方案**:
确保数据文件在正确的位置:
```
stockdata/
  train/
    sh.000001.csv
  test/
    sh.000001.csv
```

---

## 📊 评估指标说明

### 收益指标
- **总收益率**: (最终资产 - 初始资产) / 初始资产
- **年化收益率**: 考虑时间因素的年化回报

### 风险指标
- **波动率**: 收益率的标准差(年化)
- **最大回撤**: 从峰值到谷底的最大百分比损失

### 风险调整收益
- **夏普比率**: (年化收益 - 无风险利率) / 年化波动率
  - \> 1: 良好
  - \> 2: 优秀
  - \> 3: 卓越
- **卡玛比率**: 年化收益率 / 最大回撤
  - 越高越好
- **索提诺比率**: 只考虑下行风险的夏普比率

### 交易统计
- **胜率**: 盈利交易次数 / 总交易次数
- **盈亏比**: 平均盈利 / 平均亏损

---

## 🎓 学习资源

### 强化学习
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Stable Baselines文档](https://stable-baselines.readthedocs.io/)

### 量化交易
- [Investopedia](https://www.investopedia.com/)
- [QuantConnect](https://www.quantconnect.com/tutorials/)

### 技术分析
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [TradingView指标wiki](https://www.tradingview.com/wiki/)

---

## 📞 获取帮助

### 查看日志
```bash
# 查看训练日志
tensorboard --logdir=./log_enhanced

# 运行测试
python test_enhancements.py
```

### 调试模式

在代码中添加更多打印信息:

```python
# 在训练循环中
if step % 100 == 0:
    env.render()
```

---

## 🔄 版本历史

### v2.0 Enhanced (2025-01-21)
- ✅ 改进奖励函数 (夏普比率 + 风险惩罚)
- ✅ 增加技术指标特征 (32维状态空间)
- ✅ 增强评估方法 (滚动窗口 + Monte Carlo)
- ✅ 更新数据处理 (质量检查 + 异常值处理)

### v1.0 Improved
- 基础改进版本

### v0.1 Original
- 初始版本

---

**祝训练顺利！** 🚀
