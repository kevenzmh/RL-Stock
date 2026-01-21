# RL-Stock 增强版 (Enhanced v2.0)

<div align="center">

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.6+-yellow)
![License](https://img.shields.io/badge/license-MIT-orange)

**基于强化学习的智能股票交易系统 - 增强版**

[快速开始](#-快速开始) • [核心改进](#-核心改进) • [文档](#-文档) • [性能](#-性能指标)

</div>

---

## 🎯 项目简介

RL-Stock增强版是一个基于深度强化学习(PPO算法)的智能股票交易系统。本版本进行了全面升级,包含**风险调整奖励**、**20+技术指标**、**多维度评估**和**严格数据处理**。

### ✨ 版本亮点

- ✅ **风险调整奖励**: 集成夏普比率、波动惩罚、回撤控制
- ✅ **技术指标丰富**: MA、MACD、RSI、KDJ、布林带等20+指标
- ✅ **评估体系完善**: 滚动窗口、Monte Carlo、市场环境测试
- ✅ **数据处理严格**: 质量检查、异常处理、缺失值处理
- ✅ **代码质量高**: 模块化设计、完整文档、生产就绪

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
# 克隆或进入项目目录
cd RL-Stock

# 安装依赖
pip install -r requirements_enhanced.txt
```

### 2️⃣ 验证安装

```bash
# 运行测试,验证所有功能
python test_enhancements.py
```

预期输出:
```
✅ 技术指标模块测试通过!
✅ 数据预处理模块测试通过!
✅ 增强版环境测试通过!
✅ 评估器测试通过!
```

### 3️⃣ 开始训练

```bash
# 运行完整流水线
python main_enhanced.py
```

训练将自动完成:
- 📊 数据加载和预处理
- 🎯 添加技术指标
- 🤖 PPO模型训练
- 📈 多维度评估
- 🖼️ 生成可视化图表

---

## 📋 核心改进

### 1. 改进奖励函数 ⭐⭐⭐⭐⭐

**问题**: 原始奖励函数只考虑净值变化,忽视风险

**解决方案**:
```python
# 风险调整后的收益 (夏普比率)
sharpe_ratio = (mean_return - risk_free_rate) / std_return
reward += sharpe_ratio * 0.01

# 波动率惩罚
volatility_penalty = std_return * 0.1
reward -= volatility_penalty

# 回撤惩罚
if drawdown > 0.1:
    reward -= drawdown * 0.5
```

**效果**: 模型学习在控制风险的同时追求收益

---

### 2. 增加技术指标特征 ⭐⭐⭐⭐⭐

**问题**: 状态空间只有19维基础特征,缺少技术分析信息

**解决方案**: 新增20+个专业技术指标

| 类别 | 指标 | 数量 |
|------|------|------|
| 移动平均 | MA5/10/20/60 | 4 |
| 动量指标 | MACD, RSI, KDJ | 6 |
| 波动指标 | 布林带, ATR | 5 |
| 成交量 | 量比, OBV, 均量 | 4 |

**对比**:
```
旧版本: 19维 → 增强版: 32维 (+68%)
```

---

### 3. 增强评估方法 ⭐⭐⭐⭐⭐

**问题**: 只有简单的单次测试,无法全面评估策略

**解决方案**: 实现4种评估方法

#### 📊 基础测试
完整的性能指标计算
- 收益率、夏普比率、最大回撤
- 卡玛比率、索提诺比率
- 胜率、盈亏比

#### 🎲 Monte Carlo模拟
通过多次随机起始点测试稳健性
```python
evaluator.monte_carlo_simulation(
    model, env_class, df_test,
    n_simulations=50
)
```

#### 📈 滚动窗口测试
模拟真实交易中的持续学习
```python
evaluator.walk_forward_test(
    model, env_class, df_full,
    train_window=252,  # 1年训练
    test_window=63,    # 3个月测试
    step_size=21       # 每月滚动
)
```

#### 🌦️ 市场环境测试
在牛市、熊市、震荡市分别测试
```python
evaluator.test_different_market_conditions(
    model, env_class, df_full
)
```

---

### 4. 更新数据处理 ⭐⭐⭐⭐⭐

**问题**: 数据质量问题导致训练不稳定

**解决方案**: 完善的数据处理流程

#### ✅ 数据质量检查
- 缺失值比例检查
- 无穷值检测
- 异常值识别
- 重复行检测

#### ✅ 异常值处理 (3种策略)
```python
# 1. Clip (裁剪) - 推荐
preprocessor.handle_outliers(df, method='clip')

# 2. Winsorize (温莎化)
preprocessor.handle_outliers(df, method='winsorize')

# 3. Remove (删除)
preprocessor.handle_outliers(df, method='remove')
```

#### ✅ 缺失值处理 (4种策略)
```python
# 1. 线性插值 - 推荐
preprocessor.handle_missing_values(df, strategy='interpolate')

# 2. 前向填充
preprocessor.handle_missing_values(df, strategy='forward_fill')

# 3. 后向填充
preprocessor.handle_missing_values(df, strategy='backward_fill')

# 4. 删除
preprocessor.handle_missing_values(df, strategy='drop')
```

#### ✅ 标准化验证 (3种方法)
```python
# Robust - 推荐(对异常值鲁棒)
preprocessor = DataPreprocessor(method='robust')
```

---

## 📊 性能指标

系统提供10+个专业性能指标:

| 指标 | 说明 |
|------|------|
| 总收益率 | 总体投资回报 |
| 年化收益率 | 年化投资回报 |
| **夏普比率** | 风险调整后收益 (>1良好, >2优秀) |
| 最大回撤 | 最大损失百分比 |
| **卡玛比率** | 回撤调整收益 |
| 索提诺比率 | 下行风险调整收益 |
| 波动率 | 年化收益波动 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利/平均亏损 |
| 交易次数 | 总交易统计 |

---

## 📁 项目结构

```
RL-Stock/
├── rlenv/
│   ├── StockTradingEnv_enhanced.py    # ⭐ 增强版环境
│   ├── StockTradingEnv_improved.py    # 改进版环境
│   └── StockTradingEnv0.py           # 原始环境
├── utils/
│   ├── technical_indicators.py        # ⭐ 技术指标 (20+)
│   ├── enhanced_evaluator.py          # ⭐ 增强评估器
│   ├── data_preprocessing.py          # ⭐ 数据预处理
│   └── performance_metrics.py         # 性能指标
├── main_enhanced.py                   # ⭐ 增强版主程序
├── test_enhancements.py               # ⭐ 功能测试
├── generate_comparison_charts.py      # 对比图生成
├── stockdata/                         # 股票数据
├── models/                            # 模型保存
├── log_enhanced/                      # 训练日志
└── img/                               # 可视化图表
```

**⭐ = 本次新增或重大更新**

---

## 📖 文档

| 文档 | 说明 |
|------|------|
| **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** | 改进总结 |
| **[ENHANCEMENTS.md](ENHANCEMENTS.md)** | 详细技术文档 |
| **[USAGE_GUIDE.md](USAGE_GUIDE.md)** | 使用指南 |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | 快速参考 |

---

## 🎯 使用示例

### 基础使用

```python
from main_enhanced import run_complete_pipeline

# 训练默认股票 (sh.000001)
run_complete_pipeline('sh.000001', train_steps=100000)
```

### 自定义配置

```python
# 训练不同股票
run_complete_pipeline('sz.300677', train_steps=100000)

# 调整训练步数
run_complete_pipeline('sh.000001', train_steps=50000)   # 快速
run_complete_pipeline('sh.000001', train_steps=200000)  # 高质量
```

### 单独使用模块

```python
# 1. 技术指标
from utils.technical_indicators import add_all_technical_indicators
df = add_all_technical_indicators(df)

# 2. 数据预处理
from utils.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor(method='robust')
df = preprocessor.preprocess_pipeline(df, fit=True)

# 3. 评估
from utils.enhanced_evaluator import EnhancedEvaluator
evaluator = EnhancedEvaluator()
results = evaluator.monte_carlo_simulation(model, env, df, n_simulations=50)
```

---

## 🖼️ 可视化结果

训练完成后会生成:

1. **综合评估图** - `img/{stock_code}_comprehensive.png`
   - 净值曲线
   - Monte Carlo收益率分布
   - 夏普比率分布
   - 最大回撤分布

2. **滚动窗口图** - `img/{stock_code}_walk_forward.png`
   - 各窗口收益率
   - 各窗口夏普比率
   - 各窗口最大回撤

3. **改进对比图** - `img/improvements_overview.png`
   - 功能模块对比
   - 状态空间对比
   - 性能指标对比

---

## ⚙️ 配置参数

### PPO超参数

```python
learning_rate=0.0003      # 学习率
n_steps=2048              # 每次更新步数
nminibatches=32           # Minibatch数量
noptepochs=10             # 优化轮数
gamma=0.99                # 折扣因子
lam=0.95                  # GAE lambda
cliprange=0.2             # PPO裁剪
ent_coef=0.01             # 熵系数
```

### 环境参数

```python
INITIAL_ACCOUNT_BALANCE = 10000    # 初始资金
TRANSACTION_FEE_PERCENT = 0.001    # 交易费用 (0.1%)
risk_free_rate = 0.03              # 无风险利率 (年化)
```

---

## 🐛 故障排除

### 缺少依赖

```bash
# sklearn
pip install scikit-learn

# 其他依赖
pip install -r requirements_enhanced.txt
```

### TensorFlow GPU

如果没有GPU:
```bash
pip uninstall tensorflow-gpu
pip install tensorflow==1.15.0
```

### 内存不足

减少评估次数:
```python
# Monte Carlo
n_simulations=20  # 原来50

# 滚动窗口
step_size=63  # 增大步长
```

---

## 📈 性能对比

| 方面 | 旧版本 | 增强版 | 提升 |
|------|--------|--------|------|
| 状态维度 | 19 | 32 | +68% |
| 技术指标 | 0 | 20+ | ∞ |
| 评估方法 | 1 | 4 | +300% |
| 性能指标 | 5 | 10+ | +100% |
| 奖励函数 | 简单 | 风险调整 | ✅ |
| 数据处理 | 基础 | 完善 | ✅ |

---

## 🔄 更新日志

### v2.0 Enhanced (2025-01-21)
- ✅ 实现风险调整奖励函数
- ✅ 新增20+技术指标
- ✅ 实现滚动窗口测试
- ✅ 实现Monte Carlo模拟
- ✅ 实现市场环境测试
- ✅ 完善数据处理流程
- ✅ 提供完整文档

---

## 📚 参考资料

- [Stable Baselines文档](https://stable-baselines.readthedocs.io/)
- [强化学习在金融中的应用](https://arxiv.org/abs/1811.09540)
- [技术分析指标大全](https://www.investopedia.com/technical-analysis-4689657)

---

## 📝 许可证

MIT License

---

## 🙏 致谢

感谢所有开源项目的贡献者！

---

<div align="center">

**RL-Stock Enhanced v2.0**

让AI学会像专业交易员一样思考

⭐ Star this repo if you find it useful! ⭐

</div>
