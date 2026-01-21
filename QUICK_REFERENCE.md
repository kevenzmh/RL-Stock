# RL-Stock 增强版 - 快速参考

## 🚀 快速开始 (3步)

```bash
# 1. 安装依赖
pip install -r requirements_enhanced.txt

# 2. 测试验证
python test_enhancements.py

# 3. 开始训练
python main_enhanced.py
```

---

## 📋 核心改进 (4项)

### ✅ 1. 改进奖励函数
- 夏普比率 (风险调整收益)
- 波动率惩罚
- 回撤惩罚
- 交易成本 (0.1%)

### ✅ 2. 技术指标 (20+)
- MA (5/10/20/60)
- MACD
- RSI
- KDJ
- 布林带
- 成交量指标
- ATR

### ✅ 3. 评估方法 (4种)
- 基础测试
- Monte Carlo模拟
- 滚动窗口测试
- 市场环境测试

### ✅ 4. 数据处理
- 质量检查
- 异常值处理 (3种策略)
- 缺失值处理 (4种策略)
- 标准化验证 (3种方法)

---

## 📁 关键文件

| 文件 | 用途 |
|------|------|
| `main_enhanced.py` | 主程序 - 完整流水线 |
| `rlenv/StockTradingEnv_enhanced.py` | 增强环境 |
| `utils/enhanced_evaluator.py` | 评估器 |
| `utils/technical_indicators.py` | 技术指标 |
| `utils/data_preprocessing.py` | 数据处理 |
| `test_enhancements.py` | 功能测试 |

---

## 🎯 性能指标 (10个)

1. 总收益率
2. 年化收益率
3. 夏普比率
4. 最大回撤
5. 卡玛比率
6. 索提诺比率
7. 波动率
8. 胜率
9. 盈亏比
10. 交易次数

---

## ⚙️ 自定义配置

### 修改股票代码
```python
run_complete_pipeline('sz.300677', train_steps=100000)
```

### 调整训练步数
```python
train_steps=50000   # 快速 (10分钟)
train_steps=100000  # 标准 (30分钟)
train_steps=200000  # 高质量 (1小时)
```

### 修改评估参数
```python
# Monte Carlo次数
n_simulations=50   # 默认
n_simulations=100  # 更可靠

# 滚动窗口
train_window=252  # 1年
test_window=63    # 3个月
step_size=21      # 每月
```

---

## 📊 输出内容

### 模型文件
- `models/ppo2_enhanced_100000.pkl`

### 图表
- `img/{stock_code}_comprehensive.png` - 综合评估
- `img/{stock_code}_walk_forward.png` - 滚动窗口

### 日志
- `log_enhanced/` - TensorBoard日志

---

## 🔧 常见问题

**Q: 缺少sklearn?**
```bash
pip install scikit-learn
```

**Q: 训练太慢?**
- 减少训练步数
- 跳过部分评估
- 使用GPU

**Q: 内存不足?**
- 减少Monte Carlo次数
- 增大滚动步长
- 减少训练步数

---

## 📈 状态空间对比

| 版本 | 维度 | 内容 |
|------|------|------|
| 旧版 | 19维 | 基础价格+账户 |
| 新版 | 32维 | +20个技术指标 |

---

## 🎓 评估方法对比

| 方法 | 用途 | 输出 |
|------|------|------|
| 基础测试 | 整体表现 | 所有指标 |
| Monte Carlo | 稳健性 | 分布统计 |
| 滚动窗口 | 适应性 | 时间序列 |
| 市场环境 | 场景表现 | 分类结果 |

---

## 💡 最佳实践

1. ✅ 先运行测试验证
2. ✅ 使用robust标准化
3. ✅ 选择interpolate处理缺失值
4. ✅ 使用clip处理异常值
5. ✅ 保存TensorBoard日志
6. ✅ 生成完整评估报告

---

## 📚 文档索引

- `IMPROVEMENTS_SUMMARY.md` - 改进总结
- `ENHANCEMENTS.md` - 详细技术文档
- `USAGE_GUIDE.md` - 使用指南
- `QUICK_REFERENCE.md` - 本文件

---

**版本**: Enhanced v2.0  
**更新**: 2025-01-21  
**状态**: ✅ 生产就绪
