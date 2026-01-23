# 📁 项目文件索引

## 🆕 本次新增文件（简化版选股系统）

### ⭐ 核心代码文件

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `realtime_data.py` | 实时数据获取模块 | 🔴 必读 |
| `model_inference.py` | 模型推理引擎 | 🔴 必读 |
| `simple_selector.py` | 智能选股主程序 | 🔴 必读 |
| `test_selector.py` | 系统测试脚本 | 🟡 建议 |

### 📚 文档文件

| 文件 | 内容 | 何时阅读 |
|------|------|---------|
| `QUICKSTART_SELECTOR.md` | 3步快速开始 | 🔴 立即阅读 |
| `SELECTOR_GUIDE.md` | 完整使用指南（5000+字） | 🟡 详细使用时 |
| `COMPLETION_REPORT.md` | 交付完成报告 | 🟢 了解全貌 |
| `DELIVERY_CHECKLIST.md` | 交付清单 | 🟢 验收时查看 |
| `ARCHITECTURE.md` | 系统架构图 | 🟢 技术深入时 |
| `ONLINE_READINESS_REPORT.md` | 线上准备度评估（8000+字） | 🟢 规划升级时 |

### 🔧 工具文件

| 文件 | 用途 |
|------|------|
| `run_test.bat` | Windows一键测试 |
| `run_selector.bat` | Windows一键选股 |

---

## 📂 原有重要文件

### 训练相关

| 文件 | 功能 |
|------|------|
| `main_enhanced.py` | 增强版训练脚本 |
| `main_sac.py` | SAC算法训练 |
| `main_lstm_sac.py` | LSTM+SAC训练 |

### 环境定义

| 文件 | 功能 |
|------|------|
| `rlenv/StockTradingEnv_enhanced.py` | 增强版交易环境 |
| `rlenv/StockTradingEnv_Fixed.py` | 修复版环境 |

### 工具模块

| 文件 | 功能 |
|------|------|
| `utils/technical_indicators.py` | 20+技术指标 |
| `utils/data_preprocessing.py` | 数据预处理 |
| `utils/enhanced_evaluator.py` | 增强评估器 |

### 数据获取

| 文件 | 功能 |
|------|------|
| `get_stock_data.py` | 历史数据下载 |
| `stockdata/train/` | 训练数据集（5668只） |
| `stockdata/test/` | 测试数据集（5668只） |

### 模型文件

| 文件 | 说明 |
|------|------|
| `models/ppo2_enhanced_100000.pkl` | 增强版模型（推荐） |
| `models/ppo2_stock_100000.pkl` | 标准模型 |

### 文档

| 文件 | 内容 |
|------|------|
| `FEASIBILITY_REPORT.md` | 可行性分析报告 |
| `IMPROVEMENTS_SUMMARY.md` | 改进总结 |
| `ENHANCEMENTS.md` | 增强功能文档 |
| `UPGRADE_ROADMAP.md` | 16周升级路线图 |
| `USAGE_GUIDE.md` | 训练使用指南 |

---

## 🎯 快速导航

### 想立即使用选股？
👉 **先读**: `QUICKSTART_SELECTOR.md`  
👉 **运行**: `python test_selector.py`  
👉 **然后**: `python simple_selector.py`

### 想详细了解用法？
👉 **阅读**: `SELECTOR_GUIDE.md`（完整教程）

### 想了解实现原理？
👉 **阅读**: `ARCHITECTURE.md`（系统架构）

### 想训练自己的模型？
👉 **阅读**: `USAGE_GUIDE.md`  
👉 **运行**: `python main_enhanced.py`

### 想升级系统？
👉 **阅读**: `UPGRADE_ROADMAP.md`（16周计划）

### 遇到问题？
👉 **查看**: `SELECTOR_GUIDE.md` 的"故障排查"章节

---

## 📊 文件统计

### 代码文件
- Python脚本: 20+个
- 核心代码: ~10,000行
- 新增代码: ~1,030行

### 文档文件
- Markdown文档: 15+个
- 总字数: 50,000+字
- 新增文档: 18,500+字

### 数据文件
- 训练数据: 5,668只股票
- 测试数据: 5,668只股票
- 模型文件: 3个

---

## 🔄 文件关系图

```
项目根目录
│
├─ 选股系统 (NEW! 本次交付)
│  ├─ simple_selector.py      ← 主程序
│  ├─ realtime_data.py        ← 数据获取
│  ├─ model_inference.py      ← 模型推理
│  └─ test_selector.py        ← 测试脚本
│
├─ 训练系统 (已有)
│  ├─ main_enhanced.py        ← 增强训练
│  ├─ main_sac.py             ← SAC训练
│  └─ main_lstm_sac.py        ← LSTM训练
│
├─ 核心模块 (已有)
│  ├─ rlenv/                  ← 交易环境
│  └─ utils/                  ← 工具库
│
├─ 数据和模型 (已有)
│  ├─ stockdata/              ← 股票数据
│  └─ models/                 ← 训练模型
│
└─ 文档 (全面)
   ├─ QUICKSTART_SELECTOR.md  ← 快速开始
   ├─ SELECTOR_GUIDE.md       ← 使用指南
   ├─ COMPLETION_REPORT.md    ← 完成报告
   └─ 等等...
```

---

## 💡 推荐阅读顺序

### 新用户（想立即使用选股）
1. ⭐ `QUICKSTART_SELECTOR.md` - 3分钟
2. ⭐ 运行 `test_selector.py` - 5分钟
3. ⭐ 运行 `simple_selector.py` - 10分钟
4. 📖 `SELECTOR_GUIDE.md` - 需要时查阅

### 技术用户（想了解实现）
1. ⭐ `COMPLETION_REPORT.md` - 了解全貌
2. ⭐ `ARCHITECTURE.md` - 理解架构
3. 📖 阅读源码 `simple_selector.py`
4. 📖 `ONLINE_READINESS_REPORT.md` - 深入评估

### 进阶用户（想升级优化）
1. ⭐ `ONLINE_READINESS_REPORT.md` - 差距分析
2. ⭐ `UPGRADE_ROADMAP.md` - 升级计划
3. 📖 `IMPROVEMENTS_SUMMARY.md` - 已完成改进
4. 📖 `ENHANCEMENTS.md` - 技术细节

---

## 🔍 按需查找

### 需要快速开始
→ `QUICKSTART_SELECTOR.md`

### 需要详细教程
→ `SELECTOR_GUIDE.md`

### 遇到错误
→ `SELECTOR_GUIDE.md` > "故障排查"

### 想修改参数
→ `SELECTOR_GUIDE.md` > "参数说明"

### 想了解原理
→ `ARCHITECTURE.md`

### 想训练模型
→ `USAGE_GUIDE.md`

### 想看改进记录
→ `IMPROVEMENTS_SUMMARY.md`

### 想升级系统
→ `UPGRADE_ROADMAP.md`

### 想评估项目
→ `ONLINE_READINESS_REPORT.md`

---

## 📝 版本历史

### v1.0 - 简化版选股系统 (2026-01-22) 🆕
- ✅ 实时数据获取
- ✅ 模型推理引擎
- ✅ 智能选股系统
- ✅ 完整测试验证
- ✅ 详细文档支持

### v0.9 - 增强版改进 (2026-01-21)
- ✅ 改进奖励函数
- ✅ 20+技术指标
- ✅ 增强评估方法
- ✅ 优化数据处理

### v0.8 - 算法升级 (2026-01-21)
- ✅ SAC算法
- ✅ LSTM模型
- ✅ 注意力机制

### v0.7 - 代码修复 (2026-01-20)
- ✅ Bug修复
- ✅ 交易成本
- ✅ 风险控制

---

**索引最后更新**: 2026-01-22  
**总文件数**: 50+  
**总代码行**: 10,000+  
**总文档字数**: 50,000+
