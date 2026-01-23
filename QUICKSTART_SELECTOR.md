# 🚀 快速开始卡片

## 3步开始使用

### 1️⃣ 测试系统
```bash
python test_selector.py
```

### 2️⃣ 运行选股
```bash
python simple_selector.py
```

### 3️⃣ 查看结果
打开生成的 `stock_selection_*.csv` 文件

---

## 常用命令

```bash
# 默认 (Top 10, 100只候选池)
python simple_selector.py

# Top 20
python simple_selector.py --top 20

# 从500只中筛选
python simple_selector.py --pool 500

# 只要高分股票
python simple_selector.py --min-score 50

# 组合使用
python simple_selector.py --top 30 --pool 300 --min-score 40
```

---

## 输出说明

| 得分 | 推荐等级 | 符号 |
|------|---------|------|
| 70+ | 强烈推荐 | 🔥 |
| 50-70 | 推荐 | ⭐ |
| 30-50 | 中性 | ➖ |
| <30 | 不推荐 | ❌ |

---

## ⚠️ 重要提醒

1. **仅供参考** - 不是投资建议
2. **人工研判** - 需结合基本面
3. **风险控制** - 分散投资+止损
4. **理性投资** - 股市有风险

---

## 📚 完整文档

- `SELECTOR_GUIDE.md` - 详细使用指南
- `ONLINE_READINESS_REPORT.md` - 系统评估报告
