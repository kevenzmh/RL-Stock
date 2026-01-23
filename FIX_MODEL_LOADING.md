# 🔧 模型加载问题 - 快速修复指南

## 问题描述

运行 `python test_selector.py` 时出现错误：
```
模型加载失败: A load persistent id instruction was encountered,
but no persistent_load function was specified.
```

## 原因

`.pkl` 格式的模型文件使用了 `stable_baselines` 的特殊保存格式，不能直接用 `pickle.load()` 加载。

## ✅ 已修复

我已经更新了 `model_inference.py`，新版本支持：

1. **多种加载方法** - 自动尝试3种加载方式
2. **优先zip格式** - zip格式更可靠
3. **智能回退** - 如果pkl失败，自动尝试zip
4. **详细错误信息** - 失败时给出明确提示

## 🚀 现在可以运行

### 方式1: 直接测试
```bash
conda activate rl-stock
python test_selector.py
```

### 方式2: 使用批处理
```bash
run_test.bat
```

## 📊 模型优先级

系统会按以下顺序查找模型：
1. ✅ `quick_test_model.zip` ⭐ 推荐（最可靠）
2. `ppo2_enhanced_100000.pkl`
3. `ppo2_stock_100000.pkl`

## ⚠️ 如果仍然失败

### 方案A: 使用quick_test_model.zip

这个模型是训练时用 `.save()` 方法保存的，最可靠：
```bash
# 测试能否加载
python test_load.bat
```

### 方案B: 重新训练生成新模型

如果所有模型都无法加载，重新快速训练一个：

```bash
conda activate rl-stock

# 快速训练（10-15分钟）
python quick_train_test.py

# 或者完整训练（1-2小时）
python main_enhanced.py
```

训练完成后会生成新的模型文件，然后再运行选股脚本。

### 方案C: 检查环境

确保你在正确的conda环境中：
```bash
# 检查当前环境
conda info --envs

# 激活环境
conda activate rl-stock

# 检查stable_baselines版本
python -c "import stable_baselines; print(stable_baselines.__version__)"
# 应该输出: 2.10.0
```

## 🆕 新版model_inference.py的改进

### 智能加载流程

```
1. 尝试 PPO2.load(path)
   ↓ 失败
2. 尝试 PPO2.load(path_without_extension)
   ↓ 失败
3. 如果是.pkl，尝试加载对应的.zip
   ↓ 失败
4. 显示详细错误信息和解决方案
```

### 自动选择最佳模型

```python
# 优先使用.zip格式（更可靠）
priority_models = [
    'quick_test_model.zip',        # 最推荐
    'ppo2_enhanced_100000.pkl',
    'ppo2_stock_100000.pkl',
]
```

## 📝 验证修复

运行这个命令验证修复是否成功：

```bash
python test_selector.py
```

**预期输出**：
```
================================================================================
🧪 选股系统快速测试
================================================================================

【测试1】实时数据获取模块
--------------------------------------------------------------------------------
✓ 找到模型: models\quick_test_model.zip

加载模型: models\quick_test_model.zip
尝试方法1: PPO2.load()...
✓ 模型加载成功 (PPO2.load)

✓ 成功获取招商银行数据: 30 条
  日期范围: 2024-12-23 ~ 2025-01-17
  数据列: 11 列

【测试2】模型推理引擎
--------------------------------------------------------------------------------
✓ 模型推理成功
  综合得分: XX.XX
  预期收益: X.XX%
  推荐等级: XXX

【测试3】小规模选股测试 (5只股票)
--------------------------------------------------------------------------------
...
```

## 🎉 修复完成

如果看到上面的输出，说明修复成功！现在可以：

1. ✅ 运行完整测试
   ```bash
   python test_selector.py
   ```

2. ✅ 开始选股
   ```bash
   python simple_selector.py
   ```

3. ✅ 查看结果
   ```bash
   # 打开生成的CSV文件
   ```

## 📞 仍有问题？

如果按照上述步骤仍然失败，请提供：

1. 完整的错误信息
2. Python版本: `python --version`
3. stable_baselines版本: `pip show stable-baselines`
4. models目录下的文件列表: `dir models`

---

**修复日期**: 2026-01-22  
**修复内容**: 增强模型加载逻辑，支持多种格式和自动回退  
**状态**: ✅ 已修复
