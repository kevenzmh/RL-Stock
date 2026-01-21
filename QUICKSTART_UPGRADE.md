# 🚀 RL-Stock 快速入门指南

## 📦 安装依赖

### 方法1: 使用pip (推荐)

```bash
# 基础依赖
pip install stable-baselines3[extra]
pip install torch torchvision
pip install pandas numpy matplotlib
pip install gym==0.21.0
pip install tushare akshare
pip install transformers sentencepiece

# 可选: GPU支持
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 方法2: 使用requirements

```bash
pip install -r requirements_upgrade.txt
```

requirements_upgrade.txt内容:
```
stable-baselines3[extra]>=2.0.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
gym==0.21.0
tushare>=1.4.0
akshare>=1.10.0
transformers>=4.30.0
scikit-learn>=1.2.0
```

---

## 🎯 快速测试 (5分钟)

### 1. 测试现有PPO模型

```bash
python main_enhanced.py
```

预期输出:
```
==============================================================
增强版RL股票交易系统
==============================================================
股票代码: sh.000001
训练步数: 100000
==============================================================

[训练中...]

最终评估报告:
  总收益率:     25.34%
  年化收益率:   18.56%
  夏普比率:      1.523
```

---

## 🔄 升级到SAC算法 (15分钟)

### 1. 运行SAC训练

```bash
python main_sac.py --stock sh.000001 --steps 100000
```

### 2. 对比PPO和SAC

```bash
python main_sac.py --stock sh.000001 --steps 100000 --compare
```

预期输出:
```
==============================================================
性能对比
==============================================================
指标                 SAC             PPO            提升
-----------------------------------------------------------------
总收益率            32.45%          25.34%        +28.06%
年化收益率          23.12%          18.56%        +24.57%
夏普比率             1.876           1.523        +23.18%
最大回撤            12.34%          15.67%        -21.24%
```

---

## 🧠 升级到LSTM+SAC (30分钟)

### 1. 标准LSTM

```bash
python main_lstm_sac.py --stock sh.000001 --steps 200000 --window 10
```

### 2. 带注意力机制的LSTM

```bash
python main_lstm_sac.py --stock sh.000001 --steps 200000 --window 10 --attention
```

---

## 📊 完整评估流程

### 1. 单股票完整评估

```bash
# 训练
python main_lstm_sac.py --stock sh.600036 --steps 300000 --attention

# 评估(自动生成图表和报告)
# 结果保存在 ./img/sh.600036_lstm_sac.png
```

### 2. 批量测试多只股票

创建 `batch_test.py`:
```python
import subprocess

stocks = ['sh.000001', 'sh.600036', 'sz.300677', 'sh.600519', 'sz.000858']

for stock in stocks:
    print(f"\n{'='*60}")
    print(f"测试股票: {stock}")
    print('='*60)
    
    subprocess.run([
        'python', 'main_lstm_sac.py',
        '--stock', stock,
        '--steps', '200000',
        '--attention'
    ])
```

运行:
```bash
python batch_test.py
```

---

## 📈 数据获取

### 方法1: 使用baostock (免费)

```bash
python get_stock_data.py
```

### 方法2: 使用Tushare Pro (推荐)

```python
# 注册账号: https://tushare.pro/register
# 获取token: https://tushare.pro/user/token

# download_tushare_data.py
import tushare as ts

ts.set_token('YOUR_TOKEN_HERE')
pro = ts.pro_api()

# 下载日线数据
df = pro.daily(
    ts_code='000001.SZ',
    start_date='20200101',
    end_date='20240101'
)

# 下载分钟数据
df_min = ts.pro_bar(
    ts_code='000001.SZ',
    freq='1min',
    start_date='20240101',
    end_date='20240131'
)
```

### 方法3: 使用AKShare (免费实时)

```python
import akshare as ak

# 实时行情
stock_zh_a_spot_df = ak.stock_zh_a_spot_em()

# 历史数据
stock_hist = ak.stock_zh_a_hist(
    symbol="000001",
    period="daily",
    start_date="20200101",
    end_date="20240101",
    adjust="qfq"
)
```

---

## 🔬 高级功能

### 1. 财务数据集成

```bash
# 下载财务数据
python scripts/download_financial_data.py --stock sh.600036

# 训练融合模型
python main_fundamental.py --stock sh.600036 --steps 200000
```

### 2. 新闻情感分析

```bash
# 下载并分析新闻
python scripts/download_news.py --stock sh.600036 --days 365

# 训练情感增强模型
python main_sentiment.py --stock sh.600036 --steps 200000
```

### 3. 多智能体投资组合

```bash
# 训练投资组合管理智能体
python main_portfolio.py \
    --stocks sh.000001,sh.600036,sz.300677 \
    --balance 100000 \
    --steps 500000
```

---

## 📊 可视化和监控

### 1. TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir=./log_sac

# 访问 http://localhost:6006
```

### 2. 生成对比报告

```bash
python generate_comparison_report.py \
    --models ppo,sac,lstm_sac \
    --stock sh.000001
```

### 3. 交互式回测分析

```bash
# 启动Jupyter
jupyter notebook

# 打开 analysis.ipynb
```

---

## 🐛 常见问题解决

### Q1: ImportError: No module named 'stable_baselines3'

```bash
pip install stable-baselines3[extra]
```

### Q2: CUDA out of memory

解决方案:
```python
# 在main_lstm_sac.py中修改
buffer_size=100_000,  # 减小buffer
batch_size=64,        # 减小batch
```

或使用CPU:
```python
device='cpu'
```

### Q3: 训练不收敛

检查:
1. 数据质量 (`python scripts/validate_data.py`)
2. 奖励函数 (查看log)
3. 超参数 (降低learning_rate)
4. 环境配置 (检查observation_space)

### Q4: 数据下载失败

方案1: 更换数据源
```python
# 从baostock切换到akshare
import akshare as ak
df = ak.stock_zh_a_hist(symbol="000001", period="daily")
```

方案2: 使用本地CSV
```bash
# 手动下载后放入 stockdata/ 目录
```

### Q5: 模型加载错误

确保版本一致:
```bash
# 检查版本
python -c "import stable_baselines3; print(stable_baselines3.__version__)"

# 如果版本不同,重新训练
python main_sac.py --stock sh.000001 --steps 100000
```

---

## 📚 学习路径

### 初级 (1周)
1. ✅ 理解强化学习基本概念
2. ✅ 运行现有PPO模型
3. ✅ 修改简单参数观察效果
4. ✅ 尝试不同股票

### 中级 (2-3周)
1. ⏰ 掌握SAC算法原理
2. ⏰ 添加新的技术指标
3. ⏰ 调优超参数
4. ⏰ 理解奖励函数设计

### 高级 (1-2月)
1. 🔄 实现LSTM序列模型
2. 🔄 集成基本面分析
3. 🔄 开发多智能体系统
4. 🔄 实盘模拟交易

---

## 🎓 推荐资源

### 在线课程
1. [Deep RL Course](https://huggingface.co/deep-rl-course)
2. [Stable-Baselines3 Tutorial](https://stable-baselines3.readthedocs.io/)
3. [金融强化学习](https://github.com/AI4Finance-Foundation/FinRL)

### 书籍
1. "Reinforcement Learning: An Introduction" - Sutton & Barto
2. "深度强化学习" - 张伟楠
3. "Python金融大数据分析" - Yves Hilpisch

### 论文
1. Soft Actor-Critic (SAC)
2. Proximal Policy Optimization (PPO)
3. Deep Deterministic Policy Gradient (DDPG)

---

## 💡 最佳实践

### 1. 数据准备
- ✅ 始终检查数据质量
- ✅ 处理缺失值和异常值
- ✅ 归一化特征到[0,1]
- ✅ 保留验证集

### 2. 模型训练
- ✅ 从小步数开始(50k)
- ✅ 监控训练曲线
- ✅ 定期保存checkpoint
- ✅ 使用早停策略

### 3. 评估验证
- ✅ 多次随机种子测试
- ✅ 不同市场环境验证
- ✅ 滚动窗口回测
- ✅ Monte Carlo模拟

### 4. 风险控制
- ✅ 设置最大回撤限制
- ✅ 仓位管理规则
- ✅ 止损止盈策略
- ✅ 分散投资

---

## 📞 获取帮助

### GitHub Issues
提交问题: [项目Issues](https://github.com/your-repo/RL-Stock/issues)

### 讨论区
技术讨论: [Discussions](https://github.com/your-repo/RL-Stock/discussions)

### 邮件支持
联系邮箱: support@example.com

---

## 🎉 恭喜!

你已经完成了快速入门指南!

下一步:
1. ✅ 尝试训练自己的第一个模型
2. ✅ 阅读完整的升级路线图 (UPGRADE_ROADMAP.md)
3. ✅ 查看项目检查报告了解详细改进方案
4. ✅ 加入社区,分享你的经验

---

**最后更新**: 2025-01-21
**版本**: v1.0
**许可**: MIT License
