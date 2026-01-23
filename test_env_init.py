"""
简单测试：验证环境初始化
"""
import pandas as pd
import sys
sys.path.append('.')

print("测试环境初始化...")

# 创建测试数据
print("\n1. 创建小数据集（21行）")
from realtime_data import get_latest_stock_data
df = get_latest_stock_data('sh.600036', days=21)
print(f"   数据行数: {len(df)}")

# 添加技术指标
print("\n2. 添加技术指标")
from utils.technical_indicators import add_all_technical_indicators
df = add_all_technical_indicators(df)
print(f"   处理后行数: {len(df)}")
print(f"   列数: {len(df.columns)}")

# 创建环境
print("\n3. 创建交易环境")
from rlenv.StockTradingEnv_enhanced import StockTradingEnvEnhanced
env = StockTradingEnvEnhanced(df)
print(f"   ✓ 环境创建成功")

# 重置环境（多次测试）
print("\n4. 测试环境重置（10次）")
for i in range(10):
    try:
        obs = env.reset()
        print(f"   第{i+1}次: current_step={env.current_step}, obs_shape={obs.shape}")
    except Exception as e:
        print(f"   第{i+1}次: ✗ 失败 - {e}")
        import traceback
        traceback.print_exc()
        break

print("\n✅ 测试完成！")
