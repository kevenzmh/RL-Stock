"""
测试模型加载
"""
import sys
sys.path.append('.')

from stable_baselines import PPO2

print("测试加载模型...")

# 测试zip格式
try:
    print("\n1. 测试 quick_test_model.zip")
    model = PPO2.load('models/quick_test_model.zip')
    print("   ✓ 加载成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

# 测试pkl格式
try:
    print("\n2. 测试 ppo2_enhanced_100000.pkl")
    model = PPO2.load('models/ppo2_enhanced_100000.pkl')
    print("   ✓ 加载成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

try:
    print("\n3. 测试 ppo2_stock_100000.pkl")
    model = PPO2.load('models/ppo2_stock_100000.pkl')
    print("   ✓ 加载成功")
except Exception as e:
    print(f"   ✗ 失败: {e}")

print("\n测试完成!")
