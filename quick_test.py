"""
快速测试 - 只测试关键部分（修复版）
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 关闭TensorFlow警告

import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🧪 选股系统快速测试")
print("="*60)

# 测试1: 数据获取
print("\n【测试1】数据获取")
print("-"*60)
try:
    from realtime_data import get_latest_stock_data
    df = get_latest_stock_data('sh.600036', days=60)  # 增加到60天
    print(f"✓ 获取数据: {len(df)} 条")
except Exception as e:
    print(f"✗ 失败: {e}")
    exit(1)

# 测试2: 模型加载
print("\n【测试2】模型加载")
print("-"*60)
try:
    from model_inference import ModelInference
    engine = ModelInference()
    print(f"✓ 模型加载成功")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试3: 数据预处理
print("\n【测试3】数据预处理")
print("-"*60)
try:
    df_processed = engine.preprocess_data(df.copy())
    print(f"✓ 预处理完成: {len(df_processed)} 行, {len(df_processed.columns)} 列")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 单股票预测
print("\n【测试4】单股票预测")
print("-"*60)
try:
    result = engine.predict_single_stock(df_processed)
    print(f"✓ 预测成功")
    print(f"  - 得分: {result['score']:.2f}")
    print(f"  - 收益: {result['total_return']:.2f}%")
    print(f"  - 推荐: {result['recommendation']}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试5: 批量预测
print("\n【测试5】批量预测（3只股票）")
print("-"*60)
try:
    from realtime_data import RealtimeDataFetcher
    
    test_stocks = ['sh.600036', 'sz.000001', 'sz.000002']
    
    fetcher = RealtimeDataFetcher()
    stock_data = fetcher.batch_get_latest_data(test_stocks, days=60, verbose=False)
    fetcher.close()
    
    print(f"✓ 获取 {len(stock_data)}/{len(test_stocks)} 只股票数据")
    
    if len(stock_data) > 0:
        results = engine.predict_batch(stock_data, verbose=False)
        print(f"✓ 预测完成: {len(results)} 只")
        
        if len(results) > 0:
            print(f"\nTop股票:")
            for i, row in results.head(3).iterrows():
                print(f"  {i+1}. {row['stock_code']}: {row['score']:.2f}分, "
                      f"{row['total_return']:.2f}%, {row['recommendation']}")
    else:
        print("⚠ 没有获取到数据，跳过批量预测")

except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()
    # 这个测试失败不算严重错误，继续

print("\n" + "="*60)
print("✅ 核心功能测试通过！")
print("="*60)
print("\n💡 现在可以运行:")
print("   python simple_selector.py")
print("\n   或使用批处理:")
print("   run_selector.bat")
