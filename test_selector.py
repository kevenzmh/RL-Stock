"""
快速测试选股系统
验证所有模块是否正常工作
"""
import os
import sys

print("="*80)
print("🧪 选股系统快速测试")
print("="*80)

# 测试1: 实时数据获取
print("\n【测试1】实时数据获取模块")
print("-"*80)

try:
    from realtime_data import get_latest_stock_data
    
    df = get_latest_stock_data('sh.600036', days=30)
    print(f"✓ 成功获取招商银行数据: {len(df)} 条")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  数据列: {len(df.columns)} 列")

except Exception as e:
    print(f"✗ 失败: {e}")
    print("\n请检查:")
    print("  1. 网络连接是否正常")
    print("  2. baostock库是否已安装: pip install baostock")
    sys.exit(1)

# 测试2: 模型推理
print("\n【测试2】模型推理引擎")
print("-"*80)

try:
    from model_inference import ModelInference
    
    print("\n正在加载模型...")
    engine = ModelInference()
    
    print("\n正在预处理数据...")
    df_processed = engine.preprocess_data(df.copy())
    
    print("\n正在推理预测...")
    result = engine.predict_single_stock(df_processed)
    
    print(f"\n✓ 模型推理成功")
    print(f"  综合得分: {result['score']:.2f}")
    print(f"  预期收益: {result['total_return']:.2f}%")
    print(f"  推荐等级: {result['recommendation']}")

except Exception as e:
    print(f"\n✗ 失败: {e}")
    print("\n可能的原因:")
    print("  1. 模型文件加载失败")
    print("  2. stable_baselines版本不匹配")
    print("  3. 依赖库缺失")
    print("\n解决方案:")
    print("  1. 查看 FIX_MODEL_LOADING.md 获取详细解决方案")
    print("  2. 确保在 rl-stock 环境中: conda activate rl-stock")
    print("  3. 尝试重新训练模型: python quick_train_test.py")
    
    import traceback
    print("\n详细错误信息:")
    traceback.print_exc()
    sys.exit(1)

# 测试3: 小规模选股
print("\n【测试3】小规模选股测试 (5只股票)")
print("-"*80)

try:
    from realtime_data import RealtimeDataFetcher
    
    # 测试股票列表
    test_stocks = [
        'sh.600036',  # 招商银行
        'sh.600519',  # 贵州茅台
        'sz.000001',  # 平安银行
        'sz.000002',  # 万科A
        'sz.300750',  # 宁德时代
    ]
    
    print("\n正在获取测试股票数据...")
    fetcher = RealtimeDataFetcher()
    stock_data = fetcher.batch_get_latest_data(test_stocks, days=60, verbose=True)
    fetcher.close()
    
    print(f"\n✓ 获取数据成功: {len(stock_data)}/{len(test_stocks)}")
    
    if len(stock_data) == 0:
        print("✗ 没有成功获取任何数据，跳过预测")
    else:
        # 预测
        print("\n正在预测...")
        results = engine.predict_batch(stock_data, verbose=True)
        
        print(f"\n✓ 预测完成: {len(results)} 只股票")
        
        if len(results) > 0:
            print("\n结果预览:")
            print(results[['stock_code', 'score', 'total_return', 'recommendation']].to_string(index=False))

except Exception as e:
    print(f"\n✗ 失败: {e}")
    print("\n这可能是网络问题或数据源问题")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n" + "="*80)
print("✅ 所有测试通过!")
print("="*80)
print("\n💡 系统已就绪，可以运行:")
print("   python simple_selector.py")
print("\n   或者查看更多选项:")
print("   python simple_selector.py --help")
print("\n   Windows用户可以双击:")
print("   run_selector.bat")
