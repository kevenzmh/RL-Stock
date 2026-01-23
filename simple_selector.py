"""
简化版股票选股脚本
快速选出最有潜力的股票

使用方法:
    python simple_selector.py                    # 选出Top 10
    python simple_selector.py --top 20           # 选出Top 20
    python simple_selector.py --pool 100         # 从100只股票中选
    python simple_selector.py --min-score 50     # 最低分数要求50
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_data import RealtimeDataFetcher
from model_inference import ModelInference


def select_stocks(top_n=10, stock_pool_size=100, min_score=30, days=60, output_file=None):
    """
    选股主函数
    
    Args:
        top_n: 返回Top N只股票
        stock_pool_size: 候选股票池大小
        min_score: 最低分数要求
        days: 获取最近N天数据
        output_file: 输出文件路径
    
    Returns:
        DataFrame: 选股结果
    """
    print("="*80)
    print("🚀 RL-Stock 智能选股系统")
    print("="*80)
    
    # 步骤1: 获取股票列表
    print("\n【步骤1/4】获取股票列表")
    print("-"*80)
    
    fetcher = RealtimeDataFetcher()
    
    try:
        all_stocks = fetcher.get_all_stocks()
        print(f"✓ 获取到 {len(all_stocks)} 只股票")
        
        # 过滤掉ST股票
        all_stocks = all_stocks[~all_stocks['code_name'].str.contains('ST', na=False)]
        print(f"✓ 过滤ST股票后剩余 {len(all_stocks)} 只")
        
        # 随机选择候选股票池
        if len(all_stocks) > stock_pool_size:
            stock_pool = all_stocks.sample(n=stock_pool_size, random_state=42)
            print(f"✓ 随机抽取 {stock_pool_size} 只股票作为候选池")
        else:
            stock_pool = all_stocks
        
        stock_codes = stock_pool['code'].tolist()
        print(f"✓ 候选股票池: {len(stock_codes)} 只")
    
    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")
        fetcher.close()
        return None
    
    # 步骤2: 批量获取股票数据
    print(f"\n【步骤2/4】获取最近{days}天的股票数据")
    print("-"*80)
    
    try:
        stock_data = fetcher.batch_get_latest_data(stock_codes, days=days, verbose=True)
        print(f"\n✓ 成功获取 {len(stock_data)} 只股票的数据")
    
    except Exception as e:
        print(f"✗ 获取数据失败: {e}")
        return None
    
    finally:
        fetcher.close()
    
    if len(stock_data) == 0:
        print("✗ 没有成功获取任何股票数据")
        return None
    
    # 步骤3: 模型预测和评分
    print(f"\n【步骤3/4】AI模型预测和评分")
    print("-"*80)
    
    try:
        engine = ModelInference()
        results = engine.get_top_stocks(stock_data, top_n=top_n, min_score=min_score)
        
        if len(results) == 0:
            print(f"✗ 没有找到评分 >= {min_score} 的股票")
            return None
        
        print(f"\n✓ 完成预测，找到 {len(results)} 只符合条件的股票")
    
    except Exception as e:
        print(f"✗ 模型预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤4: 输出结果
    print(f"\n【步骤4/4】生成选股结果")
    print("-"*80)
    
    # 添加股票名称
    stock_name_map = dict(zip(stock_pool['code'], stock_pool['code_name']))
    results['stock_name'] = results['stock_code'].map(stock_name_map)
    
    # 重新排列列顺序
    cols = ['stock_code', 'stock_name', 'score', 'total_return', 'total_profit', 
            'sharpe_ratio', 'recommendation', 'max_profit', 'min_profit', 'volatility']
    results = results[cols]
    
    # 保存到文件
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'stock_selection_{timestamp}.csv'
    
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 结果已保存到: {output_file}")
    
    return results


def print_results(results):
    """美化打印结果"""
    if results is None or len(results) == 0:
        return
    
    print("\n" + "="*80)
    print("📊 选股结果 (Top {})".format(len(results)))
    print("="*80)
    
    print(f"\n{'排名':<6}{'股票代码':<12}{'股票名称':<15}{'得分':<8}{'预期收益':<12}{'推荐等级':<10}")
    print("-"*80)
    
    for idx, row in results.iterrows():
        rank = idx + 1
        code = row['stock_code']
        name = row['stock_name'][:10]  # 限制长度
        score = row['score']
        ret = row['total_return']
        rec = row['recommendation']
        
        # 根据推荐等级选择符号
        if rec == '强烈推荐':
            symbol = '🔥'
        elif rec == '推荐':
            symbol = '⭐'
        elif rec == '中性':
            symbol = '➖'
        else:
            symbol = '❌'
        
        print(f"{rank:<6}{code:<12}{name:<15}{score:<8.2f}{ret:>10.2f}%  {symbol} {rec}")
    
    print("-"*80)
    
    # 统计信息
    avg_score = results['score'].mean()
    avg_return = results['total_return'].mean()
    
    print(f"\n📈 统计信息:")
    print(f"   平均得分: {avg_score:.2f}")
    print(f"   平均预期收益: {avg_return:.2f}%")
    print(f"   推荐等级分布:")
    
    for rec in ['强烈推荐', '推荐', '中性', '不推荐']:
        count = len(results[results['recommendation'] == rec])
        if count > 0:
            print(f"      {rec}: {count} 只")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='RL-Stock 智能选股系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python simple_selector.py                    # 默认选出Top 10
  python simple_selector.py --top 20           # 选出Top 20
  python simple_selector.py --pool 200         # 从200只股票中筛选
  python simple_selector.py --min-score 50     # 最低分数要求50分
  python simple_selector.py --days 90          # 使用90天数据
  python simple_selector.py -o my_picks.csv    # 保存到指定文件
        """
    )
    
    parser.add_argument('--top', '-t', type=int, default=10,
                        help='返回Top N只股票 (默认: 10)')
    parser.add_argument('--pool', '-p', type=int, default=100,
                        help='候选股票池大小 (默认: 100)')
    parser.add_argument('--min-score', '-s', type=float, default=30,
                        help='最低分数要求 (默认: 30)')
    parser.add_argument('--days', '-d', type=int, default=60,
                        help='获取最近N天数据 (默认: 60)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件路径 (默认: 自动生成)')
    
    args = parser.parse_args()
    
    # 执行选股
    results = select_stocks(
        top_n=args.top,
        stock_pool_size=args.pool,
        min_score=args.min_score,
        days=args.days,
        output_file=args.output
    )
    
    # 打印结果
    print_results(results)
    
    print("\n" + "="*80)
    print("✅ 选股完成!")
    print("="*80)
    
    if results is not None and len(results) > 0:
        print("\n💡 提示:")
        print("   1. 以上结果仅供参考，不构成投资建议")
        print("   2. 请结合基本面分析和市场环境判断")
        print("   3. 控制仓位，设置止损，理性投资")
        print("   4. 股市有风险，投资需谨慎")


if __name__ == "__main__":
    main()
