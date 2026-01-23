"""
实时数据获取模块
支持多种数据源获取最新股票数据
"""
import pandas as pd
import numpy as np
import baostock as bs
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RealtimeDataFetcher:
    """实时数据获取器"""
    
    def __init__(self, source='baostock'):
        """
        初始化数据获取器
        
        Args:
            source: 数据源 ('baostock', 'tushare', 'sina')
        """
        self.source = source
        self.lg = None
        
        if source == 'baostock':
            self._init_baostock()
    
    def _init_baostock(self):
        """初始化baostock连接"""
        self.lg = bs.login()
        if self.lg.error_code != '0':
            raise Exception(f"Baostock登录失败: {self.lg.error_msg}")
        print("✓ Baostock连接成功")
    
    def get_latest_data(self, stock_code, days=60):
        """
        获取最新的股票数据
        
        Args:
            stock_code: 股票代码 (如 'sh.600036')
            days: 获取最近N天的数据 (默认60天)
        
        Returns:
            DataFrame: 股票数据
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        if self.source == 'baostock':
            return self._get_baostock_data(stock_code, start_date, end_date)
        else:
            raise NotImplementedError(f"数据源 {self.source} 暂未实现")
    
    def _get_baostock_data(self, stock_code, start_date, end_date):
        """从baostock获取数据"""
        
        # 查询历史行情数据
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,volume,amount,pctChg,tradestatus,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 后复权
        )
        
        if rs.error_code != '0':
            raise Exception(f"查询数据失败: {rs.error_msg}")
        
        # 转换为DataFrame
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        if len(df) == 0:
            raise Exception(f"股票 {stock_code} 无数据")
        
        # 数据类型转换
        df = self._convert_dtypes(df)
        
        # 获取估值数据
        df_valuation = self._get_valuation_data(stock_code, start_date, end_date)
        
        # 合并数据
        if df_valuation is not None and len(df_valuation) > 0:
            df = pd.merge(df, df_valuation, on='date', how='left')
        else:
            # 如果没有估值数据，添加空列
            df['peTTM'] = np.nan
            df['pbMRQ'] = np.nan
            df['psTTM'] = np.nan
            df['pcfNcfTTM'] = np.nan
        
        # 填充缺失值
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _get_valuation_data(self, stock_code, start_date, end_date):
        """获取估值数据"""
        try:
            rs = bs.query_history_k_data_plus(
                stock_code,
                "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date=start_date,
                end_date=end_date,
                frequency="d"
            )
            
            if rs.error_code != '0':
                return None
            
            data_list = []
            while rs.next():
                data_list.append(rs.get_row_data())
            
            if len(data_list) == 0:
                return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 转换数据类型
            for col in ['peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
        except Exception as e:
            print(f"获取估值数据失败: {e}")
            return None
    
    def _convert_dtypes(self, df):
        """转换数据类型"""
        # 数值列
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 整数列
        int_cols = ['isST']
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    
    def get_all_stocks(self, date=None):
        """
        获取所有股票列表
        
        Args:
            date: 日期 (默认今天)
        
        Returns:
            DataFrame: 股票列表
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        if self.source == 'baostock':
            return self._get_baostock_all_stocks(date)
    
    def _get_baostock_all_stocks(self, date):
        """获取所有股票"""
        # 获取沪深A股列表
        rs = bs.query_all_stock(day=date)
        
        if rs.error_code != '0':
            # 如果当天没数据，尝试前一天
            date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            rs = bs.query_all_stock(day=date)
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 只保留沪深A股
        df = df[df['code'].str.startswith(('sh.6', 'sz.0', 'sz.3'))]
        
        return df
    
    def batch_get_latest_data(self, stock_codes, days=60, verbose=True):
        """
        批量获取多只股票的最新数据
        
        Args:
            stock_codes: 股票代码列表
            days: 获取天数
            verbose: 是否显示进度
        
        Returns:
            dict: {stock_code: DataFrame}
        """
        results = {}
        total = len(stock_codes)
        
        for i, code in enumerate(stock_codes, 1):
            try:
                df = self.get_latest_data(code, days)
                results[code] = df
                
                if verbose:
                    print(f"[{i}/{total}] ✓ {code}: {len(df)}条数据")
            
            except Exception as e:
                if verbose:
                    print(f"[{i}/{total}] ✗ {code}: {str(e)}")
                continue
        
        return results
    
    def close(self):
        """关闭连接"""
        if self.source == 'baostock' and self.lg is not None:
            bs.logout()
            print("✓ Baostock连接已关闭")


# 便捷函数
def get_latest_stock_data(stock_code, days=60):
    """
    获取单只股票最新数据的便捷函数
    
    Args:
        stock_code: 股票代码
        days: 天数
    
    Returns:
        DataFrame
    """
    fetcher = RealtimeDataFetcher()
    try:
        df = fetcher.get_latest_data(stock_code, days)
        return df
    finally:
        fetcher.close()


def get_all_stock_list():
    """
    获取所有股票列表的便捷函数
    
    Returns:
        DataFrame
    """
    fetcher = RealtimeDataFetcher()
    try:
        df = fetcher.get_all_stocks()
        return df
    finally:
        fetcher.close()


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("实时数据获取模块测试")
    print("="*60)
    
    # 测试1: 获取单只股票数据
    print("\n测试1: 获取招商银行最新数据")
    print("-"*60)
    
    try:
        df = get_latest_stock_data('sh.600036', days=30)
        print(f"✓ 成功获取 {len(df)} 条数据")
        print("\n最新5条数据:")
        print(df.tail())
        print(f"\n列名: {list(df.columns)}")
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    # 测试2: 获取股票列表
    print("\n\n测试2: 获取所有股票列表")
    print("-"*60)
    
    try:
        stocks = get_all_stock_list()
        print(f"✓ 成功获取 {len(stocks)} 只股票")
        print("\n前10只股票:")
        print(stocks.head(10))
    except Exception as e:
        print(f"✗ 失败: {e}")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
