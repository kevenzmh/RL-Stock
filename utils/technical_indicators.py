"""
技术指标计算模块
包含常用的技术分析指标
"""
import numpy as np
import pandas as pd


def calculate_ma(df, periods=[5, 10, 20, 60]):
    """
    计算移动平均线
    
    Args:
        df: DataFrame with 'close' column
        periods: 移动平均周期列表
    
    Returns:
        DataFrame with MA columns
    """
    result = df.copy()
    for period in periods:
        result[f'ma{period}'] = result['close'].rolling(window=period).mean()
    return result


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    
    Args:
        df: DataFrame with 'close' column
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期
    
    Returns:
        DataFrame with MACD, Signal, Histogram columns
    """
    result = df.copy()
    
    # 计算EMA
    ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
    
    # MACD线
    result['macd'] = ema_fast - ema_slow
    
    # 信号线
    result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
    
    # 柱状图
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    return result


def calculate_rsi(df, period=14):
    """
    计算RSI指标
    
    Args:
        df: DataFrame with 'close' column
        period: RSI周期
    
    Returns:
        DataFrame with RSI column
    """
    result = df.copy()
    
    # 计算价格变化
    delta = result['close'].diff()
    
    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 计算RS和RSI
    rs = gain / loss
    result['rsi'] = 100 - (100 / (1 + rs))
    
    return result


def calculate_kdj(df, n=9, m1=3, m2=3):
    """
    计算KDJ指标
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        n: RSV周期
        m1: K值平滑周期
        m2: D值平滑周期
    
    Returns:
        DataFrame with K, D, J columns
    """
    result = df.copy()
    
    # 计算RSV
    low_n = result['low'].rolling(window=n).min()
    high_n = result['high'].rolling(window=n).max()
    
    rsv = (result['close'] - low_n) / (high_n - low_n) * 100
    
    # 计算K, D, J
    result['kdj_k'] = rsv.ewm(com=m1-1, adjust=False).mean()
    result['kdj_d'] = result['kdj_k'].ewm(com=m2-1, adjust=False).mean()
    result['kdj_j'] = 3 * result['kdj_k'] - 2 * result['kdj_d']
    
    return result


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    计算布林带
    
    Args:
        df: DataFrame with 'close' column
        period: 移动平均周期
        std_dev: 标准差倍数
    
    Returns:
        DataFrame with BB_UPPER, BB_MIDDLE, BB_LOWER columns
    """
    result = df.copy()
    
    # 中轨
    result['bb_middle'] = result['close'].rolling(window=period).mean()
    
    # 标准差
    std = result['close'].rolling(window=period).std()
    
    # 上轨和下轨
    result['bb_upper'] = result['bb_middle'] + (std * std_dev)
    result['bb_lower'] = result['bb_middle'] - (std * std_dev)
    
    # 布林带宽度
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    
    return result


def calculate_volume_indicators(df):
    """
    计算成交量指标
    
    Args:
        df: DataFrame with 'volume', 'close' columns
    
    Returns:
        DataFrame with volume indicators
    """
    result = df.copy()
    
    # 成交量移动平均
    result['volume_ma5'] = result['volume'].rolling(window=5).mean()
    result['volume_ma10'] = result['volume'].rolling(window=10).mean()
    
    # 量比 (当前成交量 / 5日平均成交量)
    result['volume_ratio'] = result['volume'] / result['volume_ma5']
    
    # OBV (能量潮)
    obv = []
    obv_value = 0
    
    for i in range(len(result)):
        if i == 0:
            obv.append(0)
        else:
            if result['close'].iloc[i] > result['close'].iloc[i-1]:
                obv_value += result['volume'].iloc[i]
            elif result['close'].iloc[i] < result['close'].iloc[i-1]:
                obv_value -= result['volume'].iloc[i]
            obv.append(obv_value)
    
    result['obv'] = obv
    
    return result


def add_all_technical_indicators(df):
    """
    添加所有技术指标
    
    Args:
        df: 原始DataFrame
    
    Returns:
        包含所有技术指标的DataFrame
    """
    result = df.copy()
    
    # 移动平均线
    result = calculate_ma(result, periods=[5, 10, 20, 60])
    
    # MACD
    result = calculate_macd(result)
    
    # RSI
    result = calculate_rsi(result)
    
    # KDJ
    result = calculate_kdj(result)
    
    # 布林带
    result = calculate_bollinger_bands(result)
    
    # 成交量指标
    result = calculate_volume_indicators(result)
    
    # 价格变化率
    result['price_change'] = result['close'].pct_change()
    
    # 真实波动幅度均值 (ATR)
    high_low = result['high'] - result['low']
    high_close = np.abs(result['high'] - result['close'].shift())
    low_close = np.abs(result['low'] - result['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    result['atr'] = true_range.rolling(14).mean()
    
    return result
