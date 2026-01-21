"""
数据预处理模块
包含数据清洗、标准化、异常值处理等功能
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, method='standard'):
        """
        Args:
            method: 标准化方法 ('standard', 'robust', 'minmax')
        """
        self.method = method
        self.scaler = None
        self.feature_columns = None
        
    def handle_missing_values(self, df, strategy='forward_fill'):
        """
        处理缺失值
        
        Args:
            df: DataFrame
            strategy: 处理策略
                - 'forward_fill': 前向填充
                - 'backward_fill': 后向填充
                - 'interpolate': 线性插值
                - 'drop': 删除含缺失值的行
        
        Returns:
            处理后的DataFrame
        """
        result = df.copy()
        
        # 检查缺失值
        missing_count = result.isnull().sum()
        if missing_count.sum() > 0:
            print(f"发现缺失值: \n{missing_count[missing_count > 0]}")
        
        if strategy == 'forward_fill':
            result = result.fillna(method='ffill')
            # 如果第一行有缺失,用后向填充
            result = result.fillna(method='bfill')
        elif strategy == 'backward_fill':
            result = result.fillna(method='bfill')
            result = result.fillna(method='ffill')
        elif strategy == 'interpolate':
            # 对数值列进行插值
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].interpolate(method='linear')
            # 处理剩余缺失值
            result = result.fillna(method='ffill').fillna(method='bfill')
        elif strategy == 'drop':
            result = result.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result
    
    def detect_outliers(self, df, columns=None, method='iqr', threshold=3):
        """
        检测异常值
        
        Args:
            df: DataFrame
            columns: 要检测的列,None表示所有数值列
            method: 检测方法
                - 'iqr': 四分位距法
                - 'zscore': Z分数法
            threshold: 阈值 (IQR倍数或Z分数)
        
        Returns:
            异常值的布尔掩码
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask[col] = z_scores > threshold
        
        return outlier_mask
    
    def handle_outliers(self, df, columns=None, method='clip', detection_method='iqr'):
        """
        处理异常值
        
        Args:
            df: DataFrame
            columns: 要处理的列
            method: 处理方法
                - 'clip': 裁剪到边界
                - 'winsorize': Winsorize处理
                - 'remove': 删除异常值行
            detection_method: 检测方法 ('iqr' or 'zscore')
        
        Returns:
            处理后的DataFrame
        """
        result = df.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns
        
        outlier_mask = self.detect_outliers(result, columns, method=detection_method)
        
        if method == 'clip':
            for col in columns:
                if col not in result.columns:
                    continue
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'winsorize':
            for col in columns:
                if col not in result.columns:
                    continue
                lower_percentile = result[col].quantile(0.05)
                upper_percentile = result[col].quantile(0.95)
                result[col] = result[col].clip(lower=lower_percentile, upper=upper_percentile)
        
        elif method == 'remove':
            # 删除任何列有异常值的行
            rows_to_remove = outlier_mask.any(axis=1)
            result = result[~rows_to_remove]
            print(f"删除了 {rows_to_remove.sum()} 行异常值")
        
        return result
    
    def normalize_data(self, df, columns=None, fit=True):
        """
        数据标准化
        
        Args:
            df: DataFrame
            columns: 要标准化的列
            fit: 是否拟合scaler (训练时True, 测试时False)
        
        Returns:
            标准化后的DataFrame
        """
        result = df.copy()
        
        if columns is None:
            # 排除非数值列和日期列
            columns = result.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if col not in ['date', 'code']]
        
        self.feature_columns = columns
        
        if fit:
            if self.method == 'standard':
                self.scaler = StandardScaler()
            elif self.method == 'robust':
                # RobustScaler对异常值更鲁棒
                self.scaler = RobustScaler()
            elif self.method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            result[columns] = self.scaler.fit_transform(result[columns])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            result[columns] = self.scaler.transform(result[columns])
        
        return result
    
    def validate_data(self, df):
        """
        验证数据质量
        
        Args:
            df: DataFrame
        
        Returns:
            验证报告字典
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
        }
        
        # 检查数值列的基本统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report[f'{col}_stats'] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'zeros': int((df[col] == 0).sum()),
                'infinite': int(np.isinf(df[col]).sum()),
            }
        
        return report
    
    def check_data_quality(self, df):
        """
        数据质量检查
        
        Args:
            df: DataFrame
        
        Returns:
            质量报告
        """
        quality_report = {
            'total_rows': len(df),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'zero_percentage': ((df == 0).sum() / len(df) * 100).to_dict(),
            'infinite_count': df.isin([np.inf, -np.inf]).sum().to_dict(),
            'duplicates': df.duplicated().sum(),
        }
        
        # 检查数值列的合理性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                quality_report[f'{col}_range'] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'has_negative': bool((df[col] < 0).any()),
                }
        
        return quality_report
    
    def preprocess_pipeline(self, df, fit=True, 
                          handle_missing=True,
                          handle_outliers_flag=True,
                          normalize=False):  # 改为False,因为技术指标已经归一化
        """
        完整的预处理流水线
        
        Args:
            df: DataFrame
            fit: 是否拟合 (训练时True, 测试时False)
            handle_missing: 是否处理缺失值
            handle_outliers_flag: 是否处理异常值
            normalize: 是否标准化 (默认False,技术指标已归一化)
        
        Returns:
            处理后的DataFrame
        """
        result = df.copy()
        
        print("=" * 50)
        print("数据预处理流水线")
        print("=" * 50)
        
        # 1. 验证数据
        print("\n1. 数据验证...")
        validation_report = self.validate_data(result)
        print(f"  总行数: {validation_report['total_rows']}")
        print(f"  总列数: {validation_report['total_columns']}")
        print(f"  重复行: {validation_report['duplicate_rows']}")
        
        # 1.5 质量检查
        quality_report = self.check_data_quality(result)
        print(f"\n  数据质量:")
        if quality_report['duplicates'] > 0:
            print(f"    警告: 发现 {quality_report['duplicates']} 行重复数据")
        
        # 检查无穷值
        infinite_cols = [k for k, v in quality_report['infinite_count'].items() if v > 0]
        if infinite_cols:
            print(f"    警告: 以下列包含无穷值: {', '.join(infinite_cols)}")
            # 替换无穷值为NaN
            result = result.replace([np.inf, -np.inf], np.nan)
        
        # 2. 处理缺失值
        if handle_missing:
            print("\n2. 处理缺失值...")
            missing_before = result.isnull().sum().sum()
            result = self.handle_missing_values(result, strategy='interpolate')
            missing_after = result.isnull().sum().sum()
            print(f"  缺失值: {missing_before} -> {missing_after}")
        
        # 3. 处理异常值
        if handle_outliers_flag:
            print("\n3. 处理异常值...")
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            # 只处理存在的列
            existing_price_cols = [col for col in price_cols if col in result.columns]
            if existing_price_cols:
                result = self.handle_outliers(result, columns=existing_price_cols, method='clip')
        
        # 4. 标准化 (可选)
        if normalize:
            print("\n4. 数据标准化...")
            result = self.normalize_data(result, fit=fit)
        
        print("\n预处理完成!")
        print("=" * 50)
        
        return result
