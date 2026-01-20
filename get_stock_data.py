import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta


OUTPUT = './stockdata'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='1990-01-01',
                 date_end=None):
        self._bs = bs
        self.date_start = date_start
        # 使用最近的交易日（通常是前几天，避免使用今天或未来日期）
        if date_end is None:
            self.date_end = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"

    def get_codes_by_date(self, date):
        print(f"获取日期 {date} 的股票列表...")
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(f"获取到 {len(stock_df)} 只股票")
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date(self.date_end)
        
        if stock_df.empty:
            print("错误：未能获取股票列表，请检查日期是否正确或网络连接")
            return
        
        success_count = 0
        fail_count = 0
        
        for index, row in stock_df.iterrows():
            try:
                print(f'正在处理 [{index+1}/{len(stock_df)}] {row["code"]} {row["code_name"]}')
                df_code = bs.query_history_k_data_plus(
                    row["code"], 
                    self.fields,
                    start_date=self.date_start,
                    end_date=self.date_end
                ).get_data()
                
                if not df_code.empty:
                    # 处理文件名中的特殊字符
                    code_name = row["code_name"].replace('*', '').replace(':', '').replace('?', '') \
                                               .replace('<', '').replace('>', '').replace('|', '')
                    filepath = f'{self.output_dir}/{row["code"]}.{code_name}.csv'
                    df_code.to_csv(filepath, index=False)
                    success_count += 1
                else:
                    print(f'  警告：{row["code"]} 没有数据')
                    fail_count += 1
            except Exception as e:
                print(f'  错误：处理 {row["code"]} 时出错: {str(e)}')
                fail_count += 1
        
        print(f"\n完成！成功: {success_count}, 失败: {fail_count}")


def main():
    # 登录 baostock
    print("正在登录 baostock...")
    login_result = bs.login()
    
    # 检查登录状态
    if login_result.error_code != '0':
        print(f"登录失败！错误代码: {login_result.error_code}")
        print(f"错误信息: {login_result.error_msg}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 尝试升级 baostock: pip install --upgrade baostock")
        print("3. 稍后重试")
        return
    
    print("登录成功！")
    
    try:
        # 使用最近的有效交易日期（避免使用未来日期）
        # baostock 通常数据延迟1-2天
        end_date = '2025-01-17'  # 最近的交易日
        
        # 获取训练数据
        print(f"\n=== 下载训练数据 (截止日期: {end_date}) ===")
        mkdir('./stockdata/train')
        downloader_train = Downloader(
            './stockdata/train', 
            date_start='1990-01-01', 
            date_end=end_date
        )
        downloader_train.run()
        
        # 获取测试数据
        print(f"\n=== 下载测试数据 (截止日期: {end_date}) ===")
        mkdir('./stockdata/test')
        downloader_test = Downloader(
            './stockdata/test', 
            date_start='2019-12-01', 
            date_end=end_date
        )
        downloader_test.run()
        
    finally:
        # 登出
        print("\n正在登出...")
        logout_result = bs.logout()
        if logout_result.error_code == '0':
            print("登出成功！")
        else:
            print(f"登出失败: {logout_result.error_msg}")


if __name__ == '__main__':
    main()
