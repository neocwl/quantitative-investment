from pydoc import describe
import tushare as ts
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
from sqlalchemy import create_engine 
import time
import os
# ts.set_token('9f0710fa21d2d373cde852f5e6b71f0d1b2fc898815e2183b176809b')
engine_ts = create_engine('mysql://root:silence913@localhost:3306/stock')

# 获取所有股票代码等信息存入本地csv中
# pro = ts.pro_api()
# data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
# data.to_csv('stock_names_all.csv')

# 获取股票代号
# stock_code = data.loc[:, 'ts_code']


# 下载所有已复权股票数据到本地
# for code in stock_code:
#     try:
#         df = ts.pro_bar(ts_code=code, adj='qfq', start_date='20010823')
#         df.sort_values(by="trade_date", ascending=True, inplace=True)
#         df.to_csv("./stock_daily/{}.csv".format(code), index=False)
#         print(code)
#     except Exception as e:
#         print('error' + code)

# 获取上证指数数据用于分析
# df = ts.pro_bar(ts_code='000001.SH', asset='I').sort_values(by="trade_date")

# 将保存到本地的股票根据交易所信息存入数据库中
for path in os.listdir("./stock_daily/"):
    df = pd.read_csv("./stock_daily/" + path)
    # print(df.dtypes)
    try:
        if path.endswith("SZ.csv"):
            df.to_sql('stock_SZ', engine_ts, index=False, if_exists='append')
        elif path.endswith("SH.csv"):
            df.to_sql('stock_SH', engine_ts, index=False, if_exists='append')
        elif path.endswith("BJ.csv"):
            df.to_sql('stock_BJ', engine_ts, index=False, if_exists='append')
        print("succeed: " + path)
    except Exception as e:
        print(e)
        print("error: " + path)
        break