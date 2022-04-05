import pandas as pd
import tushare as ts
import pymysql
pymysql.install_as_MySQLdb()
from sqlalchemy import create_engine 


engine_ts = create_engine('mysql://root:silence913@localhost:3306/mysql')

def read_data():
    sql = """SELECT * FROM stock_basic LIMIT 20"""
    df = pd.read_sql_query(sql, engine_ts)
    return df


def write_data(df):
    res = df.to_sql('stock_basic', engine_ts, index=False, if_exists='append', chunksize=5000)
    print(res)


def get_data():
    pro = ts.pro_api()
    df = pro.stock_basic()
    return df


if __name__ == '__main__':
    df = read_data()
    # df = get_data()
    # write_data(df)
    print(df)