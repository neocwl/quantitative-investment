from re import S
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tushare as ts
from dateutil.parser import parse
import datetime
# import matplotlib
# matplotlib.use('TkAgg')

from zmq import PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE

CASH = 100000
START_DATE = '20140107'
END_DATE = '20160131'


# 获取交易日信息
pro = ts.pro_api()
trade_cal = pro.trade_cal()
# print(trade_cal)

class Context:
    """保存上下文信息"""
    def __init__(self, cash, start_date, end_date) -> None:
        self.cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {} # 持仓信息
        self.benchmark = None # 基准
        self.date_range = trade_cal[(trade_cal['is_open'] == 1) \
                                     & (trade_cal['cal_date'] >= start_date)\
                                     & (trade_cal['cal_date'] <= end_date)]['cal_date'].values
        # self.dt = parse(start_date) # 回测结束日期
        self.dt = None
# context = Context(1000, '20160101', '20170101')
# print(context.date_range)

context = Context(CASH, START_DATE, END_DATE)

class G:
    pass

g = G()

# 基准收益率
def set_benchmark(security):    # 只支持一支股票作为基准
    context.benchmark = security

def attribute_history(security, count, fields=('open', 'close', 'high', 'low', 'vol')):
    """
    security : 股票代码
    count : 从回测日期往前推count天
    field : 数据属性
    return : 回测日期范围内的数据
    """
    end_date = (context.dt - datetime.timedelta(days=1)).strftime('%Y%m%d')
    start_date = trade_cal[(trade_cal['is_open'] == 1) \
                            & (trade_cal['cal_date'] <= end_date)][-count:].iloc[0, :]['cal_date']
    # print(start_date, end_date)
    return attribute_daterange_history(security, start_date, end_date)

def attribute_daterange_history(security, start_date, end_date, fields=('open', 'close', 'high', 'low', 'vol')):
    """
    security : 股票代码
    start_date: 开始日期
    end_date: 结束日期
    field : 数据属性
    return : 时间范围内的数据
    """
    try:
        f = open('./stock_daily/' + security + '.csv', 'r')
        df = pd.read_csv(f, index_col='trade_date', parse_dates=['trade_date']).loc[start_date:end_date, :]
    except FileNotFoundError:
        df = ts.pro_bar(ts_code=security, start_date=start_date, end_date=end_date)
    return df[list(fields)]

# print(attribute_daterange_history('000001.SZ', '20150101', '20160220'))
# print(attribute_history('000001.SZ', 20))

def get_today_data(security):
    today = context.dt.strftime('%Y%m%d')
    try:
        f = open('./stock_daily/' + security + '.csv', 'r')
        data = pd.read_csv(f, index_col='trade_date', parse_dates=['trade_date']).loc[today, :]
    except FileNotFoundError:
        data = ts.pro_bar(ts_code=security, start_date=today, end_date=today).iloc[0, :]
    except KeyError:
        data = pd.Series()
    return data

# print(get_today_data('000001.SZ'))

def _order(today_data, security, amount):
    """
    today_data: 今天数据
    security: 股票代码
    amount: 操作股票的数量
    """
    if len(today_data) == 0:
        print("今日停牌")
        return

    p = today_data['open']

    # 钱不够
    if context.cash - amount * p < 0:
        amount = int(context.cash / p)
        print("现金不足, 已调整为%d" % amount)

    # 股票以手为交易单位，1手=100股
    if amount % 100 != 0:
        if amount != -context.positions.get(security, 0):  # 全仓卖出
            amount = int(amount / 100) * 100
            print("不是100的倍数，已调整为%d" % amount)
    
    # 要卖出的超过持仓数
    if context.positions.get(security, 0) < -amount:
        amount = -context.positions.get(security, 0)
        print("卖出股票不能超过持仓，已调整为%d" % amount)
    
    # 更新仓位信息，用get方法避免某只股票此时并未加入仓位信息中
    context.positions[security] = context.positions.get(security, 0) + amount
    
    # 更新现金信息
    context.cash -= amount * p

    if context.positions[security] == 0:
        del context.positions[security]

# _order(get_today_data('000001.SZ'), '000001.SZ', 10000000000)
# print(context.positions)
# _order(get_today_data('000001.SZ'), '000001.SZ', -11800)
# print(context.positions)

def order(security, amount):
   today_data = get_today_data(security)
   if len(today_data) == 0:
        print("今日停牌")
        return
   _order(today_data, security, amount)

def order_target(security, amount):
    """
    amount: 买到多少股
    """
    if amount < 0:
        print("数量不能为负，已调整成0")
        amount = 0
    
    today_data = get_today_data(security)

    if len(today_data) == 0:
        print("今日停牌")
        return
    
    hold_amount = context.positions.get(security, 0)    # 当前有多少股，ToDo: T + 1
    delta_amount = amount - hold_amount     # 与买到多少股的差值
    _order(today_data, security, delta_amount)

def order_value(security, value):
    """
    value: 买多少钱的股票
    """
    today_data = get_today_data(security)
    if len(today_data) == 0:
        print("今日停牌")
        return
    # print("today_data:", today_data)
    amount = int(value / today_data['open'])
    _order(today_data, security, amount)

def order_target_value(security, value):
    """
    value: 买到多少钱的股票
    """
    today_data = get_today_data(security)
    if len(today_data) == 0:
        print("今日停牌")
        return
    if value < 0:
        print("价值不能为负，已调整为0")
        value = 0
    
    # 当前拥有股票的总价值
    hold_value = context.positions.get(security, 0) * today_data['open']

    delta_value = value - hold_value    # 计算差的价值
    order_value(security, delta_value)
    
# order('000001.SZ', 100)
# order_value('000003.SZ', 30000)
# order_target('601318.SH', 520)
# order_target_value('600519.SH', 30000)
# print(context.positions)

def run():
    # 画图用的DataFrame对象
    plt_df = pd.DataFrame(index=pd.to_datetime(context.date_range), columns=['value'])
    init_value = context.cash
    initialize(context)
    last_price = {}

    for dt in context.date_range:
        context.dt = parse(dt)  # 字符串变成时间对象
        handle_data(context)
        value = context.cash
        for stock in context.positions:
            # 考虑停牌的情况
            today_data = get_today_data(stock)
            if len(today_data) == 0:
                pass
            else:
                p = get_today_data(stock)['open']
                last_price[stock] = p    
            value += p * context.positions[stock]
        plt_df.loc[dt, 'value'] = value     # 每一天有多少价值
    # print(plt_df)
    plt_df['ratio'] = (plt_df['value'] - init_value) / init_value   # 计算每天收益率

    bm_df = attribute_daterange_history(context.benchmark, context.start_date, context.end_date)
    bm_init = bm_df['open'][0]
    plt_df['benchmark_ratio'] = (bm_df['open'] - bm_init) / bm_init # 基准收益率
    # print(plt_df)
    # print(plt_df['ratio'])
    plt_df[['ratio', 'benchmark_ratio']].plot()
    plt.show()

def initialize(context):
    set_benchmark('601318.SH')
    g.p1 = 5    # 五日均线
    g.p2 = 60   # 六日均线
    g.security = '601318.SH'

def handle_data(context):
    hist = attribute_history(g.security, g.p2)
    ma5 = hist['close'][-g.p1:].mean()
    ma60 = hist['close'].mean()

    if ma5 > ma60 and g.security not in context.positions:
        order_value(g.security, context.cash)
    elif ma5 < ma60 and g.security in context.positions:
        order_target(g.security, 0)     # 全部卖出

run()