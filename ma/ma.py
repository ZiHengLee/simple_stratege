# -*- coding: utf-8 -*-
from gm.api import *
from datetime import datetime
from datetime import timedelta
import talib
import numpy as np
from collections import deque

# 常用参量设置
DATE_STR = "%Y-%m-%d"
TIME_STR = "%Y-%m-%d %H:%M:%S"

HIST_WINDOW = 40
SHORT_PERIOD = 13
LONG_PERIOD = 35


def init(context):
    # 全局变量设置
    context.dict_stock_price = dict()
    # 以 50 EFT作为交易标的
    context.stock_pool = ['SHSE.600000']
    # 订阅日线行情
    subscribe(symbols=context.stock_pool, frequency='1d', wait_group=True)
    # 日期设定，避免出现未来函数，将起始日往前取一日
    start_date = datetime.strptime(context.backtest_start_time, TIME_STR)
    context.start_date = datetime.strftime(start_date - timedelta(days=1),
                                           TIME_STR)
    # 获取起始日之前行情，便于计算指标
    deque_close = deque(maxlen=HIST_WINDOW)
    for stock in context.stock_pool:
        history_info = history_n(symbol=stock,
                                 frequency='1d',
                                 count=HIST_WINDOW,
                                 adjust=ADJUST_PREV,
                                 adjust_end_time=context.backtest_end_time,
                                 end_time=context.start_date,
                                 fields='close')
        for bar in history_info:
            deque_close.append(bar['close'])
        context.dict_stock_price.setdefault(stock, deque_close)
    print('finish initialization')


def on_bar(context, bars):
    limit = 0
    for bar in bars:
        if bar.symbol not in context.dict_stock_price.keys():
            print('Warning: cannot obtain price of stock {} at date {}'.format(
                bar.symbol, context.now))
        # 数据填充
        context.dict_stock_price[bar.symbol].append(bar.close)
        # 计算指标，这里以双均线为例
        closes = np.array(context.dict_stock_price[bar.symbol])
        short_ma = talib.SMA(closes, SHORT_PERIOD)
        long_ma = talib.SMA(closes, LONG_PERIOD)
        macd, macd_signal, macd_hist = talib.MACD(closes,
                                                  fastperiod=12,
                                                  slowperiod=26,
                                                  signalperiod=9)
        # 金叉，满仓买入
        if short_ma[-2] <= long_ma[-2] and short_ma[-1] > long_ma[-1]:
            order_percent(symbol=bar.symbol,
                          percent=1.0,
                          side=OrderSide_Buy,
                          order_type=OrderType_Market,
                          position_effect=PositionEffect_Open,
                          price=0)
            print(context.now, limit)
            limit = 0
        limit += 1
        print(limit)
        # 死叉或者 MACD 绿柱，全部卖出
        pos = context.account().position(symbol=bar.symbol, side=OrderSide_Buy)
        if limit == 10:
            print("fuckk")
            limit = 0
            if pos is None:
                continue
            order_volume(symbol=bar.symbol,
                         volume=pos.volume,
                         side=OrderSide_Sell,
                         order_type=OrderType_Market,
                         position_effect=PositionEffect_Close,
                         price=0)
            print(context.now+"sold")


if __name__ == "__main__":
    run(strategy_id='569b4ffc-6d44-11e8-bd88-80ce62334e41',
        filename='main.py',
        mode=MODE_BACKTEST,
        backtest_adjust=ADJUST_PREV,
        token='64c33fc82f334e11e1138eefea8ffc241db4a2a0',
        backtest_start_time='2017-01-17 09:00:00',
        backtest_end_time='2018-06-21 15:00:00')