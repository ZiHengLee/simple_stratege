# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from sklearn.linear_model import LinearRegression
import multiprocessing
import numpy as np
import pandas as pd
from gm.api import *
# https://bbs.myquant.cn/topic/2418
'''
本策略计算优加换手率因子，通过10分组回测筛选有效因子；
10分组回测基本思想：设定所需优化的参数数值范围及步长，将参数数值循环输入进策略，进行遍历回测，
                 记录每次回测结果和参数，根据某种规则将回测结果排序，找到最好的参数。
1、定义策略函数
2、多进程循环输入参数数值
3、获取回测报告，生成DataFrame格式
4、排序
'''


# 原策略中的参数定义语句需要删除！
def init(context):
    # 每月的第一个交易日的09:40:00执行策略algo
    schedule(schedule_func=algo, date_rule='1m', time_rule='9:40:00')
    # 设置交易标的
    context.symbol = None
    # 设置买入股票资金比例
    context.ratio = 0.5


def MAD(data, N):
    """
    ---N倍中位数去极值---
    1 求所有因子的中位数 median
    2 求每个因子与中位数的绝对偏差值，求得到绝对偏差值的中位数 new_median
    3 根据参数 N 确定合理的范围为 [median−N*new_median,median+N*new_median]，并针对超出合理范围的因子值做调整
    """
    median = data.quantile(0.5)
    new_median = abs(data - median).quantile(0.5)
    # 定义N倍的中位数上下限
    high = median + N * new_median
    low = median - N * new_median
    # 替换上下限
    data = np.where(data > high, high, data)
    data = np.where(data < low, low, data)
    return data


def Standardize(data):
    """
    ---数据标准化---
    标准化后的数值 = (原始值 - 原始值均值) / 原始值标准差
    """
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def LR(x, y):
    '''
    ---建立回归方程---
    '''
    lr = LinearRegression()
    # 拟合
    lr.fit(x, y)
    # 预测
    y_predict = lr.predict(x)
    # 求残差
    data = y - y_predict
    return data


def algo(context):
    # 当前时间
    now_date = context.now.strftime('%Y-%m-%d')

    # 获取上一个交易日日期，即为上个月月底的日期
    last_date = get_previous_trading_date(exchange='SHSE', date=now_date)

    # 获取沪深300成分股
    stocks = get_history_constituents(index='SHSE.000300', start_date=last_date, end_date=last_date) \
        [0]['constituents'].keys()

    # 获取沪深300成分股过去20个交易日的换手率
    fundamentals = get_fundamentals_n(table='trading_derivative_indicator', symbols=stocks,
                                      end_date=last_date, count=20, fields='TURNRATE, NEGOTIABLEMV', df=True)

    # 对股票进行分组，计算每只股票过去20个交易日的换手率的均值、市值均值
    fundamental = fundamentals.groupby('symbol').agg('mean')
    fundamental.columns = ['turn_mean', 'pe_mean']

    # 对股票进行分组，计算每只股票过去20个交易日的换手率的标准差
    fundamental['turn_std'] = fundamentals.groupby('symbol').agg(np.std)['TURNRATE']

    # 计算量小换手率因子：对每只股票过去20个交易日的换手率的均值，做市值中性化处理
    # 提取回归数据 x1: 市值, y1: 换手率均值
    fundamental['turn_mean'] = MAD(fundamental['turn_mean'], 3)
    fundamental['turn_mean'] = Standardize(fundamental['turn_mean'])
    x1 = fundamental['pe_mean'].values.reshape(-1, 1)
    y1 = fundamental['turn_mean']
    # 市值中性化处理得出量小换手率因子
    fundamental['turn_20'] = LR(x1, y1)

    # 计算量稳换手率因子：对每只股票过去20个交易日的换手率的标准差，做市值中性化处理
    # 提取回归数据 x2: 市值, y2: 换手率标准差
    fundamental['turn_std'] = MAD(fundamental['turn_std'], 3)
    fundamental['turn_std'] = Standardize(fundamental['turn_std'])
    x2 = fundamental['pe_mean'].values.reshape(-1, 1)
    y2 = fundamental['turn_std']
    # 市值中性化处理得出量稳换手率因子
    fundamental['str'] = LR(x2, y2)

    # 计算优加换手率：最终得分，即为优加换手率因子的因子值
    # 先将所有样本按照量稳因子从小到大排序，打分 1,2,……,N-1,N，N 为当期样本数量，记为“得分 1”；
    fundamental.sort_values(by='str', inplace=True)
    fundamental['score_1'] = range(1, len(fundamental) + 1)

    # 取量稳因子排名靠前的50%样本，再将它们按照量小因子从大到小排序，打分 1,2,…,N/2，记为“得分 2”；
    # 这些股票的最终得分为 “得分 1”+“得分 2”
    fund_1 = fundamental.iloc[:int(len(fundamental) / 2)]
    fund1 = fund_1.copy()
    fund1.sort_values(by='turn_20', ascending=False, inplace=True)
    fund1['score_2'] = range(1, len(fund1) + 1)
    fund1['scores'] = fund1['score_1'] + fund1['score_2']

    # 量稳因子排名靠后的50%样本，则将它们按照量小因子从小到大排序，打分 1,2,…,N/2，记为“得分 3”；
    # 这些股票的最终得分为 “得分 1”+“得分 3”
    fund_2 = fundamental.iloc[int(len(fundamental) / 2):]
    fund2 = fund_2.copy()
    fund2.sort_values(by='turn_20', inplace=True)
    fund2['score_3'] = range(1, len(fund2) + 1)
    fund2['scores'] = fund2['score_1'] + fund2['score_3']

    # 合并fund1、fund2，按照最终得分从小到大排序
    fundamental = pd.concat([fund1, fund2], join='inner')
    fundamental.sort_values(by='scores', inplace=True)

    # 进行10分组回测：第N次回测，对第N组的优加换手率进行回测，其中，对第N组中10%优加换手率较小的股票进行买入
    data = fundamental.iloc[int((context.num - 1) * 0.1 * len(fundamental)): int(context.num * 0.1 * len(fundamental))]
    symbols = data.index[0:int(0.1 * len(data))].to_list()

    # 获取持仓
    positions = context.account().positions()
    # 若有持仓，平不在股票池的仓位，平仓成功则买入股票池中的股票
    for position in positions:
        symbol = position['symbol']
        if symbol not in symbols:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print('市价单平不在股票池的仓位', symbol)

    # 计算每只股票买入比例
    percent = context.ratio / len(symbols)
    # 将股票池中的股票持仓调整至percent
    for symbol in symbols:
        order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print('以市价单调仓至买入比例', symbol)


# 获取每次回测的报告数据
def on_backtest_finished(context, indicator):
    data = [indicator['pnl_ratio'], indicator['pnl_ratio_annual'], indicator['sharp_ratio'], indicator['max_drawdown'],
            context.num]
    # 将超参加入context.result
    context.result.append(data)


def run_strategy(num):
    from gm.model.storage import context
    # 用context传入回测次数参数
    context.num = num
    # context.result用以存储超参
    context.result = []
    '''
        strategy_id策略ID,由系统生成
        filename文件名,请与本文件名保持一致
        mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID,可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='01f91301-21ca-11ec-b608-001c42b9c3b1',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='367168d238b9de2c6777024b7c198f531f71fc38',
        backtest_start_time='2019-01-01 08:00:00',
        backtest_end_time='2020-12-31 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.0001)
    return context.result


if __name__ == '__main__':
    # 循环输入参数数值回测
    num = [[i] for i in range(1, 11)]

    a_list = []
    pool = multiprocessing.Pool(processes=10, maxtasksperchild=1)  # create 10 processes
    for i in range(len(num)):
        a_list.append(pool.apply_async(func=run_strategy, args=(num[i][0],)))
    pool.close()
    pool.join()
    info = []
    for pro in a_list:
        print('pro', pro.get()[0])
        info.append(pro.get()[0])
    print(info)
    info = pd.DataFrame(np.array(info), columns=['pnl_ratio', 'pnl_ratio_annual', 'sharp_ratio', 'max_drawdown', 'num'])
    info.to_csv('info.csv', index=False)