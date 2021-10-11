# coding: utf-8
import sys, os
from czsc.analyze import *
from czsc.utils.echarts_plot import *
import zipfile
from tqdm import tqdm
from czsc.analyze import *
from czsc.enum import Freq, Operate
from czsc import signals
from czsc.signals import get_default_signals, get_s_three_bi
from czsc.objects import Event, Factor, Signal

cur_path = os.path.split(os.path.realpath(__file__))[0]

def test_czsc_update():
    file_kline = os.path.join(cur_path, "data/sanqi.csv")
    kline = pd.read_csv(file_kline, encoding="utf-8")
    kline.loc[:, "dt"] = pd.to_datetime(kline.dt)
    bars = [RawBar(symbol='601318.XSHG', id=i, freq=Freq.D, open=row['open'], dt=row['dt'],
                   close=row['close'], high=row['high'], low=row['low'], vol=row['vol'])
            for i, row in kline.iterrows()]

    # 不计算任何信号
    c = CZSC(bars, max_bi_count=50)
    num = 1
    for i in c.bi_list:
        print(num, i.vol, i.change, i.power, i.length, round(i.vol/i.length, 4), round(i.power/i.length,4) )
        num += 1
    print(num)
    assert not c.signals

    # 计算信号
    # c = CZSC(bars, max_bi_count=50, get_signals=get_default_signals)
    # assert isinstance(c.signals, OrderedDict) and len(c.signals) == 38

    # 测试自定义信号
    # c = CZSC(bars, max_bi_count=50, get_signals=get_user_signals)
    # assert len(c.signals) == 10

    kline = [x.__dict__ for x in c.bars_raw]
    bi = [{'dt': x.fx_a.dt, "bi": x.fx_a.fx} for x in c.bi_list] + \
         [{'dt': c.bi_list[-1].fx_b.dt, "bi": c.bi_list[-1].fx_b.fx}]
    chart = kline_pro(kline, bi=bi, title="{} - {}".format(c.symbol, c.freq))
    file_html = "sanqi.html"
    chart.render(file_html)
    # os.remove(file_html)

test_czsc_update()
