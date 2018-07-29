#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:18:53 2018

@author: vmueller
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates




df = pd.read_csv('A/A_Price.csv', parse_dates=True, index_col=0)
#df['Volume'] = df.apply(lambda x: int("".join(x['Volume'].split(','))), axis=1)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
#print(df_ohlc.head())

df_ohlc = df_ohlc.reset_index()
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
fig = plt.figure()
fig.suptitle('Agilent Technologies (A) - CloneMind Stock Analysis \n https://goo.gl/snRfhq', fontsize=20)
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
plt.show()
