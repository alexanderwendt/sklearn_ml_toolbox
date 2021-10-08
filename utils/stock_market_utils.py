#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utils for stock market price calculations

License_info: ISC
ISC License

Copyright (c) 2020, Alexander Wendt

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

# Futures

# Built-in/Generic Imports

# Libs
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from scipy.ndimage.interpolation import shift
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Own modules
#import data_handling_support_functions as sup
#import custom_methods as custom
#import data_visualization_functions as vis

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

def find_tops_bottoms(source):
    ### Calculate Tops and Bottoms

    m = source.shape[0]
    factor = 10000000
    topsTemp = np.zeros([m, 4]);
    #topsTemp
    bottomsTemp = np.ones([m, 4]) * factor;
    #bottomsTemp
    # close=source['Close']
    # close

    # Get tops and bottoms from the chart
    # Parameter
    maxDecline = 0.02
    maxIncrease = 0.02
    factor = 10000000

    # Format: Time, High, Low, Close
    m = source.shape[0]

    topsTemp = np.zeros([m, 4])
    bottomsTemp = np.ones([m, 4]) * factor

    high = source['High']
    low = source['Low']
    close = source['Close']

    # Run 1 for the rough tops and bottoms
    for i, data in enumerate(source.values):
        # Get top
        if i > 3 and i < m - 3:
            # Decline close >2% from top high
            decline = (high[i] - min(close[i + 1:i + 2])) / high[i];
            if decline > maxDecline or high[i] == max(high[i - 3:i + 3]):
                # Top found
                topsTemp[i, 1] = high[i];
                # print("Top found at i={} value={}".format(i, high[i]));

        # %Get bottom
        if i > 3 and i < m - 3:
            #    %Decline close >2% from top high
            increase = (low[i] - max(close[i + 1:i + 2])) / low[i];
            if increase > maxIncrease or low[i] == min(low[i - 3:i + 3]):
                # Top found
                bottomsTemp[i, 1] = low[i];
                # print("Bottom found at i={} value={}".format(i, low[i]));

    print("{} tops, {} bottoms found.".format(sum(topsTemp[:, 1] > 0), sum(bottomsTemp[:, 1] < factor)));

    # %Run 2 for exacter tops and bottoms
    iTop = topsTemp[:, 1];
    iBottom = bottomsTemp[:, 1];
    for i, data in enumerate(source.values):
        # Tops
        if i > 20 and i < m - 20:
            if iTop[i] > 0 and max(iTop[i - 15:i + 15]) <= iTop[i]:
                topsTemp[i, 2] = iTop[i];
                # %fprintf("Intermediate top found at i=%i value=%.0f\n", i, iTop(i));

            if iBottom[i] < factor and min(iBottom[i - 15:i + 15]) >= iBottom[i]:
                bottomsTemp[i, 2] = iBottom[i];
                # %fprintf("Intermediate bottom found at i=%i value=%.0f\n", i, iBottom(i));

    bottomsTemp[bottomsTemp == factor] = 0
    bottoms = bottomsTemp[:, 2]
    tops = topsTemp[:, 2]
    print("Reduced to {} tops and {} bottoms.".format(sum(tops[:] > 0), sum(bottoms[:] > 0)));

    # topsTemp[topsTemp[:,1]>0]

    # bottomsTemp[0:10,:]

    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(source['Date'], source['Close'])
    plt.plot(source['Date'], tops[:])
    plt.plot(source['Date'], bottoms[:])
    plt.title("OMXS30 Tops and Bottoms")
    fig_tops_bot = plt.gcf()
    #plt.show(block = False)

    latestBottoms = calculateLatestEvent(bottoms)
    latestTops = calculateLatestEvent(tops)

    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(source['Date'], source['Close'])
    plt.plot(source['Date'], latestTops[:])
    plt.plot(source['Date'], latestBottoms[:])
    plt.title("OMXS30 Latest Tops and Bottoms")
    fig_latest_tops_latest_bot = plt.gcf()
    #plt.show(block = False)

    return bottoms, tops, latestTops, latestBottoms, fig_tops_bot, fig_latest_tops_latest_bot

def calculateLatestEvent(eventList):
    '''
    # Calculate the latest single event from a list of [0 0 0 0 2 0 0 1 0]->[0 0 0 0 2 2 2 2 1 1]


    '''
    previousItem = 0;
    result = np.zeros(eventList.shape[0])
    for i in range(len(eventList)):
        if eventList[i] != previousItem and eventList[i] != 0:
            result[i] = eventList[i]
            previousItem = eventList[i]
        else:
            result[i] = previousItem
    return result

def calculate_lowess(source, days_to_consider):
    '''
    # Fraction for the lowess smoothing function


    '''
    if sum(np.isnan(source['Close']))>0:
        raise Exception("Notice: If there are any NaN in the data, these rows are removed. It causes a dimension problem.")
        #print("Notice: If there are any NaN in the data, these rows are removed. It causes a dimension problem.")

    frac = days_to_consider / len(source['Close'])
    filtered = lowess(source['Close'], source['Date'], frac=frac)
    # Calculate the dlowess/dt to see if it is raising or declining
    shiftCol = filtered[:, 1] - shift(filtered[:, 1], 1, cval=np.NaN)
    pos_trend = shiftCol > 0
    # print(pos_trend[0:5])

    fig = plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(source['Date'], source['Close'])
    plt.plot(source['Date'], filtered[:, 1], 'r-', linewidth=3)
    plt.plot(source['Date'], filtered[:, 1] * pos_trend, 'g-', linewidth=3)
    plt.title("Lowess_Days_" + str(days_to_consider))
    # plt.plot(source['Date'], filtered[:, 1]*pos_trend_cleaned, 'y-', linewidth=3)

    return pos_trend, fig