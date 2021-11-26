#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2X Data generation: Generate ground truth for stock markets based on OHLC data
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
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from scipy.ndimage.interpolation import shift
from pandas.plotting import register_matplotlib_converters

from filepaths import Paths

register_matplotlib_converters()

# Own modules
import utils.data_handling_support_functions as sup
import utils.custom_methods as custom
import utils.data_visualization_functions as vis
import utils.stock_market_utils as stock

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'ISC'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Step 2.0 - Generate features and outcomes from raw data')
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


# def generate_custom_class_labels():
#     '''
#
#
#     '''
#
#     ## Class Generation
#     #Here, 4 classes will be generated:
#     #- LongTerm
#     #- Intermediate term 20d
#     #- Short term 5d
#     #- very short term 1 d
#
#     #### Create labels
#
#     y_labels = {
#         'neutral': 0,
#         'positive': 1,
#         'negative': 2
#     }
#
#     return y_labels

# def find_tops_bottoms(source):
#     ### Calculate Tops and Bottoms
#
#     m = source.shape[0]
#     factor = 10000000
#     topsTemp = np.zeros([m, 4]);
#     #topsTemp
#     bottomsTemp = np.ones([m, 4]) * factor;
#     #bottomsTemp
#     # close=source['Close']
#     # close
#
#     # Get tops and bottoms from the chart
#     # Parameter
#     maxDecline = 0.02
#     maxIncrease = 0.02
#     factor = 10000000
#
#     # Format: Time, High, Low, Close
#     m = source.shape[0]
#
#     topsTemp = np.zeros([m, 4])
#     bottomsTemp = np.ones([m, 4]) * factor
#
#     high = source['High']
#     low = source['Low']
#     close = source['Close']
#
#     # Run 1 for the rough tops and bottoms
#     for i, data in enumerate(source.values):
#         # Get top
#         if i > 3 and i < m - 3:
#             # Decline close >2% from top high
#             decline = (high[i] - min(close[i + 1:i + 2])) / high[i];
#             if decline > maxDecline or high[i] == max(high[i - 3:i + 3]):
#                 # Top found
#                 topsTemp[i, 1] = high[i];
#                 # print("Top found at i={} value={}".format(i, high[i]));
#
#         # %Get bottom
#         if i > 3 and i < m - 3:
#             #    %Decline close >2% from top high
#             increase = (low[i] - max(close[i + 1:i + 2])) / low[i];
#             if increase > maxIncrease or low[i] == min(low[i - 3:i + 3]):
#                 # Top found
#                 bottomsTemp[i, 1] = low[i];
#                 # print("Bottom found at i={} value={}".format(i, low[i]));
#
#     print("{} tops, {} bottoms found.".format(sum(topsTemp[:, 1] > 0), sum(bottomsTemp[:, 1] < factor)));
#
#     # %Run 2 for exacter tops and bottoms
#     iTop = topsTemp[:, 1];
#     iBottom = bottomsTemp[:, 1];
#     for i, data in enumerate(source.values):
#         # Tops
#         if i > 20 and i < m - 20:
#             if iTop[i] > 0 and max(iTop[i - 15:i + 15]) <= iTop[i]:
#                 topsTemp[i, 2] = iTop[i];
#                 # %fprintf("Intermediate top found at i=%i value=%.0f\n", i, iTop(i));
#
#             if iBottom[i] < factor and min(iBottom[i - 15:i + 15]) >= iBottom[i]:
#                 bottomsTemp[i, 2] = iBottom[i];
#                 # %fprintf("Intermediate bottom found at i=%i value=%.0f\n", i, iBottom(i));
#
#     bottomsTemp[bottomsTemp == factor] = 0
#     bottoms = bottomsTemp[:, 2]
#     tops = topsTemp[:, 2]
#     print("Reduced to {} tops and {} bottoms.".format(sum(tops[:] > 0), sum(bottoms[:] > 0)));
#
#     # topsTemp[topsTemp[:,1]>0]
#
#     # bottomsTemp[0:10,:]
#
#     plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
#     plt.plot(source['Date'], source['Close'])
#     plt.plot(source['Date'], tops[:])
#     plt.plot(source['Date'], bottoms[:])
#     plt.title("OMXS30 Tops and Bottoms")
#     fig_tops_bot = plt.gcf()
#     #plt.show(block = False)
#
#     latestBottoms = calculateLatestEvent(bottoms)
#     latestTops = calculateLatestEvent(tops)
#
#     plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
#     plt.plot(source['Date'], source['Close'])
#     plt.plot(source['Date'], latestTops[:])
#     plt.plot(source['Date'], latestBottoms[:])
#     plt.title("OMXS30 Latest Tops and Bottoms")
#     fig_latest_tops_latest_bot = plt.gcf()
#     #plt.show(block = False)
#
#     return bottoms, tops, latestTops, latestBottoms, fig_tops_bot, fig_latest_tops_latest_bot



# def calculateLatestEvent(eventList):
#     '''
#     # Calculate the latest single event from a list of [0 0 0 0 2 0 0 1 0]->[0 0 0 0 2 2 2 2 1 1]
#
#
#     '''
#     previousItem = 0;
#     result = np.zeros(eventList.shape[0])
#     for i in range(len(eventList)):
#         if eventList[i] != previousItem and eventList[i] != 0:
#             result[i] = eventList[i]
#             previousItem = eventList[i]
#         else:
#             result[i] = previousItem
#     return result


# def MA(mov, n, shift):
#     '''
#     # Calculate varios MA
#     # mov = pd.Series(np.arange(0, 100, 1), name='test')
#     # print(mov)
#     # Moving Average
#     # mov: close
#     # n: Number of samples
#     # shift: shift of the window. shift < 0 future, shift > 0 history
#
#     '''
#     MA = mov.rolling(n).mean()
#     # print(MA)
#     source = pd.DataFrame(MA)
#     source.columns = ['SMA' + str(n) + 'shift' + str(shift)]
#     shiftedMA = source.shift(shift)
#
#     return shiftedMA


# def adder(Data, times):
#     for i in range(1, times + 1):
#         z = np.zeros((len(Data), 1), dtype=float)
#         Data = np.append(Data, z, axis=1)
#
#
#     return Data

# def fractal_indicator(Data, high, low, ema_lookback, min_max_lookback, where):
#     '''
#     A fractal indicator that gives values around 1 if the trend is going to change
#
#     Source: https://medium.com/swlh/the-fractal-indicator-detecting-tops-bottoms-in-markets-1d8aac0269e8
# 
#
#     '''
#
#     Data = ema(Data, 2, ema_lookback, high, where)
#     Data = ema(Data, 2, ema_lookback, low, where + 1)
#
#     Data = volatility(Data, ema_lookback, high, where + 2)
#     Data = volatility(Data, ema_lookback, low, where + 3)
#
#     Data[:, where + 4] = Data[:, high] - Data[:, where]
#     Data[:, where + 5] = Data[:, low] - Data[:, where + 1]
#
#
#     for i in range(len(Data)):
#         try:
#             Data[i, where + 6] = max(Data[i - min_max_lookback + 1:i + 1, where + 4])
#
#         except ValueError:
#             pass
#
#     for i in range(len(Data)):
#         try:
#             Data[i, where + 7] = min(Data[i - min_max_lookback + 1:i + 1, where + 5])
#
#         except ValueError:
#             pass
#
#     Data[:, where + 8] = (Data[:, where + 2] + Data[:, where + 3]) / 2
#     Data[:, where + 9] = (Data[:, where + 6] - Data[:, where + 7]) / Data[:, where + 8]
#
#     return Data

def calculate_y_signals(source, bottoms, tops, latestBottoms, latestTops, pos_trend_long, pos_trend_short):
    '''
    Calculate the Y values for 1d, 5d, 20d and the Long Trend

    '''


    # %%

    # Calculte the 1d trend

    high = source['High']
    low = source['Low']
    close = source['Close']
    m = source.shape[0]

    y1day = np.zeros(m)
    # Calculate the 5d trend
    y5day = np.zeros(m)
    # 20d trend
    y20day = np.zeros(m)
    # long term trend
    ylong = np.zeros(m)
    signalLong = 0;

    for i in range(m - 50):
        # === 1d trend ===#
        if close[i + 1] > close[i]:
            y1day[i] = 1

        # === 5day short trend ===#
        # if (pos_trend_short[i+10]==True) and (pos_trend_short[i+1]==True) and (pos_trend_short[i+2]==True) and (future_difference>0.001) and close[i+1]>close[i]:
        # Positive buy
        if i > 5 and np.max(bottoms[i - 5:i - 1]) > 0 and np.mean(close[i + 1:i + 5]) > close[i]:
            y5day[i] = 1;

        # negtive, sell
        if i > 5 and np.max(tops[i - 5:i - 1]) > 0 and np.mean(close[i + 1:i + 5]) < close[i]:
            y5day[i] = 2;

        # === median trend 20d ===#
        if close[i + 20] > close[i]:
            y20day[i] = 1;

        # === long term trend ===#
        # Trigger positive, buy
        if pos_trend_long[i] == True and close[i] > latestTops[i]:
            signalLong = 1;
        # negative, sell
        elif pos_trend_long[i] == False and close[i] < latestBottoms[i]:
            signalLong = 2;

        if signalLong == 1:
            ylong[i] = 1;
        elif signalLong == 2:
            ylong[i] = 2;
        else:
            ylong[i] = 0;

        # === end ===#
    print("y1day", sum(y1day))
    print("y5day", sum(y5day))
    print("y20day", sum(y20day))
    print("ylong", sum(ylong))
    print("Generated trends 1d, 5d, 20d, long.")

    return y1day, y5day, y20day, ylong

def clean_bad_signals_1(y1day, y5day, y20day, ylong, close, latestBottoms, latestTops):
    '''
    # Clean bad signals 1

    '''

    previousSignalCount = sum(y1day)
    for i in range(y1day.shape[0] - 50):
        # If the signal is only valid for one or 2 days the signal was bad and
        # noisy. Only if the signal is valid for 3 days, it can be consideres as
        # a real signal
        if np.mean(y1day[i:i + 3]) < 0.75:
            y1day[i] = 0

    print("Previous signal count y1day={}. New signal count={}".format(previousSignalCount, sum(y1day)))
    print("Cleaned bad signals 1");

    return y1day, y5day, y20day, ylong

def clean_bad_signals_2(y1day, y5day, y20day, ylong, close, latestBottoms, latestTops):
    '''
    # Clean bad signals 2, filter single days, enhance trend

    '''

    print("signals y1day=", sum(y1day))
    print("signals ylong=", sum(ylong))

    # for i in range(m-50):
    # short term +1d
    # if i>1 and y1day[i-1]==0 and y1day[i+1]==0:
    #    y1day[i]=0;

    # long term, remove all values < 5 days to remove noise
    # use sliding window
    # if i>5 and ylong[i]==1:
    #    slideresult = np.zeros(5);
    #    for j in range(-5,0):
    #        slideresult[j+5] = np.mean(ylong[i+j:i+j+4])
    #
    #    if max(slideresult)<1:
    #        ylong[i]=0;

    print("signals y1day=", sum(y1day))
    print("signals ylong=", sum(ylong))
    print("Cleaned bad signals 2");

    return y1day, y5day, y20day, ylong

def clean_bad_signals_3(y1day, y5day, y20day, ylong, close, latestBottoms, latestTops):
    '''
    # Clean bad signals 3, filter single days

    '''

    print("signals ylong=", sum(ylong))
    for i in range(ylong.shape[0] - 50):
        # long term, fill in all values < 5 days to remove noise
        # Fill gaps
        # Use sliding window
        if i > 20 - 1 and ylong[i] == 0 and np.mean(ylong[i - 20:i + 20]) > 0.5:
            ylong[i] = 1;

        # Enhance the trend to run as far as possible
        if i >= 1 and ylong[i - 1] == 1 and ylong[i] == 0 and close[i] > latestBottoms[i]:
            ylong[i] = 1;

    print("signals ylong=", sum(ylong))
    print("Cleaned bad signals 3.");

    return y1day, y5day, y20day, ylong

def define_tops_bottoms(bottoms, tops):
    '''
    Merge tops and bottoms

    '''

    print("Merge tops and bottoms. Tops=1, Bottoms=2")
    col = (bottoms > 0) * 2 + (tops > 0) * 1
    return col


def generate_features_outcomes(image_save_directory, source):
    '''


    '''


    # Outcome and Feature Construction
    #Generate the class values, i.e.the y for the data.Construct features. The following dataframes are
    #generated:
    #- source
    #- features
    #- outcomes

    #Load only a subset of the whole raw data to create a debug dataset
    #source = custom(conf['source_path']).iloc[0:1000, :]

    #Plot source
    #plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    #plt.plot(source['Date'], source['Close'])
    #plt.title(conf['source_path'])
    #plt.show()

    bottoms, tops, latestTops, latestBottoms, fig_tops_bot, fig_latest_tops_latest_bot = stock.find_tops_bottoms(source)
    vis.save_figure(fig_tops_bot, image_save_directory=image_save_directory, filename=str(fig_tops_bot.axes[0].get_title()).replace(' ', '_'))
    vis.save_figure(fig_latest_tops_latest_bot, image_save_directory=image_save_directory, filename=str(fig_latest_tops_latest_bot.axes[0].get_title()).replace(' ', '_'))

    topsBottoms = define_tops_bottoms(bottoms, tops)

    pos_trend_long, fig_long = stock.calculate_lowess(source, 300)
    plt.gca()

    vis.save_figure(fig_long, image_save_directory=image_save_directory, filename=str(fig_long.axes[0].get_title()).replace(' ', '_'))
    #plt.show(block = False)

    pos_trend_short, fig_short = stock.calculate_lowess(source, 10)
    plt.gca()
    #plt.show(block = False)
    vis.save_figure(fig_short, image_save_directory=image_save_directory, filename=str(fig_short.axes[0].get_title()).replace(' ', '_'))

    y1day, y5day, y20day, ylong = calculate_y_signals(source, bottoms, tops, latestBottoms, latestTops, pos_trend_long, pos_trend_short)
    y1day, y5day, y20day, ylong = clean_bad_signals_1(y1day, y5day, y20day, ylong, source['Close'], latestBottoms, latestTops)
    y1day, y5day, y20day, ylong = clean_bad_signals_2(y1day, y5day, y20day, ylong, source['Close'], latestBottoms, latestTops)
    y1day, y5day, y20day, ylong = clean_bad_signals_3(y1day, y5day, y20day, ylong, source['Close'], latestBottoms, latestTops)

    # Merge all y values to the series start
    outcomes = pd.DataFrame(index=source.index).join(
        pd.Series(y1day, name="1dTrend").astype('int64')).join(
        pd.Series(y5day, name="5dTrend").astype('int64')).join(
        pd.Series(y20day, name="20dTrend").astype('int64')).join(
        pd.Series(ylong, name="LongTrend").astype('int64')).join(
        pd.Series(topsBottoms, name="TopsBottoms").astype('int64'))

    return outcomes


def main(config_path):
    conf = sup.load_config(config_path)
    # Load annotations file
    y_labels = pd.read_csv(conf['Paths'].get('source_path'), sep=';', header=None).set_index(0).to_dict()[1]

    # Generating filenames for saving the files
    image_save_directory = os.path.join(conf['Paths'].get('results_directory'), "data_generation")
    outcomes_filename_raw = os.path.join(conf['Paths'].get('prepared_data_directory'), "temp", "temp_outcomes_uncut" + ".csv")

    #if os.path.isdir(conf['Paths'].get('prepared_data_directory'))==False:
    os.makedirs(os.path.dirname(outcomes_filename_raw), exist_ok=True)
    #    print("Created directory ", conf['Paths'].get('training_data_directory'))

    #if os.path.isdir(conf['Paths'].get('result_directory'))==False:
    #os.makedirs(conf['Paths'].get('result_directory'), exist_ok=True)
    #    print("Created directory ", conf['Paths'].get('result_directory'))

    #Load only a subset of the whole raw data to create a debug dataset
    source = custom.load_source(conf['Paths'].get('source_path')) #.iloc[0:1000, :]

    #Plot source
    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(source['Date'], source['Close'])
    plt.title(conf['Paths'].get('source_path'))
    #plt.show(block = False)

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename="Source_data")

    #y_labels = annotations #generate_custom_class_labels()
    outcomes = generate_features_outcomes(image_save_directory, source)

    # Drop the 50 last values as they cannot be used for prediction as +50 days ahead is predicted
    source_cut = source.drop(source.tail(50).index, inplace=False)
    outcomes_cut = outcomes.drop(outcomes.tail(50).index, inplace=False)

    vis.plot_three_class_graph(outcomes_cut['1dTrend'].values,
                               source_cut['Close'], source_cut['Date'],
                               0,0,0, ('close', 'neutral', 'positive', 'negative'),
                               title=conf['Common'].get('dataset_name') + '_GT_1dTrend',
                               save_fig_prefix=image_save_directory)

    vis.plot_three_class_graph(outcomes_cut['5dTrend'].values,
                               source_cut['Close'], source_cut['Date'],
                               0,0,0, ('close', 'neutral', 'positive', 'negative'),
                               title=conf['Common'].get('dataset_name') + '_GT_5dTrend',
                               save_fig_prefix=image_save_directory)

    vis.plot_three_class_graph(outcomes_cut['20dTrend'].values,
                               source_cut['Close'], source_cut['Date'],
                               0,0,0, ('close', 'neutral', 'positive', 'negative'),
                               title=conf['Common'].get('dataset_name') + '_GT_20dTrend',
                               save_fig_prefix=image_save_directory)

    vis.plot_three_class_graph(outcomes_cut['LongTrend'].values,
                               source_cut['Close'], source_cut['Date'],
                               0,0,0, ('close', 'neutral', 'positive', 'negative'),
                               title=conf['Common'].get('dataset_name') + '_GT_LongTrend',
                               save_fig_prefix=image_save_directory)

    vis.plot_three_class_graph(outcomes_cut['TopsBottoms'].values,
                               source_cut['Close'], source_cut['Date'],
                               0,0,0, ('close', 'neutral', 'top', 'bottom'),
                               title=conf['Common'].get('dataset_name') + '_GT_TopsBottoms',
                               save_fig_prefix=image_save_directory)

    def binarize(outcomes, class_number):
        return (outcomes == class_number).astype(int)

    vis.plot_two_class_graph(binarize(outcomes_cut['1dTrend'], conf['Common'].getint('class_number')),
                             source_cut['Close'], source_cut['Date'],
                             0,
                             ('close', 'Positive Trend'),
                             title=conf['Common'].get('dataset_name') + '_GT_1dTrend',
                             save_fig_prefix=image_save_directory)

    vis.plot_two_class_graph(binarize(outcomes_cut['5dTrend'], conf['Common'].getint('class_number')),
                             source_cut['Close'], source_cut['Date'],
                             0,
                             ('close', 'Positive Trend'),
                             title=conf['Common'].get('dataset_name') + '_GT_5dTrend',
                             save_fig_prefix=image_save_directory)

    vis.plot_two_class_graph(binarize(outcomes_cut['20dTrend'], conf['Common'].getint('class_number')),
                             source_cut['Close'], source_cut['Date'],
                             0,
                             ('close', 'Positive Trend'),
                             title=conf['Common'].get('dataset_name') + '_GT_20dTrend',
                             save_fig_prefix=image_save_directory)

    vis.plot_two_class_graph(binarize(outcomes_cut['LongTrend'], conf['Common'].getint('class_number')),
                             source_cut['Close'], source_cut['Date'],
                             0,
                             ('close', 'Positive Trend'),
                             title=conf['Common'].get('dataset_name') + '_GT_LongTrend',
                             save_fig_prefix=image_save_directory)

    # Save file
    # Save outcomes to a csv file
    print("Outcomes shape {}".format(outcomes_cut.shape))
    outcomes_cut.to_csv(outcomes_filename_raw, sep=';', index=True, header=True)
    print("Saved outcomes to " + outcomes_filename_raw)


if __name__ == "__main__":
    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    main(args.config_path)


    print("=== Program end ===")