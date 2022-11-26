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
import numpy as np
from pandas.plotting import register_matplotlib_converters

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
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)

args = parser.parse_args()

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
    signalLong = 0

    for i in range(m - 50):
        # === 1d trend ===#
        if close[i + 1] > close[i]:
            y1day[i] = 1

        # === 5day short trend ===#
        # if (pos_trend_short[i+10]==True) and (pos_trend_short[i+1]==True) and (pos_trend_short[i+2]==True) and (future_difference>0.001) and close[i+1]>close[i]:
        # Positive buy
        if i > 5 and np.max(bottoms[i - 5:i - 1]) > 0 and np.mean(close[i + 1:i + 5]) > close[i]:
            y5day[i] = 1

        # negtive, sell
        if i > 5 and np.max(tops[i - 5:i - 1]) > 0 and np.mean(close[i + 1:i + 5]) < close[i]:
            y5day[i] = 2

        # === median trend 20d ===#
        if close[i + 20] > close[i]:
            y20day[i] = 1

        # === long term trend ===#
        # Trigger positive, buy
        if pos_trend_long[i] == True and close[i] > latestTops[i]:
            signalLong = 1
        # negative, sell
        elif pos_trend_long[i] == False and close[i] < latestBottoms[i]:
            signalLong = 2

        if signalLong == 1:
            ylong[i] = 1
        elif signalLong == 2:
            ylong[i] = 2
        else:
            ylong[i] = 0

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
    print("Cleaned bad signals 1")

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
    print("Cleaned bad signals 2")

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
            ylong[i] = 1

        # Enhance the trend to run as far as possible
        if i >= 1 and ylong[i - 1] == 1 and ylong[i] == 0 and close[i] > latestBottoms[i]:
            ylong[i] = 1

    print("signals ylong=", sum(ylong))
    print("Cleaned bad signals 3.")

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

    bottoms, tops, latestTops, latestBottoms, fig_tops_bot, fig_latest_tops_latest_bot = stock.find_tops_bottoms(source)
    vis.save_figure(fig_tops_bot, image_save_directory=image_save_directory, filename=str(fig_tops_bot.axes[0].get_title()).replace(' ', '_'))
    vis.save_figure(fig_latest_tops_latest_bot, image_save_directory=image_save_directory, filename=str(fig_latest_tops_latest_bot.axes[0].get_title()).replace(' ', '_'))

    topsBottoms = define_tops_bottoms(bottoms, tops)

    pos_trend_long, fig_long = stock.calculate_lowess(source, 300)
    plt.gca()

    vis.save_figure(fig_long, image_save_directory=image_save_directory, filename=str(fig_long.axes[0].get_title()).replace(' ', '_'))

    pos_trend_short, fig_short = stock.calculate_lowess(source, 10)
    plt.gca()
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

    os.makedirs(os.path.dirname(outcomes_filename_raw), exist_ok=True)

    #Load only a subset of the whole raw data to create a debug dataset
    source = custom.load_source(conf['Paths'].get('source_path'))

    #Plot source
    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(source['Date'], source['Close'])
    plt.title(conf['Paths'].get('source_path'))

    vis.save_figure(plt.gcf(), image_save_directory=image_save_directory, filename="Source_data")

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
    main(args.config_path)


    print("=== Program end ===")