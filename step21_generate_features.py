#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2X Data generation: Generate features for stock markets based on OHLC data
License_info: TBD
"""

# Futures

# Built-in/Generic Imports

# Libs
import pandas_ta as ta

from math import ceil
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from scipy.ndimage.interpolation import shift
from pandas.plotting import register_matplotlib_converters

# Own modules
import data_visualization_functions as vis
import custom_methods as custom
import data_handling_support_functions as sup

__author__ = 'Alexander Wendt'
__copyright__ = 'Copyright 2020, Christian Doppler Laboratory for ' \
                'Embedded Machine Learning'
__credits__ = ['']
__license__ = 'TBD'
__version__ = '0.2.0'
__maintainer__ = 'Alexander Wendt'
__email__ = 'alexander.wendt@tuwien.ac.at'
__status__ = 'Experiental'

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)
register_matplotlib_converters()

parser = argparse.ArgumentParser(description='Step 2.1 - Generate features from raw data')
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument("-conf", '--config_path', default="config/debug_timedata_omxS30.ini",
                    help='Configuration file path', required=False)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


def generate_smoothed_trigger(values, alpha=0.5, tailclip=0.1):
    ''' From a value array with signals in the range -1, 0, 1, generate smoothed decay'''

    smoothed_sign_change = np.zeros(values.shape)
    for i, value in enumerate(values):
        previous_value = 0.0
        if i > 0:
            previous_value = smoothed_sign_change[i - 1]
        if np.isnan(value):
            value = 0

        # Now use expoential smoothing to smooth the values
        def exponential_smoothing(current_value, previous_value, alpha=alpha, tailclip=tailclip):
            new_value = current_value + (1 - alpha) * previous_value
            if current_value == 1 or current_value == -1:
                new_value = current_value

            if new_value < -1:
                newValue = -1
            elif new_value > 1:
                new_value = 1
            elif np.abs(new_value) < tailclip:
                new_value = 0

            return new_value

        smoothed_sign_change[i] = np.round(exponential_smoothing(value, previous_value, alpha=0.4, tailclip=0.1), 3)
        # print("new val: {}, Val: {}, prev val: {}".format(smoothed_sign_change[i], value, previous_value))

    return smoothed_sign_change


# def cscheme(colors):
#     aliases = {
#         'BkBu': ['black', 'blue'],
#         'gr': ['green', 'red'],
#         'grays': ['silver', 'gray'],
#         'mas': ['black', 'green', 'orange', 'red'],
#     }
#     aliases['default'] = aliases['gr']
#     return aliases[colors]
#
#
# def machart(kind, fast, medium, slow, append=True, last=last_, figsize=price_size, colors=cscheme('mas')):
#     title = ctitle(f"{kind.upper()}s", ticker=ticker, length=last)
#     ma1 = df.ta(kind=kind, length=fast, append=append)
#     ma2 = df.ta(kind=kind, length=medium, append=append)
#     ma3 = df.ta(kind=kind, length=slow, append=append)
#
#     madf = pd.concat([closedf, df[[ma1.name, ma2.name, ma3.name]]], axis=1, sort=False).tail(last)
#     madf.plot(figsize=figsize, title=title, color=colors, grid=True)
#
#
# def volumechart(kind, length=10, last=last_, figsize=ind_size, alpha=0.7, colors=cscheme('gr')):
#     title = ctitle("Volume", ticker=ticker, length=last)
#     volume = pd.DataFrame({'V+': volumedf[closedf > opendf], 'V-': volumedf[closedf < opendf]}).tail(last)
#
#     volume.plot(kind='bar', figsize=figsize, width=0.5, color=colors, alpha=alpha, stacked=True)
#     vadf = df.ta(kind=kind, close=volumedf, length=length).tail(last)
#     vadf.plot(figsize=figsize, lw=1.4, color='black', title=title, rot=45, grid=True)

def price_normalizer(source):
    '''
    Max and Min Price Values
    Normalize the price compared to e.g.the last 200 days to find new highs and lows.
    '''

    # 5d, 20d, 100d, and 200d norm value from [0,1]

    # list of normed days that are interesting
    normed_days = [50, 200]

    normed_days_features = pd.DataFrame(index=source.index)
    close = source['Close']

    for d in normed_days:
        temp_col = np.zeros(close.shape)
        for i, c in enumerate(close[:]):
            if i >= d:
                min_value = np.min(close[i - d + 1:i + 1])
                max_value = np.max(close[i - d + 1:i + 1])
                current_value = close[i]

                normed_value = (current_value - min_value) / (max_value - min_value)
                temp_col[i] = normed_value
            else:
                temp_col[i] = np.nan

            # display(close[i-4:i+1])
            # print(close[i])
            # normed_column = (close[i]-np.min(close[i-4:i+1]))/(np.max(close[i-5:i])-np.min(close[i-5:i]))

        normed_days_features = normed_days_features.join(pd.DataFrame(temp_col, columns=['NormKurs' + str(d)]))
    # display(close[190:210])
    # normed_days_features.iloc[190:210]

    print("Number of features: {}".format(normed_days_features.shape))
    print(normed_days_features.head(5))

    return normed_days_features

def impulse_count(source):
    '''
    Number of last days increase/decrease

    '''
    number_days_features = pd.DataFrame(index=source.index)
    close = source['Close']

    diff = close - close.shift(1)
    a = np.where(diff[0:20] > 0)
    a[0].shape[0] / 20

    # In the last 10days, the price increased x% of the time. 1=all days, 0=no days

    # list of normed days that are interesting
    number_days = [50, 200]

    #number_days_features = pd.DataFrame(index=features.index)


    for n in number_days:
        temp_col = np.zeros(diff.shape)
        for i, c in enumerate(diff[:]):
            if i >= n:
                rise_value = np.where(diff[i - n + 1:i + 1] > 0)[0].shape[0] / n
                temp_col[i] = rise_value
            else:
                temp_col[i] = np.nan

            # display(close[i-4:i+1])
            # print(close[i])
            # normed_column = (close[i]-np.min(close[i-4:i+1]))/(np.max(close[i-5:i])-np.min(close[i-5:i]))

        number_days_features = number_days_features.join(pd.DataFrame(temp_col, columns=['NumberRise' + str(n)]))
    # display(close[0:20])
    # normed_days_features.iloc[190:210]

    print("Number of features: {}".format(number_days_features.shape))
    print(number_days_features.head(10))

    return number_days_features

def calculate_moving_average(source):
    '''
    ### Generate mean values
    # Generate features - Mean value

    '''

    # meanList = [2, 5, 8, 10, 13, 15, 18, 20, 22, 34, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400];
    # meanList = [5 10 20 50 100 1 50 200]';
    # meanList = [5, 10, 20, 50, 100, 200];
    meanList = [50, 200]

    close = source['Close']
    meanfeatures = pd.DataFrame(index=source.index)

    for i in meanList:
        # Mean
        ta.sma(close, i, 0)
        meanCol = ta.sma(close, i, 0) #MA(close, i, 0);  # Trailing MA with i
        # meanCol.fillna(0, inplace=True)
        meanColreshaped = np.reshape(meanCol.values, (1, np.product(meanCol.values.shape)))[0]
        # Calculate diff from price in %
        diffPriceCol = np.divide(meanColreshaped - close.values, close.values)
        temp_source = pd.DataFrame(diffPriceCol, columns=['MA' + str(i) + 'Norm'])
        # print(temp_source)
        meanfeatures = meanfeatures.join(temp_source)
        # meanTable(:,i) = diffPriceCol;
        # print("Calculated MA{}".format(i));

    print("Number of features: {}".format(meanfeatures.shape))
    print(meanfeatures.head(5))

    return meanfeatures

def calculate_moving_average_direction(source, meanfeatures):
    ### Generate mean value directions

    # Create empty dataframe for the differences between the current time and previous time
    madiff_features = pd.DataFrame(index=source.index)

    for col in meanfeatures.columns:
        currentCol = meanfeatures[col]
        shiftCol = meanfeatures[col].shift(1)
        diff = currentCol - shiftCol
        diff.name = col + 'Diff'
        # display(currentCol)
        # display(shiftCol)
        madiff_features = madiff_features.join(diff)

    print("Number of features: {}".format(madiff_features.shape))
    print(madiff_features.head(5))

    return madiff_features

def get_rsi(source):
    '''
    ### Generate RSI


    # from talib import RSI
    # import ta   #https://github.com/bukosabino/ta
    #import pandas_ta as ta  # https://github.com/twopirllc/pandas-ta
    '''


    # rsiList = [2, 3, 5, 9, 10, 14, 20, 25];
    rsiList = [9, 14];

    close = source['Close']
    rsi_features = pd.DataFrame(index=source.index)

    for i in rsiList:
        # rsi(close, length=None, drift=None, offset=None, **kwargs)
        rsicol = ta.rsi(close, length=i)
        rsi_features = rsi_features.join(rsicol)

    print("Number of features: {}".format(rsi_features.shape))
    print(rsi_features.head(10))

    return rsi_features

def get_rsi_difference(source):
    '''
    # RSI shift, in which direction it is moving
    #import pandas_ta as ta  # https://github.com/twopirllc/pandas-ta
    ### Generate RSI difference

    '''

    rsiList = [9, 14];

    rsi_values = rsiList
    close = source['Close']
    rsi_change_features = pd.DataFrame(index=source.index)

    for period in rsi_values:
        rsi = ta.rsi(close, length=period)
        # Other column, here the same column shifted to find out if the direction changes
        rsi_diff = rsi - rsi.shift(1)
        rsi_diff.name = 'RSI' + str(period) + '_diff'
        rsi_change_features = rsi_change_features.join(rsi_diff)

    print("Number of features: {}".format(rsi_change_features.shape))
    print(rsi_change_features.head(10))

    return rsi_change_features

def get_rsi_signal(source):
    '''
    ### RSIx < value
    If RSI3 < 2 give signal, buying signal

    '''

    close = source['Close']
    rsi_signal_features = pd.DataFrame(index=source.index)

    # If RSI3 < 2 give signal, buying signal
    rsi3 = ta.rsi(close, length=3)
    rsi3_signal = (rsi3 < 5) * 1
    rsi3_decay_signal = generate_smoothed_trigger(rsi3_signal)
    rsi_signal_features = rsi_signal_features.join(pd.DataFrame(rsi3_decay_signal, columns=['RSI' + str(3) + 'sign']))

    print("Number of features: {}".format(rsi_signal_features.shape))
    print(rsi_signal_features[rsi_signal_features['RSI3sign'] == 1].head(5))

    return rsi_signal_features

def get_stochastics(source):
    '''
    ### Generate Stochastic
    # import pandas_ta as ta   #https://github.com/twopirllc/pandas-ta
    # help(ta.stoch)
    # from talib import STOCH
    import pandas_ta as ta  # https://github.com/twopirllc/pandas-ta
    # Recommended stochastics: [fk, sk, sd], [5,3,3], [21,7,7], [21,14,14]
    '''

    fastk_parameter = [13, 5, 21, 21, 3]
    slowk_parameter = [3, 3, 7, 14, 14]
    slowd_parameter = [8, 3, 7, 14, 14]

    close = source['Close']
    high = source['High']
    low = source['Low']
    stoch_features = pd.DataFrame(index=source.index)

    for fk, sd, sk in zip(fastk_parameter, slowk_parameter, slowd_parameter):
        print("Parameter: fastk={}, slowk={}, slowd={}".format(fk, sk, sd))

        df = ta.stoch(high, low, close, fast_k=fk, slow_k=sk, slow_d=sd)
        # print(df.columns)

        # print(pd.Series(df['STOCH_' + str(sk)], name='Stoch_Sk' + str(fk)+str(sk)+str(sd)))

        stoch_features = stoch_features.join(
            pd.Series(df['STOCH_' + str(sk)], name='Stoch_Sk' + str(fk) + str(sk) + str(sd)))
        stoch_features = stoch_features.join(
            pd.Series(df['STOCH_' + str(sd)], name='Stoch_Sd' + str(fk) + str(sk) + str(sd)))

    print("Number of features: {}".format(stoch_features.shape))
    print(stoch_features.head(5))

    return stoch_features

def get_macd(source):
    '''
    ### MACD
    help(ta.macd)
    '''

    # MACD
    # Recommended parameters: 12_26_9, 5, 35, 5
    fast_macd = [12, 5]
    slow_macd = [26, 35]
    signal_macd = [9, 5]

    close = source['Close']
    macd_features = pd.DataFrame(index=source.index)

    # def ctitle(indicator_name, ticker='SPY', length=100):
    #    return f"{ticker}: {indicator_name} from {recent_startdate} to {recent_startdate} ({length})"

    # recent_startdate = source_cut.tail(recent).index[0]
    # recent_enddate = source_cut.tail(recent).index[-1]
    # price_size = (16, 8)
    # ind_size = (16, 2)
    # ticker = 'SPY'
    # recent = 126
    # half_of_recent = int(0.5 * recent)

    # def plot_MACD(macddf):
    #    macddf[[macddf.columns[0], macddf.columns[2]]].tail(recent).plot(figsize=(16, 2), color=cscheme('BkBu'), linewidth=1.3)
    #    macddf[macddf.columns[1]].tail(recent).plot.area(figsize=ind_size, stacked=False, color=['silver'], linewidth=1, title=ctitle(macddf.name, ticker=ticker, length=recent), grid=True).axhline(y=0, color="black", lw=1.1)

    for fmacd, smacd, sigmacd in zip(fast_macd, slow_macd, signal_macd):
        print("Generate fast mcd={}, slow macd={}, signal macd={}".format(fmacd, smacd, sigmacd))
        macddf = ta.macd(close, fast=fmacd, slow=smacd, signal=sigmacd)
        # display(macddf.iloc[:,0].head(50))
        # plot_MACD(macddf)

        macd_features = macd_features.join(
            pd.Series(macddf[macddf.columns[0]], name='MACD_' + str(fmacd) + "_" + str(smacd) + "_" + str(sigmacd)))
        macd_features = macd_features.join(
            pd.Series(macddf[macddf.columns[2]], name='MACDS_' + str(fmacd) + "_" + str(smacd) + "_" + str(sigmacd)))

    print("Number of features: {}".format(macd_features.shape))
    print(macd_features.iloc[20:40, :])

    #macddf = ta.macd(close, fast=8, slow=21, signal=9, min_periods=None, append=True)

    # features = features.join(macd_features)

    return macd_features

def get_macd_difference(macd_features):
    '''
    ### MACD Difference
    # MACD direction
    import pandas_ta as ta  # https://github.com/twopirllc/pandas-ta
    '''

    # Create empty dataframe for the differences between the current time and previous time
    macd_diff_features = pd.DataFrame(index=macd_features.index)

    for col in macd_features.columns:
        currentCol = macd_features[col]
        shiftCol = macd_features[col].shift(1)
        diff = currentCol - shiftCol
        diff.name = col + 'Diff'
        # display(currentCol)
        # display(shiftCol)
        macd_diff_features = macd_diff_features.join(diff)

    print("Number of features: {}".format(macd_diff_features.shape))
    print(macd_diff_features.iloc[30:40])

    return macd_diff_features

def get_trigger_signals(macd_diff_features):
    '''
    Signals for Trigger

    '''

    # If MACD changes direction
    macd_direction_change_features = pd.DataFrame(index=macd_diff_features.index)

    for col in macd_diff_features.columns:
        # Column to find signal on
        currentCol = macd_diff_features[col]
        # Other column, here the same column shifted to find out if the direction changes
        shiftCol = macd_diff_features[col].shift(1)

        print(currentCol.iloc[90:100])
        # display(shiftCol.iloc[30:60])

        # Multiply current diff with previous diff and get the sign of the product. If sign is negative, then direction change
        # has occured. The multiply with the sign of the current value to get the sign of the direction change. If 1, then
        # it was a change from negative to positive. If it was negative, then it was a change from negative to positive
        signChange = (np.sign(currentCol * shiftCol) == -1) * 1 * np.sign(currentCol)
        print(signChange[90:100])

        smoothed_sign_change = generate_smoothed_trigger(signChange)
        macd_direction_change_features = macd_direction_change_features.join(
            pd.Series(data=smoothed_sign_change, name=col + 'DirChange'))

    print("Number of features: {}".format(macd_direction_change_features.shape))
    print(macd_direction_change_features.iloc[90:100])

    return macd_direction_change_features

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))

def get_periodical_indicators(source):
    '''
    ### Periodical indicators

    '''

    # Generate periodical values
    periodic_values = pd.DataFrame(index=source.index)
    timelist = source['Date']
    # Get month of year
    periodic_values['month_of_year'] = timelist.apply(lambda x: x.month)
    # Get week of year
    periodic_values['week_of_year'] = timelist.apply(lambda x: x.week)
    # Get day of year
    periodic_values['day_of_year'] = timelist.apply(lambda x: x.timetuple().tm_yday)
    # Get day of month
    periodic_values['day_of_month'] = timelist.apply(lambda x: x.day)
    # Get day of week
    periodic_values['day_of_week'] = timelist.apply(lambda x: x.weekday())
    # Get week of month
    periodic_values['week_of_month'] = timelist.apply(week_of_month)
    print(periodic_values.head())

    # Make one-hot-encoding of the values as they do not depend on each other
    from sklearn.preprocessing import OneHotEncoder

    # One hot encoding for day of week
    periodic_values = periodic_values.join(pd.get_dummies(periodic_values['day_of_week'], prefix='day_week_')).drop(
        ['day_of_week'], axis=1)
    # For special weeks, there are day of week 5 and 6. Remove them, as they are special cases
    periodic_values.drop(columns=['day_week__5', 'day_week__6'], errors='ignore', inplace=True)

    # One hot encoding for month of year
    periodic_values = periodic_values.join(pd.get_dummies(periodic_values['month_of_year'], prefix='month_year_')).drop(
        ['month_of_year'], axis=1)

    # One hot encoding week of month
    periodic_values = periodic_values.join(pd.get_dummies(periodic_values['week_of_month'], prefix='week_month_')).drop(
        ['week_of_month'], axis=1)

    print("Number of features: {}".format(periodic_values.shape))

    # features = features.join(periodic_values)

    print(periodic_values.head())
    # onehot_encoder = OneHotEncoder(sparse=False)
    #i  # nteger_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # onehot_encoded = onehot_encoder.fit_transform(periodic_values['day_of_week'])
    # print(onehot_encoded)

    return periodic_values

# def get_candle_stick_patterns():
#     '''
#     ### Candlestick patterns
#     Link: https: // mrjbq7.github.io / ta - lib / func_groups / pattern_recognition.html
#
#     '''
#
#     from talib import CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE
#
#     # FIXME: Get Open and create the patterns
#
#     pattern1 = CDL2CROWS(close, high, low, close)
#     pattern1 = CDL3BLACKCROWS(close, high, low, close)
#     i = CDL3INSIDE(close, high, low, close)
#     display(np.sum(pattern1))

def main():
    conf = sup.load_config(args.config_path)

    image_save_directory = os.path.join(conf['Paths'].get('result_directory'), "data_preparation_images")
    #outcomes_filename = conf['training_data_directory'] + "/" + conf['dataset_name'] + "_outcomes" + ".csv"
    features_filename_uncut = os.path.join(conf['Paths'].get('training_data_directory'), conf['Common'].get('dataset_name') + "_features_uncut" + ".csv")
    #features_filename_uncut = conf['training_data_directory'] + "/" + conf['dataset_name'] + "_features_uncut" + ".csv"


    #Load only a subset of the whole raw data to create a debug dataset
    source = custom.load_source(conf['Paths'].get('source_path')) #.iloc[0:1000, :]
    #Plot source
    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(source['Date'], source['Close'])
    plt.title(conf['Paths'].get('source_path'))
    plt.show(block = False)

    # Define features df
    features = pd.DataFrame(index=source.index)

    # Generate Price Based Values

    normed_days_features = price_normalizer(source)
    features = features.join(normed_days_features)

    number_days_features = impulse_count(source)
    features = features.join(number_days_features)

    mean_features = calculate_moving_average(source)
    features = features.join(mean_features)

    madiff_features = calculate_moving_average_direction(source, mean_features)
    features = features.join(madiff_features)

    rsi_features = get_rsi(source)
    features = features.join(rsi_features)

    rsi_change_features = get_rsi_difference(source)
    features = features.join(rsi_change_features)

    rsi_signal_features = get_rsi_signal(source)
    features = features.join(rsi_signal_features)

    stoch_features = get_stochastics(source)
    features = features.join(stoch_features)

    plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(311)
    plt.plot(source['Date'][0:100], source['Close'][0:100])
    plt.title("Close")
    plt.subplot(312)
    plt.title("Stochastic Variant " + str(stoch_features.columns[1]))
    plt.plot(source['Date'][0:100], stoch_features.iloc[:, 1][0:100])
    plt.plot(source['Date'][0:100], stoch_features.iloc[:, 0][0:100])
    plt.subplot(313)
    plt.title("Stochastic Variant " + str(stoch_features.columns[-1]))
    plt.plot(source['Date'][0:100], stoch_features.iloc[:, -1][0:100])
    plt.plot(source['Date'][0:100], stoch_features.iloc[:, -2][0:100])
    plt.tight_layout()

    macd_features = get_macd(source)
    features = features.join(macd_features)

    plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(311)
    plt.plot(source['Date'][0:100], source['Close'][0:100])
    plt.title("Close")
    plt.subplot(312)
    plt.title("MACD Variant 1")
    plt.plot(source['Date'][0:100], macd_features.iloc[:, 0][0:100])
    plt.plot(source['Date'][0:100], macd_features.iloc[:, 1][0:100])
    plt.legend(("MACD", "MACD Signal"))
    plt.subplot(313)
    plt.title("MACD Variant 1")
    plt.plot(source['Date'][0:100], macd_features.iloc[:, -2][0:100])
    plt.plot(source['Date'][0:100], macd_features.iloc[:, -1][0:100])
    plt.legend(("MACD", "MACD Signal"))
    plt.tight_layout()

    macd_diff_features = get_macd_difference(macd_features)
    features = features.join(macd_diff_features)

    macd_direction_change_features = get_trigger_signals(macd_diff_features)
    features = features.join(macd_direction_change_features)

    periodic_values = get_periodical_indicators(source)
    features = features.join(periodic_values)

    # Features structure
    print("Features: ", features.head(10))
    print("Features shape: ", features.shape)

    # Save features to a csv file
    print("Features shape {}".format(features.shape))
    features.to_csv(features_filename_uncut, sep=';', index=True, header=True)
    print("Saved features to " + features_filename_uncut)

    print("=== Data for {} prepared to be trained ===".format(conf['Common'].get('dataset_name')))


if __name__ == "__main__":
    main()


    print("=== Program end ===")