#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2X Data generation: Generate a comprehensive set of features for stock markets based on OHLC data.

This script calculates a wide variety of technical indicators and features from raw
stock market OHLC data. These features can then be used to train a machine
learning model for stock market prediction.

This script generates a broad range of features, including:
    - Price normalization over various periods.
    - Impulse counts (number of rising days in a period).
    - A comprehensive set of Simple Moving Averages (SMAs) with different window sizes.
    - Direction of the moving averages.
    - Relative Strength Index (RSI) with various periods.
    - RSI differences.
    - Stochastic oscillators with multiple parameter sets.
    - Moving Average Convergence Divergence (MACD) with different settings.
    - MACD differences and trigger signals.
    - Periodical indicators (day of week, month of year, etc.).

Inputs:
    - Configuration file (specified by --config_path argument): Contains paths
      for raw data, prepared data, and results directories.
    - Raw stock market OHLC data: Loaded from the path specified in the config file.

Outputs:
    - `temp_features_uncut.csv`: A CSV file containing all the generated features.
    - Various PNG plots: Visualizations of the raw data and some of the generated
      features (e.g., Stochastics, MACD). These are saved in a 'data_generation'
      subdirectory within the configured results directory.

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
# import utils.data_visualization_functions as vis
import utils.custom_methods as custom
import utils.data_handling_support_functions as sup

__author__ = "Alexander Wendt"
__copyright__ = (
    "Copyright 2020, Christian Doppler Laboratory for " "Embedded Machine Learning"
)
__credits__ = [""]
__license__ = "ISC"
__version__ = "0.2.0"
__maintainer__ = "Alexander Wendt"
__email__ = "alexander.wendt@tuwien.ac.at"
__status__ = "Experiental"

# Global settings
np.set_printoptions(precision=3)
# Suppress print out in scientific notiation
np.set_printoptions(suppress=True)
register_matplotlib_converters()

parser = argparse.ArgumentParser(description="Step 2.1 - Generate features from raw data")
# parser.add_argument("-r", '--retrain_all_data', action='store_true',
#                    help='Set flag if retraining with all available data shall be performed after ev')
parser.add_argument(
    "-conf",
    "--config_path",
    default="config/debug_timedata_omxS30.ini",
    help="Configuration file path",
    required=False,
)
parser.add_argument(
    "-debug",
    "--debug_param",
    default=False,
    action="store_true",
    help="Use debug parameters",
)
# parser.add_argument("-i", "--on_inference_data", action='store_true',
#                    help="Set inference if only inference and no training")

args = parser.parse_args()


def generate_smoothed_trigger(values, alpha=0.5, tailclip=0.1):
    """
    From a value array with signals in the range -1, 0, 1, generate smoothed decay.

    Parameters
    ----------
    values : np.ndarray
        The input array with signals.
    alpha : float, optional
        The smoothing factor, by default 0.5.
    tailclip : float, optional
        The tail clipping value, by default 0.1.

    Returns
    -------
    np.ndarray
        The array with smoothed signals.
    """

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
                new_value = -1
            elif new_value > 1:
                new_value = 1
            elif np.abs(new_value) < tailclip:
                new_value = 0

            return new_value

        smoothed_sign_change[i] = np.round(
            exponential_smoothing(value, previous_value, alpha=0.4, tailclip=0.1), 3
        )
        # print("new val: {}, Val: {}, prev val: {}".format(smoothed_sign_change[i], value, previous_value))

    return smoothed_sign_change


def price_normalizer(source, debug_param=False):
    """
    Normalize the price compared to e.g.the last 200 days to find new highs and lows.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.
    debug_param : bool, optional
        If True, use a smaller set of parameters for debugging, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the normalized price features.
    """

    # 5d, 20d, 100d, and 200d norm value from [0,1]

    # list of normed days that are interesting
    if debug_param:
        normed_days = [5, 200]
    else:
        normed_days = [5, 20, 50, 100, 200]

    normed_days_features = pd.DataFrame(index=source.index)
    close = source["Close"]

    for d in normed_days:
        temp_col = np.zeros(close.shape)
        for i, c in enumerate(close[:]):
            if i >= d:
                min_value = np.min(close[i - d + 1 : i + 1])
                max_value = np.max(close[i - d + 1 : i + 1])
                current_value = close[i]

                normed_value = (current_value - min_value) / (max_value - min_value)
                temp_col[i] = normed_value
            else:
                temp_col[i] = np.nan

        normed_days_features = normed_days_features.join(
            pd.DataFrame(temp_col, columns=["NormKurs" + str(d)])
        )

    print("Number of features: {}".format(normed_days_features.shape))
    print(normed_days_features.head(5))

    return normed_days_features


def impulse_count(source, debug_param=False):
    """
    Calculate the number of days the price increased in a given period.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.
    debug_param : bool, optional
        If True, use a smaller set of parameters for debugging, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the impulse count features.
    """
    number_days_features = pd.DataFrame(index=source.index)
    close = source["Close"]

    diff = close - close.shift(1)
    a = np.where(diff[0:20] > 0)
    a[0].shape[0] / 20

    # In the last 10days, the price increased x% of the time. 1=all days, 0=no days

    # list of normed days that are interesting
    if not debug_param:
        number_days = [50, 100, 200]
    else:
        number_days = [200]

    for n in number_days:
        temp_col = np.zeros(diff.shape)
        for i, c in enumerate(diff[:]):
            if i >= n:
                rise_value = np.where(diff[i - n + 1 : i + 1] > 0)[0].shape[0] / n
                temp_col[i] = rise_value
            else:
                temp_col[i] = np.nan

        number_days_features = number_days_features.join(
            pd.DataFrame(temp_col, columns=["NumberRise" + str(n)])
        )

    print("Number of features: {}".format(number_days_features.shape))
    print(number_days_features.head(10))

    return number_days_features


def calculate_moving_average(source, debug_param=False):
    """
    Generate moving average features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.
    debug_param : bool, optional
        If True, use a smaller set of parameters for debugging, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the moving average features.
    """

    if not debug_param:
        meanList = [
            2,
            5,
            8,
            10,
            13,
            15,
            18,
            20,
            22,
            34,
            40,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
        ]
    else:
        meanList = [50, 200]

    close = source["Close"]
    meanfeatures = pd.DataFrame(index=source.index)

    for i in meanList:
        meanCol = ta.sma(close, i, 0)
        meanColreshaped = np.reshape(meanCol.values, (1, np.product(meanCol.values.shape)))[0]
        diffPriceCol = np.divide(close.values - meanColreshaped, meanColreshaped)
        temp_source = pd.DataFrame(diffPriceCol, columns=["SMA" + str(i) + ""])
        meanfeatures = meanfeatures.join(temp_source)

    print("Number of features: {}".format(meanfeatures.shape))
    print(meanfeatures.head(5))

    return meanfeatures


def calculate_moving_average_direction(source, meanfeatures):
    """
    Generate moving average direction features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.
    meanfeatures : pd.DataFrame
        The moving average features.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the moving average direction features.
    """

    madiff_features = pd.DataFrame(index=source.index)

    for col in meanfeatures.columns:
        currentCol = meanfeatures[col]
        shiftCol = meanfeatures[col].shift(1)
        diff = currentCol - shiftCol
        diff.name = col + "Diff"
        madiff_features = madiff_features.join(diff)

    print("Number of features: {}".format(madiff_features.shape))
    print(madiff_features.head(5))

    return madiff_features


def get_rsi(source, debug_param=False):
    """
    Generate Relative Strength Index (RSI) features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.
    debug_param : bool, optional
        If True, use a smaller set of parameters for debugging, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the RSI features.
    """

    if not debug_param:
        rsiList = [2, 3, 5, 9, 10, 14, 20, 25]
    else:
        rsiList = [9, 14]

    close = source["Close"]
    rsi_features = pd.DataFrame(index=source.index)

    for i in rsiList:
        rsicol = ta.rsi(close, length=i)
        rsi_features = rsi_features.join(rsicol)

    print("Number of features: {}".format(rsi_features.shape))
    print(rsi_features.head(10))

    return rsi_features


def get_rsi_difference(source):
    """
    Generate RSI difference features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the RSI difference features.
    """
    rsiList = [2, 3, 5, 9, 10, 14, 20, 25]

    rsi_values = rsiList
    close = source["Close"]
    rsi_change_features = pd.DataFrame(index=source.index)

    for period in rsi_values:
        rsi = ta.rsi(close, length=period)
        rsi_diff = rsi - rsi.shift(1)
        rsi_diff.name = "RSI" + str(period) + "_diff"
        rsi_change_features = rsi_change_features.join(rsi_diff)

    print("Number of features: {}".format(rsi_change_features.shape))
    print(rsi_change_features.head(10))

    return rsi_change_features


def get_rsi_signal(source):
    """
    Generate RSI signal features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the RSI signal features.
    """

    close = source["Close"]
    rsi_signal_features = pd.DataFrame(index=source.index)

    rsi3 = ta.rsi(close, length=3)
    rsi3_signal = (rsi3 < 5) * 1
    rsi3_decay_signal = generate_smoothed_trigger(rsi3_signal)
    rsi_signal_features = rsi_signal_features.join(
        pd.DataFrame(rsi3_decay_signal, columns=["RSI" + str(3) + "sign"])
    )

    print("Number of features: {}".format(rsi_signal_features.shape))
    print(rsi_signal_features[rsi_signal_features["RSI3sign"] == 1].head(5))

    return rsi_signal_features


def get_stochastics(source):
    """
    Generate stochastic oscillator features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the stochastic oscillator features.
    """

    fastk_parameter = [14, 13, 5, 21, 21, 3]
    slowk_parameter = [3, 3, 3, 7, 14, 14]
    slowd_parameter = [3, 8, 3, 7, 14, 14]

    close = source["Close"]
    high = source["High"]
    low = source["Low"]
    stoch_features = pd.DataFrame(index=source.index)

    for fk, sk, sd in zip(fastk_parameter, slowk_parameter, slowd_parameter):
        print("Parameter: fastk={}, slowk={}, slowd={}".format(fk, sk, sd))

        df = ta.stoch(high, low, close, k=fk, d=sk, smooth_k=sd)

        stoch_features = stoch_features.join(
            pd.Series(
                df["STOCHk_" + str(fk) + "_" + str(sk) + "_" + str(sd)],
                name="Stoch_Sk" + str(fk) + str(sk) + str(sd),
            )
        )
        stoch_features = stoch_features.join(
            pd.Series(
                df["STOCHd_" + str(fk) + "_" + str(sk) + "_" + str(sd)],
                name="Stoch_Sd" + str(fk) + str(sk) + str(sd),
            )
        )

    print("Number of features: {}".format(stoch_features.shape))
    print(stoch_features.head(5))

    return stoch_features


def get_macd(source):
    """
    Generate Moving Average Convergence Divergence (MACD) features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the MACD features.
    """

    fast_macd = [12, 5]
    slow_macd = [26, 35]
    signal_macd = [9, 5]

    close = source["Close"]
    macd_features = pd.DataFrame(index=source.index)

    for fmacd, smacd, sigmacd in zip(fast_macd, slow_macd, signal_macd):
        print(
            "Generate fast mcd={}, slow macd={}, signal macd={}".format(
                fmacd, smacd, sigmacd
            )
        )
        macddf = ta.macd(close, fast=fmacd, slow=smacd, signal=sigmacd)

        macd_features = macd_features.join(
            pd.Series(
                macddf[macddf.columns[0]],
                name="MACD_" + str(fmacd) + "_" + str(smacd) + "_" + str(sigmacd),
            )
        )
        macd_features = macd_features.join(
            pd.Series(
                macddf[macddf.columns[2]],
                name="MACDS_" + str(fmacd) + "_" + str(smacd) + "_" + str(sigmacd),
            )
        )

    print("Number of features: {}".format(macd_features.shape))
    print(macd_features.iloc[20:40, :])

    return macd_features


def get_macd_difference(macd_features):
    """
    Generate MACD difference features.

    Parameters
    ----------
    macd_features : pd.DataFrame
        The MACD features.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the MACD difference features.
    """

    macd_diff_features = pd.DataFrame(index=macd_features.index)

    for col in macd_features.columns:
        currentCol = macd_features[col]
        shiftCol = macd_features[col].shift(1)
        diff = currentCol - shiftCol
        diff.name = col + "Diff"
        macd_diff_features = macd_diff_features.join(diff)

    print("Number of features: {}".format(macd_diff_features.shape))
    print(macd_diff_features.iloc[30:40])

    return macd_diff_features


def get_trigger_signals(macd_diff_features):
    """
    Generate trigger signals from MACD difference features.

    Parameters
    ----------
    macd_diff_features : pd.DataFrame
        The MACD difference features.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the trigger signals.
    """

    macd_direction_change_features = pd.DataFrame(index=macd_diff_features.index)

    for col in macd_diff_features.columns:
        currentCol = macd_diff_features[col]
        shiftCol = macd_diff_features[col].shift(1)

        print(currentCol.iloc[90:100])

        signChange = (np.sign(currentCol * shiftCol) == -1) * 1 * np.sign(currentCol)
        print(signChange[90:100])

        smoothed_sign_change = generate_smoothed_trigger(signChange)
        macd_direction_change_features = macd_direction_change_features.join(
            pd.Series(data=smoothed_sign_change, name=col + "DirChange")
        )

    print("Number of features: {}".format(macd_direction_change_features.shape))
    print(macd_direction_change_features.iloc[90:100])

    return macd_direction_change_features


def week_of_month(dt):
    """
    Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))


def get_periodical_indicators(source):
    """
    Generate periodical indicator features.

    Parameters
    ----------
    source : pd.DataFrame
        The source OHLC data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the periodical indicator features.
    """

    periodic_values = pd.DataFrame(index=source.index)
    timelist = source["Date"]
    periodic_values["month_of_year"] = timelist.apply(lambda x: x.month)
    periodic_values["week_of_year"] = timelist.apply(lambda x: x.week)
    periodic_values["day_of_year"] = timelist.apply(lambda x: x.timetuple().tm_yday)
    periodic_values["day_of_month"] = timelist.apply(lambda x: x.day)
    periodic_values["day_of_week"] = timelist.apply(lambda x: x.weekday())
    periodic_values["week_of_month"] = timelist.apply(week_of_month)
    print(periodic_values.head())

    periodic_values = periodic_values.join(
        pd.get_dummies(periodic_values["day_of_week"], prefix="day_week_")
    ).drop(["day_of_week"], axis=1)
    periodic_values.drop(
        columns=["day_week__5", "day_week__6"], errors="ignore", inplace=True
    )

    periodic_values = periodic_values.join(
        pd.get_dummies(periodic_values["month_of_year"], prefix="month_year_")
    ).drop(["month_of_year"], axis=1)

    periodic_values = periodic_values.join(
        pd.get_dummies(periodic_values["week_of_month"], prefix="week_month_")
    ).drop(["week_of_month"], axis=1)

    print("Number of features: {}".format(periodic_values.shape))
    print(periodic_values.head())

    return periodic_values


def main(config_path, debug_param):
    """
    Main function to execute the script.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    debug_param : bool
        If True, use a smaller set of parameters for debugging.
    """
    conf = sup.load_config(config_path)

    image_save_directory = os.path.join(
        conf["Paths"].get("results_directory"), "data_generation"
    )
    features_filename_uncut = os.path.join(
        conf["Paths"].get("prepared_data_directory"),
        "temp",
        "temp_features_uncut" + ".csv",
    )
    os.makedirs(os.path.dirname(features_filename_uncut), exist_ok=True)

    source = custom.load_source(conf["Paths"].get("source_path"))

    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor="w", edgecolor="k")
    plt.plot(source["Date"], source["Close"])
    plt.title(conf["Paths"].get("source_path"))
    plt.show(block=False)

    features = pd.DataFrame(index=source.index)

    normed_days_features = price_normalizer(source, debug_param=debug_param)
    features = features.join(normed_days_features)

    number_days_features = impulse_count(source, debug_param=debug_param)
    features = features.join(number_days_features)

    mean_features = calculate_moving_average(source, debug_param=debug_param)
    features = features.join(mean_features)

    madiff_features = calculate_moving_average_direction(source, mean_features)
    features = features.join(madiff_features)

    rsi_features = get_rsi(source, debug_param=debug_param)
    features = features.join(rsi_features)

    rsi_change_features = get_rsi_difference(source)
    features = features.join(rsi_change_features)

    stoch_features = get_stochastics(source)
    features = features.join(stoch_features)

    plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor="w", edgecolor="k")
    plt.subplot(311)
    plt.plot(source["Date"][0:100], source["Close"][0:100])
    plt.title("Close")
    plt.subplot(312)
    plt.title("Stochastic Variant " + str(stoch_features.columns[1]))
    plt.plot(source["Date"][0:100], stoch_features.iloc[:, 1][0:100])
    plt.plot(source["Date"][0:100], stoch_features.iloc[:, 0][0:100])
    plt.subplot(313)
    plt.title("Stochastic Variant " + str(stoch_features.columns[-1]))
    plt.plot(source["Date"][0:100], stoch_features.iloc[:, -1][0:100])
    plt.plot(source["Date"][0:100], stoch_features.iloc[:, -2][0:100])
    plt.tight_layout()

    macd_features = get_macd(source)
    features = features.join(macd_features)

    plt.figure(num=None, figsize=(10, 7), dpi=80, facecolor="w", edgecolor="k")
    plt.subplot(311)
    plt.plot(source["Date"][0:100], source["Close"][0:100])
    plt.title("Close")
    plt.subplot(312)
    plt.title("MACD Variant 1")
    plt.plot(source["Date"][0:100], macd_features.iloc[:, 0][0:100])
    plt.plot(source["Date"][0:100], macd_features.iloc[:, 1][0:100])
    plt.legend(("MACD", "MACD Signal"))
    plt.subplot(313)
    plt.title("MACD Variant 1")
    plt.plot(source["Date"][0:100], macd_features.iloc[:, -2][0:100])
    plt.plot(source["Date"][0:100], macd_features.iloc[:, -1][0:100])
    plt.legend(("MACD", "MACD Signal"))
    plt.tight_layout()

    macd_diff_features = get_macd_difference(macd_features)
    features = features.join(macd_diff_features)

    macd_direction_change_features = get_trigger_signals(macd_diff_features)
    features = features.join(macd_direction_change_features)

    periodic_values = get_periodical_indicators(source)
    features = features.join(periodic_values)

    print("Features: ", features.head(10))
    print("Features shape: ", features.shape)

    print("Features shape {}".format(features.shape))
    features.to_csv(features_filename_uncut, sep=";", index=True, header=True)
    print("Saved features to " + features_filename_uncut)

    print(
        "=== Data for {} prepared to be trained or inferred ===".format(
            conf["Common"].get("dataset_name")
        )
    )


if __name__ == "__main__":
    main(args.config_path, args.debug_param)

    print("=== Program end ===")
