import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

#import data_visualization_functions as vis


def load_source(source_path):
    '''
    Load stock charts as source


    '''
    source = pd.read_csv(source_path, sep=';')
    source.index.name = "id"
    source.columns = ['Date', 'Open', 'High', 'Low', 'Close']
    source['Date'] = pd.to_datetime(source['Date'])
    source['Date'].apply(mdates.date2num)

    return source