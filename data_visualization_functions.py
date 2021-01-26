import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from statsmodels import robust
import scikitplot as skplt
import itertools
from scipy.stats import ks_2samp
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import data_handling_support_functions as sup

def paintBarChartForMissingValues(xlabels, yvalues):
    # Create the figure
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

    #Crate a color scale
    #Create a color map
    colorMap = m.colors.LinearSegmentedColormap.from_list("colorMap",["green", "yellow", "red", "red", "red", "red", "red"])

    def calculateProportion(value, minValue, maxValue):
        colorValue = 0
        if value>maxValue:
            colorValue=1
        elif value < minValue:
            colorValue=0
        else:
            colorValue=(value)/(maxValue-minValue)
        return colorValue

    barProportions = [calculateProportion(i, 0, 0.3) for i in yvalues]
    barColors = colorMap(barProportions)

    # Create a color bar
    cscalar = m.cm.ScalarMappable(cmap=colorMap, norm=plt.Normalize(0,0.3))
    cscalar.set_array([])
    colorbar = plt.colorbar(cscalar, orientation='vertical', ticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3])
    colorbar.set_label('Low and high values', rotation=270,labelpad=25)

    #Plot
    plt.xlabel("Missing features")
    plt.ylabel('Share of missing features')
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_ticks(np.arange(0, 0.3, 0.01))
    plt.title('Missing Features Bar chart')
    plt.gca().set_ylim([0, 0.3])
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    bar = plt.bar(xlabels, yvalues, color=barColors, width=1.0, capsize=10, edgecolor='black')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def paintBarChartForCategorical(series, xlegend="Feature Label", ylegend="Share of Feature Labels", title="Distribution of Labels for Feature"):
    ''' Plot bar chart for categorical values, ranked by occurence

    :xlabels: Labels on X axis as list
    :yvalues: Values on y axis for categorical values as series

    :return: figure
    '''
    # Create the figure
    fig = plt.figure(num=None, figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')

    #Crate a color scale
    #Create a color map
    colorMap = plt.cm.get_cmap('GnBu') #m.colors.LinearSegmentedColormap.from_list("colorMap",["green", "yellow", "red"])

    def calculateProportion(value, minValue, maxValue):
        colorValue = 0
        if value>maxValue:
            colorValue=1
        elif value < minValue:
            colorValue=0
        else:
            colorValue=(value)/(maxValue-minValue)
        return colorValue

    barProportions = [calculateProportion(i, 0, max(series)) for i in series]
    barColors = colorMap(barProportions)

    # Create a color bar
    cscalar = m.cm.ScalarMappable(cmap=colorMap, norm=plt.Normalize(0,max(series)))
    cscalar.set_array([])
    #colorbar = plt.colorbar(cscalar, orientation='vertical', ticks=[0, 0.05, max(yvalues)])
    #colorbar.set_label('Low and high values', rotation=270,labelpad=25)

    #fix xlabel
    xlabel_string = list(map(str, series.index))
    xlabel_string = list(map(lambda x: x[0:20], xlabel_string))

    #Plot
    plt.xlabel(xlegend)
    plt.ylabel(ylegend)
    plt.xticks(rotation=90)
    #plt.gca().yaxis.set_ticks(np.arange(0, 0.1, 0.005))
    plt.title(title)
    #plt.gca().set_ylim([0, 0.1])
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3, left=0.1)
    bar = plt.bar(xlabel_string, series, color=barColors, width=1.0, capsize=10, edgecolor='black')
	
    return fig


# Calculate Interval length according to Freedman und Diaconis
def getNumberOfIntervals(valueDistribution):
    numFeatures = valueDistribution.shape[0]
    featuregroup = valueDistribution.value_counts().sort_index(ascending=True)
    numUniqueFeatures = featuregroup.shape[0]
    print("Number of unique features {} ".format(numUniqueFeatures))

    n = valueDistribution.shape[0]
    q75 = np.quantile(valueDistribution, 0.75)
    q25 = np.quantile(valueDistribution, 0.25)
    minValue = min(valueDistribution)
    maxValue = max(valueDistribution)
    IQR = q75 - q25
    hn = 2 * IQR / np.power(n, 1 / 3)
    if hn != 0:
        hnIntervalCount = int(round((maxValue - minValue) / hn, 0))
    else:
        hnIntervalCount = 0

        # If unique features is too big, i.e. > 100, then reduce to max 100 bins for the historgram
    if hnIntervalCount > 0 and numUniqueFeatures >= hnIntervalCount:
        numIntervals = hnIntervalCount
        print("Number of bins with Freedman und Diaconis: ", numIntervals)
    elif hn > 0 and numUniqueFeatures < hnIntervalCount:
        numIntervals = numUniqueFeatures
        print("Number of bins = number of features {}: ", numIntervals)
    else:
        if (numUniqueFeatures > 100):
            bins = np.linspace(min(featuregroup), max(featuregroup), 100)
            print("Number of bins limited to 100: ", bins.shape[0])
        else:
            bins = np.linspace(min(featuregroup), max(featuregroup), featuregroup.shape[0])
            print("Number of bins <= 100: ", bins.shape[0])
        numIntervals = bins.shape[0]
    print(
        "n={}, q25={:.2f}, q75={:.2f}, "
        "min={:.2f}, max={:.2f}, "
        "interval length={:.2f}. Number of intervals={}".format(n,q25,q75,minValue,maxValue,hn,numIntervals))
    return numIntervals


# Get number of missing values per feature, i.e. share of '?' per feature
# d=df['workclass'].iloc[27]
def paintHistogram(df, colName):
    # colName = 'age'
    # Create the histogram data
    mean = df[colName].mean()
    median = df[colName].median()
    sigma = df[colName].std()
    s_mad = robust.mad(df[colName])
    skew = df[colName].skew()
    kurtosis = df[colName].kurtosis()

    bins = getNumberOfIntervals(df[colName])

    # Plot the graph
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel(colName)
    plt.ylabel('Probability')
    plt.title('{} Histogram'.format(colName))

    # bar = plt.bar(featuregroup.index, featuregroup/numFeatures, width=1.0, capsize=10, edgecolor='black')
    hist = plt.hist(df[colName], bins, density=True, edgecolor='black', stacked=True)
    plt.gca().axvline(x=mean, color='red', alpha=0.8, linewidth=2)  # alpha =transperacy
    plt.gca().axvline(x=mean + 2*sigma, color='red', alpha=0.8, linewidth=2)  # alpha =transperacy
    plt.gca().axvline(x=mean - 2*sigma, color='red', alpha=0.8, linewidth=2)  # alpha =transperacy
	
    plt.gca().axvline(x=median, color='green', alpha=0.8, linewidth=2)  # alpha =transperacy
    plt.gca().axvline(x=median + 2*s_mad, color='green', alpha=0.8, linewidth=2)  # alpha =transperacy
    plt.gca().axvline(x=median - 2*s_mad, color='green', alpha=0.8, linewidth=2)  # alpha =transperacy
	
    # Add text
    plt.text(mean * 1.01, plt.gca().get_ylim()[1] * 0.97, 'Mean={}'.format(round(mean, 2)))
    plt.text((mean + 2*sigma) * 1.01, plt.gca().get_ylim()[1] * 0.95, 'Mean+2std={}'.format(round(mean + 2*sigma, 2)))
    plt.text((mean - 2*sigma) * 1.01, plt.gca().get_ylim()[1] * 0.95, 'Mean-2std={}'.format(round(mean - 2*sigma, 2)))
	
    plt.text(median * 1.01, plt.gca().get_ylim()[1] * 0.90, 'Median={}'.format(round(median, 2)))
    plt.text((median + 2*s_mad) * 1.01, plt.gca().get_ylim()[1] * 0.93, 'Median+2s_mad={}'.format(round(median + 2*s_mad, 2)))
    plt.text((median - 2*s_mad) * 1.01, plt.gca().get_ylim()[1] * 0.93, 'Median-2s_mad={}'.format(round(median - 2*s_mad, 2)))

    # Get data
    print("Feature characteristics for {}:".format(colName))
    print("Min value = ", min(df[colName]))
    print("Max value = ", max(df[colName]))
    print("Mean = ", round(mean, 2))
    print("Median =", median)
    print("Standard deviation =", round(sigma, 4))
    print("Skew =", round(skew, 4))
    print("kurtosis =", round(kurtosis, 4))

    return fig	
	

def plotBinaryValues(df_dig, cols):
    #Plot binary values including NaN
    figlen = np.int(len(cols)/2)+1
    
    fig = plt.figure(figsize=(8, figlen))

    N = len(df_dig)

    for x, f in enumerate(cols):

        pos = (df_dig[f] == 1).sum()
        neg = (df_dig[f] == 0).sum()
        nanValue = (df_dig[f].isna()).sum()

        plt.barh([x], [pos], color='#81C784')
        plt.barh([x], [N-pos], left=pos,  color='#f1f442')
        plt.barh([x], [N-pos-nanValue], left=pos+nanValue,  color='#FFCDD2')
        #Plot Percentage
        plt.text(pos / 2, x, '{:.0f}%'.format(pos / N * 100), horizontalalignment='center', verticalalignment='center')
        plt.text(pos + (N - pos - neg) / 2, x, '{:.0f}%'.format(nanValue / N * 100), horizontalalignment='center', verticalalignment='center')
        plt.text(pos + nanValue + (N - pos - nanValue) / 2, x, '{:.0f}%'.format(neg / N * 100), horizontalalignment='center', verticalalignment='center')

    plt.yticks(range(len(cols)), ['{}'.format(f) for f in cols])
    plt.legend(['Value = 1', 'Value=NaN', 'Value = 0'])
    plt.title('Binary Features')
    plt.xlabel('Number of rows')

    plt.tight_layout()
    
    return fig
	
def plot_confusion_matrix_multiclass(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
	
def plotUmap(embedding, y, classList, title, cmapString='RdYlGn'):
    fig, ax = plt.subplots(1, figsize=(14, 10))
    #plt.scatter(*embedding.T, s=0.3, c=y, cmap='Spectral', alpha=1.0)
    plt.scatter(*embedding.T, s=0.3, c=y, cmap=cmapString, alpha=1.0)
    plt.setp(ax, xticks=[], yticks=[])
    #cbar = plt.colorbar(boundaries=np.arange(11)-0.5)
    cbar = plt.colorbar(boundaries=np.arange(len(classList)+1)-0.5)
    #cbar.set_ticks(np.arange(10))
    cbar.set_ticks(np.arange(len(classList)))
    cbar.set_ticklabels(classList)
    plt.title(title);
	
def amplifyForPlot(binaryArray, targetArray, distance):
    return binaryArray * targetArray * (1-distance)

def plot_three_class_graph(y_class, y_ref, y_time, offset1, offset2, offset3, legend, title="Prediction Results", save_fig_prefix=None):
    
    y0 = (y_class==0)*1
    y1 = (y_class==1)*1
    y2 = (y_class==2)*1
    
    plot_data_OK = amplifyForPlot(y0, y_ref, offset1)
    plot_data_blim = amplifyForPlot(y1, y_ref, offset2)
    plot_data_tlim = amplifyForPlot(y2, y_ref, offset3)
    
    # Plot test data
    plt.figure(num=None, figsize=(11.5, 7), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(y_time, y_ref)
    plt.plot(y_time, plot_data_OK, color='grey')
    plt.plot(y_time, plot_data_blim, color='green')
    plt.plot(y_time, plot_data_tlim, color='red')
    plt.title(title)
    plt.ylim([np.min(y_ref)*0.99999, np.max(y_ref)*1.00002])
    plt.grid()
    plt.legend(legend)

    if save_fig_prefix:
        if not os.path.isdir(save_fig_prefix):
            os.makedirs(save_fig_prefix)
        plt.savefig(os.path.join(save_fig_prefix, title + '_3class'), dpi=300)

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


# %% pycharm={"is_executing": false}
def plot_two_class_graph(binclass, y_ref, y_time, offset_binclass, legend, title="Prediction Results", save_fig_prefix=None):
    plt.figure(num=None, figsize=(12.5, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(y_time, y_ref)
    plt.plot(y_time, amplifyForPlot(binclass, y_ref, offset_binclass), color='orange')
    #plt.title(conf['source_path'])
    plt.title(title)
    plt.ylim([np.min(y_ref)*0.99999, np.max(y_ref)*1.00002])
    plt.grid()
    plt.legend(legend)

    if save_fig_prefix:
        if not os.path.isdir(save_fig_prefix):
            os.makedirs(save_fig_prefix)
        plt.savefig(os.path.join(save_fig_prefix, title + '_3class'), dpi=300)

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

#def plot_two_class_graph(y_order_train, y_order_train_pred):
#    plt.figure(num=None, figsize=(11.5, 7), dpi=80, facecolor='w', edgecolor='k')
#    plt.plot(df_timegraph['Date'][y_order_train.index], df_timegraph['Close'][y_order_train.index])
#    plt.plot(df_timegraph['Date'][y_order_train.index],
#             vis.amplifyForPlot(y_order_train['y'].values, df_timegraph['Close'][y_order_train.index].values, 0.00),
#             color='green')
#    plt.plot(df_timegraph['Date'][y_order_train.index],
#             vis.amplifyForPlot(y_order_train_pred['y'].values, df_timegraph['Close'][y_order_train.index].values,
#                              0.00), color='yellow')
#    plt.title("Plot results")
#    plt.show()

def plot_grid_search_validation_curve(grid, param_to_vary, refit_scorer_name, title='Validation Curve', ylim=None, xlim=None, log=None):
    """Plots train and cross-validation scores from a GridSearchCV instance's
    best params while varying one of those params.

    This method does not work
    """
    #https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results

    df_cv_results = pd.DataFrame(grid.cv_results_)
    train_scores_mean = df_cv_results['mean_train_' + refit_scorer_name]
    valid_scores_mean = df_cv_results['mean_test_' + refit_scorer_name]
    train_scores_std = df_cv_results['std_train_' + refit_scorer_name]
    valid_scores_std = df_cv_results['std_test_' + refit_scorer_name]

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
    param_ranges_lengths = [len(pr) for pr in param_ranges]

    train_scores_mean = np.array(train_scores_mean).reshape(*param_ranges_lengths)
    valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)
    train_scores_std = np.array(train_scores_std).reshape(*param_ranges_lengths)
    valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)

    param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))

    slices = []
    for idx, param in enumerate(grid.best_params_):
        if (idx == param_to_vary_idx):
            slices.append(slice(None))
            continue
        best_param_val = grid.best_params_[param]
        idx_of_best_param = 0
        if isinstance(param_ranges[idx], np.ndarray):
            idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
        else:
            idx_of_best_param = param_ranges[idx].index(best_param_val)
        slices.append(idx_of_best_param)

    train_scores_mean = train_scores_mean[tuple(slices)]
    valid_scores_mean = valid_scores_mean[tuple(slices)]
    train_scores_std = train_scores_std[tuple(slices)]
    valid_scores_std = valid_scores_std[tuple(slices)]

    plt.figure(figsize=(5,5))
    plt.clf()

    plt.title(title)
    plt.xlabel(param_to_vary)
    plt.ylabel('Score')

    if (ylim is None):
        plt.ylim(0.0, 1.1)
    else:
        plt.ylim(*ylim)

    if (not (xlim is None)):
        plt.xlim(*xlim)

    lw = 2

    plot_fn = plt.plot
    if log:
        plot_fn = plt.semilogx

    param_range = param_ranges[param_to_vary_idx]
    #if (not isinstance(param_range[0], numbers.Number)):
    #    param_range = [str(x) for x in param_range]
    plot_fn(param_range, train_scores_mean, label='Training score', color='r',
            lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r', lw=lw)
    plot_fn(param_range, valid_scores_mean, label='Cross-validation score',
            color='b', lw=lw)
    plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1,
                     color='b', lw=lw)

    plt.legend(loc='lower right')
    plt.grid()

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


def plot_heatmap_xy(scores, parameters, xlabel, ylabel, title, normalizeScale=False):
    '''Plot heat map for 2 variables'''
    # Source of inspiration: https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html

    # Plot a heatmap of 2 variables
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True, sharey=True)
    # fig = plt.figure(figsize=(7,5), constrained_layout=True)
    # ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    # ax=axs[0]

    # ax=plt.gca()
    colorMap = plt.cm.bone  # plt.cm.gist_gray #plt.cm.hot

    if normalizeScale == True:
        im1 = plt.imshow(scores, interpolation='catrom', cmap=colorMap, vmin=0, vmax=1)
    else:
        # im1 = plt.imshow(scores, interpolation='nearest', cmap=colorMap)
        im1 = plt.imshow(scores, interpolation='catrom', origin='lower', cmap=colorMap)

    levels = np.linspace(np.min(scores), np.max(scores), 20)

    # contours = plt.contour(scores, 10, colors='black')
    # contours = plt.contour(scores, 10, colors='black')
    contours = ax.contourf(scores, levels=levels, cmap=plt.cm.bone)

    ax.contour(scores, levels=levels, colors='k', linestyles='solid', alpha=1, linewidths=.5, antialiased=True)

    # plt.contourf =
    ax.clabel(contours, inline=True, fontsize=8, colors='r')

    # Get best value
    def get_Top_n_values_from_array(arr, n):
        result = []
        for i in range(1, n + 1):
            x = np.partition(arr.flatten(), -2)[-i]
            r = np.where(arr == x)
            # print(r[0][0], r[1][0])
            value = [r[0][0], r[1][0]]
            result.append(value)
        return result

    bestvalues = get_Top_n_values_from_array(scores, 10)
    [plt.plot(pos[1], pos[0], 'o', markersize=12, fillstyle='none', c='r', mew=3, label="best_value") for pos in
     bestvalues]
    # best = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    # print(best)
    # plt.plot(parameters[xlabel][best[1]], parameters[ylabel][best[0]], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="value1")
    # plt.plot(3, 8, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="value1")

    plt.sca(ax)
    plt.xticks(np.arange(len(parameters[xlabel])), ['{:.1E}'.format(x) for x in parameters[xlabel]],
               rotation='vertical')
    plt.yticks(np.arange(len(parameters[ylabel])), ['{:.1E}'.format(x) for x in parameters[ylabel]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, len(parameters[xlabel]) - 1)
    plt.ylim(0, len(parameters[ylabel]) - 1)
    ax.set_title(title)

    # cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.8])
    cbar = fig.colorbar(im1, ax=ax)  # cax=axs[1])
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

# #%matplotlib inline
# For an identical distribution, we cannot reject the null hypothesis since the p-value is high, 41%. To reject the null
# hypothesis, the p value shall be <5%

def calculate_significance_matrix(parameter_name, unique_param_values, results, refit_scorer_name, alpha_limit=0.05):
    '''Function that calculate a matrix for the significance of the inputs with the Computes the
    Kolmogorov-Smirnov statistic on 2 samples and plots it.

    For an identical distribution, we cannot reject the null hypothesis since the p-value is high. The null hypothesis
    is that some distributions belong to the same distribution. A high p value -> much overlap in the distributions.
    To reject the null hypothesis, the p value shall be <5%

    :name: Parameter name e.g. scaler
    :param_values: Unique values of the measured feature
    :results: Grid search results
    :refit_scorer_name: Refit scorer name
    :alpha_limit: p value: default 0.05. If values < 0.05, then the result is 1 and the distributions are not part of the
    same distribution

    '''

    print(unique_param_values)
    print(parameter_name)
    significance_calculations = pd.DataFrame(index=unique_param_values, columns=unique_param_values)

    for i in significance_calculations:
        for j in significance_calculations.columns:
            p0 = results[results['param_' + parameter_name].astype(str) == str(i)]['mean_test_' + refit_scorer_name]
            p1 = results[results['param_' + parameter_name].astype(str) == str(j)]['mean_test_' + refit_scorer_name]
            if (len(p0) == 0 or len(p1) == 0):
                significance_calculations[j].loc[i] = 0
            else:
                significance_calculations[j].loc[i] = ks_2samp(p0, p1).pvalue

    label = list(map(str, unique_param_values))
    label = list(map(lambda x: x[0:20], label))  # param_values#[str(t)[:9] for t in merged_params_run1[name]]

    index = label
    cols = label
    statistics_df = pd.DataFrame(significance_calculations.values < alpha_limit, index=index, columns=cols)

    fig = plt.Figure()
    sns.heatmap(statistics_df, cmap='Blues', linewidths=0.5, annot=True, vmin=0, vmax=1)
    plt.title("Statistical Difference Significance Matrix for " + parameter_name)

    fig = plt.gcf()

    return statistics_df, fig

def plotOverlayedHistorgrams(parameter_name, unique_param_values, results, median_results, refit_scorer_name):
    '''Plot layered histograms from feature distributions

    :parameter_name: Parameter name e.g. scaler
    :unique_param_values: Unique values of the measured feature
    :results: Grid search results
    :median_results: median results of the distributions
    :refit_scorer_name: Refit scorer name

    :return: figure

    '''
    #min_range = np.percentile(results['mean_test_' + refit_scorer_name], 25)  #25% percentile
    min_range = np.min(results['mean_test_' + refit_scorer_name])

    #median_result = dict()

    plt.figure(figsize=(12, 8))
    max_counts = 0

    for i in unique_param_values:
        #print(i)
        p0 = results[results['param_' + parameter_name].astype(str) == str(i)]['mean_test_' + refit_scorer_name]
        if (len(p0) > 0):
            bins = 100
            counts, _ = np.histogram(p0, bins=bins, range=(min_range, 1))
            if np.max(counts) > max_counts:
                max_counts = np.max(counts)
            #print(counts)
            #median_hist = np.round(np.percentile(p0, 50), 3)
            #median_result[i] = median_hist
            median_hist = median_results[i]
            s = str(i).split('(', 1)[0] + ": " + str(median_hist)
            label = str(i).split('(', 1)[0]

            plt.hist(p0,
                     alpha=0.5,
                     bins=bins,
                     range=(min_range, 1),
                     label=label)
            plt.vlines(median_hist, 0, np.max(counts))
            plt.text(median_hist, np.max(counts) + 1, s, fontsize=12)
            #print("Median for {}:{}".format(s, median_hist))
        else:
            print("No results for ", i)
        #plt.hist(p1, alpha=0.5)
    plt.legend(loc='upper left')
    plt.title("Distribution of different {}".format(parameter_name))
    plt.xlabel('Test {}'.format(refit_scorer_name))
    plt.ylabel('Number of occurances')
    plt.ylim([0, max_counts+2])

    fig = plt.gcf()

    return fig

def visualize_parameter_grid_search(param_name, search_cv_parameters, search_cv_results, refit_scorer_name,
                                    save_fig_prefix=None):
    '''
    Create visualizations and data for a certain grid search parameter.

    :param_name:
    :search_cv_parameters:
    :search_cv_results:
    :refit_scorer_name:
    :save_fig_prefix: Save figures in the folder specified in the path and file prefix if it is set and not none

    :return:

    '''

    # Get all values from all keys scaler in a list
    #sublist = [x[name] for x in params_run1] # Get a list of lists with all values from all keys
    #flatten = lambda l: [item for sublist in l for item in sublist]  # Lambda flatten function
    unique_list = sup.get_unique_values_from_list_of_dicts(param_name, search_cv_parameters) #Get unique values of list by converting it into a set and then to list

    #Get list of all scalers and their indices

    #indexList = [list(results_run1[results_run1['param_' + name].astype(str) == i].index) for i in unique_list]

    #indexList = [
    #    results_run1.loc[results_run1['param_' + name] == results_run1['param_' + name].unique()[i]].iloc[0, :].name for
    #    i in unique_list]
    print("Plot best {} values".format(param_name))
    #display(results_run1.loc[indexList[0]].round(3))

    # number of results to consider
    #number_results = np.int(results_run1.shape[0] * 1.00)
    #print("The top 10% of the results are used, i.e {} samples".format(number_results))

    # Bar chart for categorical values
    # Create series to display
    hist_label = search_cv_results['param_' + param_name].astype(str)  # .apply(str).apply(lambda x: x[:20])
    source = hist_label.value_counts() / search_cv_results.shape[0]
    # Chart
    fig1 = paintBarChartForCategorical(source, title='Distribution of Labels for Feature <{}>'.format(source.name))
    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_' + param_name + '_categorical_bar', dpi=300)
    #plt.ion()
    plt.show(block=False)
    #plt.show()
    plt.pause(0.1)
    plt.close()

    # Significance matrix for distributions
    significance_matrix, fig2 = calculate_significance_matrix(param_name, unique_list, search_cv_results, refit_scorer_name)
    plt.tight_layout()
    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_' + param_name + '_significance_matrix', dpi=300)
    #plt.ion()
    plt.show(block=False)
    #plt.show()
    plt.pause(0.1)
    plt.close()

    # Overlayed histograms
    medians = sup.get_median_values_from_distributions(param_name, unique_list, search_cv_results, refit_scorer_name)
    fig3 = plotOverlayedHistorgrams(param_name, unique_list, search_cv_results, medians, refit_scorer_name)
    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_' + param_name + '_overlayed_histograms', dpi=300)
    #plt.ion()
    plt.show(block=False)
    #plt.show()
    plt.pause(0.1)
    plt.close()

    print("Method end")

    return significance_matrix, medians

def plot_precision_recall_evaluation(y_trainsub, y_trainsub_pred, y_trainsub_pred_proba, reduced_class_dict, save_fig_prefix=None):
    # Calculate Precision, Recall, Accuracy, F1, Confusion Matrix, ROC
    np.set_printoptions(precision=3)

    print("Accuracy: ", accuracy_score(y_trainsub, y_trainsub_pred))
    report = classification_report(y_trainsub, y_trainsub_pred, target_names=list(reduced_class_dict.values()), output_dict=True)
    df = pd.DataFrame(report).transpose()
    if save_fig_prefix != None:
        df.to_csv(save_fig_prefix + '_classification_report.csv')

    print(report)

    cnf_matrix = confusion_matrix(y_trainsub, y_trainsub_pred, labels=list(reduced_class_dict.keys()))

    # Plot non-normalized confusion matrix
    #matrixSize = len(y_classes) * 2
    # plt.figure(figsize=(matrixSize, matrixSize))
    fig_confusion_matrix = plot_confusion_matrix_multiclass(cnf_matrix, classes=list(reduced_class_dict.values()),
                                             title='Confusion matrix with normalization', normalize=True)

    if save_fig_prefix != None:
        fig_confusion_matrix.savefig(save_fig_prefix + '_confusion_matrix', dpi=300)

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    fig_pr_curve = skplt.metrics.plot_precision_recall_curve(np.array(y_trainsub), np.array(y_trainsub_pred_proba), figsize=(10, 10))
    if save_fig_prefix != None:
        fig_pr_curve.figure.savefig(save_fig_prefix + '_pr_curve')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    fig_roc_curve = skplt.metrics.plot_roc(np.array(y_trainsub), np.array(y_trainsub_pred_proba), figsize=(10, 10))
    if save_fig_prefix != None:
        fig_roc_curve.figure.savefig(save_fig_prefix + '_roc_curve')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

#    if save_fig_prefix != None:
#        fig_confusion_matrix.savefig(save_fig_prefix + '_confusion_matrix', dpi=300)
#        #plt.savefig(save_fig_prefix + '_confusion_matrix', dpi=300)
#        fig_pr_curve.figure.savefig(save_fig_prefix + '_pr_curve')
#        #plt.savefig(save_fig_prefix + '_pr_curve', dpi=300)
#        #plt.savefig(save_fig_prefix + '_roc_curve', dpi=300)
#        fig_roc_curve.figure.savefig(save_fig_prefix + '_roc_curve')



def precision_recall_threshold(y_adjusted_classes, y_test, p, r, thresholds, t, save_fig_prefix=None):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = y_adjusted_classes #adjusted_classes(y_scores, t)
    print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))

    # plot the curve
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.fill_between(r, p, step='post', alpha=0.2, color='b')
    plt.ylim([0.5, 1.01]);
    plt.xlim([0.5, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)
    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_pr_adjusted')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, optimal_threshold, save_fig_prefix=None):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89

    #https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    close_default_clf = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.plot(optimal_threshold, precisions[close_default_clf], 'o', markersize = 12, fillstyle = 'none', c='r', mew=3, label="Optimal threshold")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.grid()

    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_pr_scores_of_decision_threshold')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def plot_roc_curve(fpr, tpr, label=None, save_fig_prefix=None):
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    #https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')

    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_roc_curve')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def plot_decision_boundary(X, y, model, save_fig_prefix=None):
    X_Train_embedded = TSNE(n_components=2).fit_transform(X)
    print(X_Train_embedded.shape)
    # model = LogisticRegression().fit(X,y)
    #model = optclf
    y_predicted = model.predict(X)
    # replace the above by your data and model

    # create meshgrid
    resolution = 100  # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:, 0]), np.max(X_Train_embedded[:, 0])
    X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:, 1]), np.max(X_Train_embedded[:, 1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    # plot
    plt.figure(figsize=(11.5, 7))
    plt.contourf(xx, yy, voronoiBackground)
    plt.scatter(X_Train_embedded[:, 0], X_Train_embedded[:, 1], c=y)
    plt.title("Decision Boundary Plot Projected")

    if save_fig_prefix != None:
        plt.savefig(save_fig_prefix + '_decision_boundary_plot')

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def plot_autocorrelation(feature, title, mode=None, lags=None, xlim=None, ylim=None, image_save_directory=None):
    '''
    Plot autocorrelations or partial autocorrelations

    '''

    if mode=="pacf":
        plot_pacf(feature, lags=lags)
        plt.title("Partial Autocorrelation of " + title + ", lags=" + str(lags))
        savepath = os.path.join(image_save_directory, "AC_" + title + "_lags_" + str(lags))
    else:
        plot_acf(feature, lags=lags)
        plt.title("Autocorrelation of " + title + ", lags=" + str(lags))
        savepath = os.path.join(image_save_directory, "PAC_" + title + "_lags_" + str(lags))

    print("=====================================")
    print("Title: ", plt.gcf().axes[0].get_title())


    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if image_save_directory:
        if not os.path.isdir(image_save_directory):
            os.makedirs(image_save_directory)
        plt.savefig(savepath)

    print(
        "Plot the total autocorrelation of the price.The dark blue values are "
        "the correlation of the price with the lag." +
        "The light blue cone is the confidence interval. If the correlation is > cone, "
        "the value is significant.")

    print("Ljung-Box statistics: p-value=", acorr_ljungbox(feature, lags=None, boxpierce="Ljung-Box")[1])
    print("If p values > 0.05 then there are significant autocorrelations.")

    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def plot_temporal_correlation_feature(X_scaled, dataset_name, image_save_directory, source, y_scaled):
    '''

    https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9

    '''

    window_size = 90
    # Only for time series continuous correlation
    # Check correlations for the following attributes
    for col in X_scaled.columns:
        # cols = ['MA100Norm', 'RSI20']
        # col_index = [X_scaled.columns.get_loc(c) for c in cols if c in X_scaled]
        col_index = [X_scaled.columns.get_loc(col)]

        #FIXME: Drop columns to well specified
        tmp = X_scaled.iloc[:, col_index].join(source['Date']).join(y_scaled).reset_index().set_index('Date').drop(
            columns=['id'])
        # display(tmp)
        #tmp.dropna().resample('Q').apply(lambda x: x.corr(method='spearman')).iloc[:, -1].unstack().iloc[:, :-1].plot(
        #    title='Correlation of Features to Outcome', figsize=(8, 2))
        fig = tmp.dropna().interpolate()[tmp.columns[0]].rolling(window=window_size, center=True).corr(tmp[tmp.columns[-1]]).plot(
            title='Correlation of Feature ' + tmp.columns[0] + ' to Outcome ' + tmp.columns[-1], figsize=(8, 2))
        plt.ylim(-1, 1)
        plt.hlines([-0.2, 0.2], xmin=tmp.index[0], xmax=tmp.index[-1], colors='r')

        if image_save_directory:
            if not os.path.isdir(image_save_directory):
                os.makedirs(image_save_directory)
            plt.savefig(os.path.join(image_save_directory, 'corr_{}_outcome_{}'.format(col, tmp.columns[-1])), dpi=300)

        plt.show(block=False)
        plt.pause(0.1)
        plt.close()