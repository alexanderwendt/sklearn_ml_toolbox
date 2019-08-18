import numpy as np
import matplotlib.pyplot as plt
import matplotlib as m
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from statsmodels import robust
import itertools

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
    plt.show()

def paintBarChartForCategorical(xlabels, yvalues):
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

    barProportions = [calculateProportion(i, 0, max(yvalues)) for i in yvalues]
    barColors = colorMap(barProportions)

    # Create a color bar
    cscalar = m.cm.ScalarMappable(cmap=colorMap, norm=plt.Normalize(0,max(yvalues)))
    cscalar.set_array([])
    #colorbar = plt.colorbar(cscalar, orientation='vertical', ticks=[0, 0.05, max(yvalues)])
    #colorbar.set_label('Low and high values', rotation=270,labelpad=25)

    #fix xlabel
    xlabel_string = list(map(str, xlabels))
    xlabel_string = list(map(lambda x: x[0:20], xlabel_string))

    #Plot
    plt.xlabel("Feature Label")
    plt.ylabel('Share of Feature Labels')
    plt.xticks(rotation=90)
    #plt.gca().yaxis.set_ticks(np.arange(0, 0.1, 0.005))
    plt.title('Distribution of Labels for Feature <{}>'.format(yvalues.name))
    #plt.gca().set_ylim([0, 0.1])
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3, left=0.1)
    bar = plt.bar(xlabel_string, yvalues, color=barColors, width=1.0, capsize=10, edgecolor='black')
    plt.show()


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
    plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
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
    plt.show()

    # Get data
    print("Feature characteristics:")
    print("Min value = ", min(df[colName]))
    print("Max value = ", max(df[colName]))
    print("Mean = ", round(mean, 2))
    print("Median =", median)
    print("Standard deviation =", round(sigma, 4))
    print("Skew =", round(skew, 4))
    print("kurtosis =", round(kurtosis, 4))

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