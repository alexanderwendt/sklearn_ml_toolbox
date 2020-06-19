import matplotlib.pyplot as plt
import numpy as np

#Generate Scatter plot with results
def visualize_random_search_results(random_search, refit_scorer_name):
    '''Generate a 2D scatter plot with results for SVM'''
    cols = random_search.cv_results_['mean_test_' + refit_scorer_name]
    x = random_search.cv_results_['param_svm__C']
    y = random_search.cv_results_['param_svm__gamma']

    fig = plt.figure()
    ax = plt.gca()
    sc = ax.scatter(x=x,
                    y=y,
                    s=50,
                    c=cols,
                    alpha=0.5,
                    edgecolors='none',
                    cmap=plt.cm.bone)
    #ax.pcolormesh(x, y, cols, cmap=plt.cm.BuGn_r)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(y), np.max(y)])
    plt.grid(True)
    plt.colorbar(sc)

    return ax