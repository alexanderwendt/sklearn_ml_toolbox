import matplotlib.pyplot as plt
import numpy as np

#Generate Scatter plot with results
def visualize_random_search_results(random_search, refit_scorer_name, xlim=None, ylim=None):
    '''
    Generate a 2D scatter plot with results for SVM



    '''
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

    if xlim:
        ax.set_xlim([np.min(x), np.max(x)])
    else:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim([np.min(y), np.max(y)])
    else:
        ax.set_ylim(ylim)
    plt.grid(True)
    plt.colorbar(sc)

    return ax


def add_best_results_to_random_search_visualization(ax_random_search, results, plot_best):
    '''
    Plot random search results and add the best

    :args:
        :ax_random_search: Prepared heatmap of random search results
        :results: random search results
        :plot_best: top x values to be highlighted in the plot

    :return:
        :ax_random_search: enhanced heatmap of random search result including highlighted top values

    '''
    # Plot results
    #ax = visualize_random_search_results(random_search_run, refit_scorer_name)

    # Add plot for best x values from results
    [ax_random_search.plot(p[0], p[1], 'o', markersize=12, fillstyle='none', c='r', mew=3, label="best_value") for p in
     zip(results['param_svm__C'].head(plot_best).values, results['param_svm__gamma'].head(plot_best).values)]
    ax_random_search.set_ylabel("gamma")
    ax_random_search.set_xlabel("C")

    # Visualize the difference between training and test results for C and gamma
    # print(random_search_run)
    # plot_grid_search_validation_curve(random_search_run, 'svm__C', refit_scorer_name, log=True, ylim=(0.50, 1.01))
    # plot_grid_search_validation_curve(random_search_run, 'svm__gamma', refit_scorer_name, log=True, ylim=(0.50, 1.01))

    #plt.show()
    return ax_random_search