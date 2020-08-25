import data_visualization_functions as vis

import numpy as np



existing_classes = ['a', 'b', 'c']
cnf_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vis.plot_confusion_matrix_multiclass(cnf_matrix, classes=existing_classes, title='Confusion matrix with normalization', normalize=True)
